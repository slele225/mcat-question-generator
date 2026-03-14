"""
generate_bank.py

Generate an MCAT question bank from topics.json and write incrementally to JSONL.

Design goals:
- Works with either a mock backend or a real OpenAI-compatible backend
- Suitable for append-only bulk generation
- Checkpoints progress so interrupted runs can resume
- Supports both science and CARS topics
- Keeps question bank generation separate from study-time logic

Typical usage examples:

Mock backend test:
    python generate_bank.py --topics topics.json --output data/question_bank.jsonl --backend mock

OpenAI-compatible backend (e.g. vLLM serve):
    python generate_bank.py \
        --topics topics.json \
        --output data/question_bank.jsonl \
        --backend openai_compat \
        --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct

Notes:
- This script writes one JSON object per line to JSONL immediately.
- It also maintains a checkpoint file listing completed generation units.
- For science topics, each requested question counts as one generation unit.
- For CARS topics, each requested set counts as one generation unit.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional

from backend import (
    GenerationConfig,
    GenerationResult,
    build_backend,
    normalize_cars_output,
    normalize_science_output,
)
from prompt_templates import (
    CARS_PROMPT_VERSION,
    SCIENCE_PROMPT_VERSION,
    build_generation_prompt,
    get_schema_hint_for_mode,
)
from utils import (
    append_jsonl,
    atomic_write_json,
    is_cars_topic,
    load_json,
    load_topics,
    make_question_id,
    make_question_set_id,
    normalize_text,
    safe_get,
    stable_hash,
    truncate,
    validate_bank_entry,
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """
    Load checkpoint state.

    Structure:
    {
      "completed_units": ["topic_id::science::0", ...],
      "meta": {...}
    }
    """
    path = Path(path)
    if not path.exists():
        return {"completed_units": [], "meta": {}}

    raw = load_json(path, default={})
    if not isinstance(raw, dict):
        return {"completed_units": [], "meta": {}}

    completed_units = raw.get("completed_units", [])
    meta = raw.get("meta", {})

    if not isinstance(completed_units, list):
        completed_units = []
    if not isinstance(meta, dict):
        meta = {}

    return {
        "completed_units": completed_units,
        "meta": meta,
    }


def save_checkpoint(path: str | Path, completed_units: set[str], meta: dict[str, Any]) -> None:
    """Persist checkpoint state atomically."""
    payload = {
        "completed_units": sorted(completed_units),
        "meta": meta,
    }
    atomic_write_json(path, payload, indent=2)


def make_generation_unit_id(topic_id: str, mode: str, index: int) -> str:
    """Stable identifier for one generation attempt/unit."""
    return f"{topic_id}::{mode}::{index}"


# ---------------------------------------------------------------------------
# Prompt / topic planning
# ---------------------------------------------------------------------------


def plan_generation_units(
    topics: list[dict[str, Any]],
    science_per_topic: int,
    cars_sets_per_topic: int,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Expand topics into concrete generation units.

    Example output entry:
    {
      "unit_id": "BB_123::science::0",
      "topic": {...},
      "mode": "science",
      "index": 0
    }
    """
    units: list[dict[str, Any]] = []

    for topic in topics:
        topic_id = str(topic["topic_id"])
        if is_cars_topic(topic):
            for i in range(cars_sets_per_topic):
                units.append(
                    {
                        "unit_id": make_generation_unit_id(topic_id, "cars", i),
                        "topic": topic,
                        "mode": "cars",
                        "index": i,
                    }
                )
        else:
            for i in range(science_per_topic):
                units.append(
                    {
                        "unit_id": make_generation_unit_id(topic_id, "science", i),
                        "topic": topic,
                        "mode": "science",
                        "index": i,
                    }
                )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(units)

    return units


# ---------------------------------------------------------------------------
# Output normalization / rescue
# ---------------------------------------------------------------------------


def infer_mode_from_topic(topic: dict[str, Any]) -> str:
    """Infer canonical generation mode from the topic."""
    return "cars" if is_cars_topic(topic) else "science"


def ensure_science_ids(obj: dict[str, Any], topic_id: str, fallback_suffix: str = "") -> dict[str, Any]:
    """Fill missing science identifiers."""
    question = str(obj.get("question", "")).strip()
    if not obj.get("question_id"):
        if question:
            obj["question_id"] = make_question_id(topic_id, question)
        else:
            obj["question_id"] = f"q_{stable_hash(topic_id + fallback_suffix, length=16)}"
    return obj


def ensure_cars_ids(obj: dict[str, Any], topic_id: str, fallback_suffix: str = "") -> dict[str, Any]:
    """Fill missing CARS identifiers."""
    passage = str(obj.get("passage", "")).strip()
    if not obj.get("question_set_id"):
        if passage:
            obj["question_set_id"] = make_question_set_id(topic_id, passage)
        else:
            obj["question_set_id"] = f"qs_{stable_hash(topic_id + fallback_suffix, length=16)}"
    return obj


def coerce_to_single_object(parsed: Any, mode: str) -> Optional[dict[str, Any]]:
    """
    Coerce backend output into the expected single-object form.

    Accepts:
    - dict directly
    - single-item list containing dict
    """
    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        return parsed[0]

    return None


def rescue_and_normalize_output(
    result: GenerationResult,
    topic: dict[str, Any],
    mode: str,
    fallback_suffix: str,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """
    Normalize backend output into one or more canonical bank entries.

    Returns:
        (normalized_objects, error_message)

    For science:
        accepts multi-item outputs and returns multiple normalized entries

    For CARS:
        returns exactly one normalized entry in a list
    """
    if not result.success or result.parsed is None:
        return [], result.error or "Backend generation failed"

    topic_id = str(topic["topic_id"])

    if mode == "science":
        items = extract_science_items(result.parsed)
        if items is None or not items:
            return [], "Parsed science backend output was not a valid item list"

        normalized_items: list[dict[str, Any]] = []

        for idx, obj in enumerate(items):
            obj = dict(obj)
            obj["topic_id"] = topic_id
            obj["mode"] = "science"

            normalized = normalize_science_output(
                obj=obj,
                model_name=result.model,
                prompt_version=SCIENCE_PROMPT_VERSION,
            )
            normalized = ensure_science_ids(
                normalized,
                topic_id,
                fallback_suffix=f"{fallback_suffix}::{idx}",
            )

            validation = validate_bank_entry(normalized)
            if not validation.ok:
                return [], f"Science item {idx} failed validation: {'; '.join(validation.errors)}"

            normalized_items.append(normalized)

        return normalized_items, None

    if mode == "cars":
        obj = coerce_to_single_object(result.parsed, mode=mode)
        if obj is None:
            return [], "Parsed CARS backend output was not a single JSON object"

        obj = dict(obj)
        obj["topic_id"] = topic_id
        obj["mode"] = "cars"

        normalized = normalize_cars_output(
            obj=obj,
            model_name=result.model,
            prompt_version=CARS_PROMPT_VERSION,
        )
        normalized = ensure_cars_ids(normalized, topic_id, fallback_suffix=fallback_suffix)

        validation = validate_bank_entry(normalized)
        if not validation.ok:
            return [], "; ".join(validation.errors)

        return [normalized], None

    return [], f"Unsupported mode: {mode}"


def extract_science_items(parsed: Any) -> Optional[list[dict[str, Any]]]:
    """
    Accept any of these science output shapes:
    - {"items": [ ... ]}
    - [ ... ]
    - single dict
    """
    if isinstance(parsed, dict):
        items = parsed.get("items")
        if isinstance(items, list) and all(isinstance(x, dict) for x in items):
            return items
        return [parsed]

    if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
        return parsed

    return None


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def print_run_header(args: argparse.Namespace, total_units: int) -> None:
    """Print a concise run header."""
    print("=" * 72)
    print("MCAT question bank generation")
    print("=" * 72)
    print(f"topics file:      {args.topics}")
    print(f"output file:      {args.output}")
    print(f"checkpoint file:  {args.checkpoint}")
    print(f"backend:          {args.backend}")
    print(f"model:            {args.model}")
    print(f"science/topic:    {args.science_per_topic}")
    print(f"cars sets/topic:  {args.cars_sets_per_topic}")
    print(f"batch size:       {args.batch_size}")
    print(f"shuffle:          {args.shuffle}")
    print(f"total units:      {total_units}")
    print(f"science num questions: {args.science_num_questions}")
    print("=" * 72)


def print_unit_failure(unit: dict[str, Any], error: str) -> None:
    """Print a concise failure line."""
    topic_id = unit["topic"]["topic_id"]
    mode = unit["mode"]
    index = unit["index"]
    print(f"[FAIL] {topic_id} {mode} #{index}: {truncate(error, 240)}")


def print_unit_success(unit: dict[str, Any], obj: dict[str, Any]) -> None:
    """Print a concise success line."""
    topic_id = unit["topic"]["topic_id"]
    mode = unit["mode"]
    index = unit["index"]
    if mode == "science":
        desc = truncate(str(obj.get("question", "")), 100)
        identifier = obj.get("question_id", "")
    else:
        desc = truncate(str(obj.get("passage", "")), 100)
        identifier = obj.get("question_set_id", "")
    print(f"[OK]   {topic_id} {mode} #{index} -> {identifier} | {desc}")


# ---------------------------------------------------------------------------
# Core batch generation
# ---------------------------------------------------------------------------


def generate_batch_for_units(
    backend,
    units: list[dict[str, Any]],
    science_difficulty: str,
    cars_difficulty: str,
    cars_num_questions: int,
    cars_passage_word_target: int,
    science_num_questions: int,
) -> list[tuple[dict[str, Any], GenerationResult, str]]:
    """
    Generate one backend result per unit.

    Returns a list of:
        (unit, result, schema_hint)
    """
    prompts: list[str] = []
    schema_hints: list[str] = []

    for unit in units:
        mode = unit["mode"]
        topic = unit["topic"]

        if mode == "science":
            difficulty = science_difficulty
        else:
            difficulty = cars_difficulty

        prompt = build_generation_prompt(
            topic=topic,
            mode=mode,
            difficulty=difficulty,
            num_questions=cars_num_questions,
            passage_word_target=cars_passage_word_target,
            science_num_questions=science_num_questions,
        )
        schema_hint = get_schema_hint_for_mode(mode)

        prompts.append(prompt)
        schema_hints.append(schema_hint)

    # Backend interface supports one schema_hint for batch; prompts already include schema.
    results = backend.generate_json_batch(prompts, schema_hint=None)

    if len(results) != len(units):
        raise RuntimeError(
            f"Backend returned {len(results)} results for {len(units)} prompts"
        )

    return list(zip(units, results, schema_hints))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an MCAT question bank from topics.json."
    )

    parser.add_argument("--topics", type=str, default="topics.json", help="Path to topics.json")
    parser.add_argument(
        "--output",
        type=str,
        default="data/question_bank.jsonl",
        help="Path to append-only raw question bank JSONL",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/generation_checkpoint.json",
        help="Path to checkpoint JSON",
    )
    parser.add_argument(
        "--fail-log",
        type=str,
        default="data/generation_failures.jsonl",
        help="Path to JSONL file containing failed generations",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "openai_compat"],
        help="Generation backend type",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock-backend",
        help="Model name for the backend",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible backend, e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for OpenAI-compatible backend",
    )

    parser.add_argument(
        "--science-per-topic",
        type=int,
        default=3,
        help="Number of science questions to generate per science topic",
    )
    parser.add_argument(
        "--cars-sets-per-topic",
        type=int,
        default=2,
        help="Number of CARS sets to generate per CARS topic",
    )
    parser.add_argument(
        "--cars-num-questions",
        type=int,
        default=4,
        help="Number of questions per generated CARS set",
    )
    parser.add_argument(
        "--cars-passage-word-target",
        type=int,
        default=450,
        help="Approximate target passage length for CARS generation",
    )

    parser.add_argument(
        "--science-difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Nominal science difficulty target",
    )
    parser.add_argument(
        "--cars-difficulty",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Nominal CARS difficulty target",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of generation units to send through the backend per loop",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle generation units before processing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling",
    )

    parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Optional cap on total generation units processed this run",
    )
    parser.add_argument(
        "--topic-id",
        type=str,
        default=None,
        help="Optional single topic_id to generate for",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoint and regenerate already-completed units",
    )

    parser.add_argument("--temperature", type=float, default=0.7, help="Backend temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Backend top_p")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Backend max tokens")
    parser.add_argument("--timeout-seconds", type=int, default=180, help="Backend timeout")
    parser.add_argument("--retries", type=int, default=2, help="Backend retries")
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Linear retry backoff in seconds",
    )
    parser.add_argument(
        "--science-num-questions",
        type=int,
        default=5,
        help="Number of science questions to request per science generation prompt",
    )

    return parser.parse_args()


def filter_topics(topics: list[dict[str, Any]], topic_id: Optional[str]) -> list[dict[str, Any]]:
    """Optionally restrict to one topic_id."""
    if not topic_id:
        return topics
    filtered = [t for t in topics if str(t["topic_id"]) == str(topic_id)]
    if not filtered:
        raise ValueError(f"No topic found with topic_id={topic_id}")
    return filtered


def build_generation_backend(args: argparse.Namespace):
    """Instantiate backend from CLI args."""
    config = GenerationConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )

    backend = build_backend(
        backend_type=args.backend,
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        config=config,
    )
    return backend


def write_failure_record(
    fail_log_path: str | Path,
    unit: dict[str, Any],
    error: str,
    raw_text: str = "",
    parsed: Any = None,
) -> None:
    """Append one failure record for debugging later."""
    record = {
        "unit_id": unit["unit_id"],
        "topic_id": unit["topic"]["topic_id"],
        "mode": unit["mode"],
        "index": unit["index"],
        "error": error,
        "raw_text": raw_text,
        "parsed": parsed,
    }
    append_jsonl(fail_log_path, record)


def main() -> None:
    args = parse_args()

    topics = load_topics(args.topics)
    topics = filter_topics(topics, args.topic_id)

    units = plan_generation_units(
        topics=topics,
        science_per_topic=args.science_per_topic,
        cars_sets_per_topic=args.cars_sets_per_topic,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    checkpoint = load_checkpoint(args.checkpoint)
    completed_units = set(checkpoint.get("completed_units", []))

    if not args.force:
        units = [u for u in units if u["unit_id"] not in completed_units]

    if args.max_units is not None:
        units = units[: args.max_units]

    print_run_header(args, total_units=len(units))

    if not units:
        print("Nothing to do. All planned units are already completed.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.fail_log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    backend = build_generation_backend(args)

    run_meta = {
        "backend": args.backend,
        "model": args.model,
        "topics_path": args.topics,
        "output_path": args.output,
        "science_per_topic": args.science_per_topic,
        "cars_sets_per_topic": args.cars_sets_per_topic,
        "science_difficulty": args.science_difficulty,
        "cars_difficulty": args.cars_difficulty,
        "science_num_questions": args.science_num_questions,
    }

    num_success = 0
    num_fail = 0

    for start in range(0, len(units), args.batch_size):
        batch_units = units[start : start + args.batch_size]

        batch_results = generate_batch_for_units(
            backend=backend,
            units=batch_units,
            science_difficulty=args.science_difficulty,
            cars_difficulty=args.cars_difficulty,
            cars_num_questions=args.cars_num_questions,
            cars_passage_word_target=args.cars_passage_word_target,
            science_num_questions=args.science_num_questions,
        )

        for unit, result, _schema_hint in batch_results:
            fallback_suffix = f"::{unit['unit_id']}"

            normalized_items, error = rescue_and_normalize_output(
                result=result,
                topic=unit["topic"],
                mode=unit["mode"],
                fallback_suffix=fallback_suffix,
            )

            if not normalized_items:
                num_fail += 1
                print_unit_failure(unit, error or "unknown normalization error")
                write_failure_record(
                    fail_log_path=args.fail_log,
                    unit=unit,
                    error=error or "unknown normalization error",
                    raw_text=result.raw_text,
                    parsed=result.parsed,
                )
                continue

            for item in normalized_items:
                append_jsonl(args.output, item)
                print_unit_success(unit, item)
                num_success += 1

            completed_units.add(unit["unit_id"])


        save_checkpoint(
            path=args.checkpoint,
            completed_units=completed_units,
            meta=run_meta,
        )

        processed = min(start + args.batch_size, len(units))
        print(
            f"[BATCH] processed {processed}/{len(units)} units | "
            f"success={num_success} fail={num_fail}"
        )

    print("=" * 72)
    print("Generation complete")
    print(f"successful writes: {num_success}")
    print(f"failed units:      {num_fail}")
    print(f"output:            {args.output}")
    print(f"checkpoint:        {args.checkpoint}")
    print(f"fail log:          {args.fail_log}")
    print("=" * 72)


if __name__ == "__main__":
    main()