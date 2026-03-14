"""
optimize_prompt.py

Small-scale prompt refinement / evaluation loop for MCAT generation prompts.

Purpose:
- Sample a small subset of topics from topics.json
- Generate candidate outputs with one or more prompt variants
- Critique the generated outputs using a second evaluation prompt
- Save structured results to JSON for inspection before bulk generation

Important:
- This is for prompt optimization, not for large-scale bank generation.
- It is intentionally small-batch and inspection-friendly.
- Backend is swappable via backend.py.

Typical usage:

Mock backend sanity check:
    python optimize_prompt.py --backend mock --sample-size 8

OpenAI-compatible backend (e.g. vLLM serve):
    python optimize_prompt.py \
        --backend openai_compat \
        --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --sample-size 12
"""

from __future__ import annotations

import argparse
from statistics import mean
from typing import Any, Optional

from backend import GenerationConfig, build_backend
from prompt_templates import (
    CRITIQUE_PROMPT_VERSION,
    build_generation_critique_prompt,
    build_generation_prompt,
    get_schema_hint_for_mode,
)
from utils import (
    atomic_write_json,
    is_cars_topic,
    load_topics,
    sample_topics,
    validate_bank_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def infer_mode(topic: dict[str, Any]) -> str:
    return "cars" if is_cars_topic(topic) else "science"


def parse_overall_score(obj: dict[str, Any]) -> Optional[float]:
    value = obj.get("overall_score")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    generated_ok = [r for r in results if r.get("generation_success")]
    critique_ok = [r for r in results if r.get("critique_success")]

    scores = []
    for r in critique_ok:
        critique = r.get("critique_parsed")
        if isinstance(critique, dict):
            s = parse_overall_score(critique)
            if s is not None:
                scores.append(s)

    by_mode: dict[str, list[dict[str, Any]]] = {"science": [], "cars": []}
    for r in results:
        by_mode.setdefault(r["mode"], []).append(r)

    mode_summary: dict[str, Any] = {}
    for mode, rows in by_mode.items():
        row_scores = []
        for r in rows:
            critique = r.get("critique_parsed")
            if isinstance(critique, dict):
                s = parse_overall_score(critique)
                if s is not None:
                    row_scores.append(s)

        mode_summary[mode] = {
            "count": len(rows),
            "generation_successes": sum(1 for r in rows if r.get("generation_success")),
            "critique_successes": sum(1 for r in rows if r.get("critique_success")),
            "average_score": round(mean(row_scores), 3) if row_scores else None,
        }

    return {
        "num_topics_evaluated": len(results),
        "num_generation_successes": len(generated_ok),
        "num_critique_successes": len(critique_ok),
        "average_overall_score": round(mean(scores), 3) if scores else None,
        "mode_summary": mode_summary,
    }


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def evaluate_topic(
    backend,
    topic: dict[str, Any],
    science_difficulty: str,
    cars_difficulty: str,
    cars_num_questions: int,
    cars_passage_word_target: int,
) -> dict[str, Any]:
    mode = infer_mode(topic)
    difficulty = cars_difficulty if mode == "cars" else science_difficulty

    generation_prompt = build_generation_prompt(
        topic=topic,
        mode=mode,
        difficulty=difficulty,
        num_questions=cars_num_questions,
        passage_word_target=cars_passage_word_target,
    )

    generation_schema_hint = get_schema_hint_for_mode(mode)
    generation_result = backend.generate_json(generation_prompt, schema_hint=generation_schema_hint)

    row: dict[str, Any] = {
        "topic_id": topic["topic_id"],
        "mode": mode,
        "generation_success": generation_result.success,
        "generation_error": generation_result.error,
        "generation_raw_text": generation_result.raw_text,
        "generation_parsed": generation_result.parsed,
        "generation_schema_valid": False,
        "critique_success": False,
        "critique_error": None,
        "critique_raw_text": None,
        "critique_parsed": None,
    }

    if not generation_result.success or not isinstance(generation_result.parsed, dict):
        return row

    validation = validate_bank_entry(generation_result.parsed)
    row["generation_schema_valid"] = validation.ok
    row["generation_schema_errors"] = validation.errors

    critique_prompt = build_generation_critique_prompt(
        topic=topic,
        generated_object=generation_result.parsed,
        mode=mode,
    )

    critique_schema_hint = """
{
  "overall_score": "integer 1-10",
  "verdict": "keep|revise|reject",
  "strengths": ["string"],
  "weaknesses": ["string"],
  "schema_issues": ["string"],
  "content_issues": ["string"],
  "difficulty_assessment": "string",
  "realism_assessment": "string",
  "recommended_prompt_changes": ["string"]
}
""".strip()

    critique_result = backend.generate_json(critique_prompt, schema_hint=critique_schema_hint)

    row["critique_success"] = critique_result.success
    row["critique_error"] = critique_result.error
    row["critique_raw_text"] = critique_result.raw_text
    row["critique_parsed"] = critique_result.parsed

    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small-scale prompt optimization loop for MCAT generation."
    )

    parser.add_argument("--topics", type=str, default="topics.json", help="Path to topics.json")
    parser.add_argument(
        "--output",
        type=str,
        default="data/prompt_eval_results.json",
        help="Output JSON file for prompt evaluation results",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "openai_compat"],
        help="Backend type",
    )
    parser.add_argument("--model", type=str, default="mock-backend", help="Model name")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible backend",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key")

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of topics to sample for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for topic sampling",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable science/CARS stratified topic sampling",
    )

    parser.add_argument(
        "--science-difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Target difficulty for science prompt eval",
    )
    parser.add_argument(
        "--cars-difficulty",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Target difficulty for CARS prompt eval",
    )
    parser.add_argument(
        "--cars-num-questions",
        type=int,
        default=4,
        help="Questions per CARS set during eval",
    )
    parser.add_argument(
        "--cars-passage-word-target",
        type=int,
        default=450,
        help="Approximate passage length during eval",
    )

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    topics = load_topics(args.topics)
    sampled = sample_topics(
        topics=topics,
        n=args.sample_size,
        seed=args.seed,
        stratify_cars=not args.no_stratify,
    )

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
        config=config,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    print("=" * 72)
    print("Prompt optimization run")
    print("=" * 72)
    print(f"topics file:   {args.topics}")
    print(f"sample size:   {len(sampled)}")
    print(f"backend:       {args.backend}")
    print(f"model:         {args.model}")
    print(f"output:        {args.output}")
    print("=" * 72)

    results: list[dict[str, Any]] = []

    for i, topic in enumerate(sampled, start=1):
        print(f"[{i}/{len(sampled)}] Evaluating {topic['topic_id']} ({infer_mode(topic)})")
        row = evaluate_topic(
            backend=backend,
            topic=topic,
            science_difficulty=args.science_difficulty,
            cars_difficulty=args.cars_difficulty,
            cars_num_questions=args.cars_num_questions,
            cars_passage_word_target=args.cars_passage_word_target,
        )
        results.append(row)

        critique = row.get("critique_parsed")
        if isinstance(critique, dict):
            print(
                f"    verdict={critique.get('verdict')} "
                f"score={critique.get('overall_score')}"
            )
        elif row.get("generation_success"):
            print("    generated ok; critique unavailable")
        else:
            print(f"    generation failed: {row.get('generation_error')}")

    summary = summarize_results(results)

    payload = {
        "meta": {
            "topics_path": args.topics,
            "backend": args.backend,
            "model": args.model,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "science_difficulty": args.science_difficulty,
            "cars_difficulty": args.cars_difficulty,
            "cars_num_questions": args.cars_num_questions,
            "cars_passage_word_target": args.cars_passage_word_target,
            "critique_prompt_version": CRITIQUE_PROMPT_VERSION,
        },
        "summary": summary,
        "results": results,
    }

    atomic_write_json(args.output, payload, indent=2)

    print("=" * 72)
    print("Prompt optimization complete")
    print(f"generation successes: {summary['num_generation_successes']}")
    print(f"critique successes:   {summary['num_critique_successes']}")
    print(f"average score:        {summary['average_overall_score']}")
    print(f"results saved to:     {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()