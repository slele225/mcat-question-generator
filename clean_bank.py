"""
clean_bank.py

Validate, deduplicate, and lightly filter a raw MCAT question bank JSONL file.

Inputs:
- raw question bank JSONL (default: data/question_bank.jsonl)

Outputs:
- cleaned question bank JSONL (default: data/question_bank_clean.jsonl)
- bad entries JSONL (optional)
- duplicate entries JSONL (optional)

Design goals:
- Keep heuristics simple and practical
- Preserve valid entries whenever possible
- Remove malformed entries
- Remove near-exact duplicates using stable fingerprints
- Flag low-quality entries using lightweight heuristics

Typical usage:
    python clean_bank.py

    python clean_bank.py \
        --input data/question_bank.jsonl \
        --output data/question_bank_clean.jsonl \
        --bad-output data/bad_entries.jsonl \
        --dup-output data/duplicate_entries.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import (
    append_jsonl,
    fingerprint_question_entry,
    iter_jsonl,
    normalize_text,
    truncate,
    validate_bank_entry,
)


# ---------------------------------------------------------------------------
# Simple quality heuristics
# ---------------------------------------------------------------------------


def word_count(text: str) -> int:
    """Count rough words in a string."""
    return len([x for x in text.strip().split() if x])


def unique_option_count(options: dict[str, str]) -> int:
    """Count unique normalized answer choices."""
    return len({normalize_text(v) for v in options.values()})


def is_placeholder_text(text: str) -> bool:
    """
    Catch obvious placeholder / mock / degenerate text.
    """
    lowered = normalize_text(text)
    bad_markers = [
        "placeholder",
        "lorem ipsum",
        "insert explanation",
        "to be filled",
        "tbd",
        "dummy text",
        "correct interpretation of the tested concept",
        "distractor based on a common conceptual confusion",
    ]
    return any(marker in lowered for marker in bad_markers)


def score_science_quality(entry: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Very simple practical heuristics for science questions.

    These are not meant to be perfect.
    They are only meant to catch obvious junk.
    """
    reasons: list[str] = []

    question = str(entry.get("question", "")).strip()
    explanation = str(entry.get("explanation", "")).strip()
    options = entry.get("options", {})

    if word_count(question) < 8:
        reasons.append("question too short")

    if question.endswith("?") is False:
        reasons.append("question missing terminal question mark")

    if word_count(explanation) < 12:
        reasons.append("explanation too short")

    if is_placeholder_text(question):
        reasons.append("question contains placeholder-like text")

    if is_placeholder_text(explanation):
        reasons.append("explanation contains placeholder-like text")

    if isinstance(options, dict):
        if unique_option_count(options) < 4:
            reasons.append("options are duplicated or nearly duplicated")

        very_short = [k for k, v in options.items() if word_count(str(v)) < 2]
        if very_short:
            reasons.append(f"some options are suspiciously short: {', '.join(very_short)}")

    # Very weak heuristic for answer leakage:
    correct = str(entry.get("correct_answer", "")).strip()
    if correct in {"A", "B", "C", "D"} and isinstance(options, dict):
        correct_text = normalize_text(str(options.get(correct, "")))
        if "correct" in correct_text or "best answer" in correct_text:
            reasons.append("correct option appears to leak answer")

    return (len(reasons) == 0, reasons)


def score_cars_quality(entry: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Simple practical heuristics for CARS sets.
    """
    reasons: list[str] = []

    passage = str(entry.get("passage", "")).strip()
    questions = entry.get("questions", [])

    if word_count(passage) < 180:
        reasons.append("passage too short for realistic CARS")

    if is_placeholder_text(passage):
        reasons.append("passage contains placeholder-like text")

    if not isinstance(questions, list) or len(questions) < 2:
        reasons.append("too few CARS questions")

    if isinstance(questions, list):
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                reasons.append(f"questions[{i}] is not an object")
                continue

            q_text = str(q.get("question", "")).strip()
            expl = str(q.get("explanation", "")).strip()
            opts = q.get("options", {})

            if word_count(q_text) < 5:
                reasons.append(f"questions[{i}] too short")

            if word_count(expl) < 8:
                reasons.append(f"questions[{i}] explanation too short")

            if is_placeholder_text(q_text):
                reasons.append(f"questions[{i}] contains placeholder-like text")

            if isinstance(opts, dict) and unique_option_count(opts) < 4:
                reasons.append(f"questions[{i}] options are duplicated or nearly duplicated")

    return (len(reasons) == 0, reasons)


def assess_quality(entry: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Lightweight quality gate.
    """
    mode = str(entry.get("mode", "")).strip().lower()
    if mode == "science":
        return score_science_quality(entry)
    if mode == "cars":
        return score_cars_quality(entry)
    return False, ["unknown mode"]


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------


def classify_entry(
    entry: dict[str, Any],
    seen_fingerprints: set[str],
    strict_quality: bool,
) -> tuple[str, dict[str, Any]]:
    """
    Classify one entry.

    Returns:
        ("keep" | "bad" | "duplicate", payload)
    """
    validation = validate_bank_entry(entry)
    if not validation.ok:
        payload = {
            "reason": "schema_validation_failed",
            "errors": validation.errors,
            "entry": entry,
        }
        return "bad", payload

    fingerprint = fingerprint_question_entry(entry)
    if fingerprint in seen_fingerprints:
        payload = {
            "reason": "duplicate",
            "fingerprint": fingerprint,
            "entry": entry,
        }
        return "duplicate", payload

    quality_ok, reasons = assess_quality(entry)
    if strict_quality and not quality_ok:
        payload = {
            "reason": "quality_heuristic_failed",
            "errors": reasons,
            "entry": entry,
        }
        return "bad", payload

    seen_fingerprints.add(fingerprint)
    return "keep", entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and deduplicate a raw MCAT question bank JSONL."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/question_bank.jsonl",
        help="Raw question bank JSONL input",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/question_bank_clean.jsonl",
        help="Cleaned question bank JSONL output",
    )
    parser.add_argument(
        "--bad-output",
        type=str,
        default="data/bad_entries.jsonl",
        help="Where to write malformed / rejected entries",
    )
    parser.add_argument(
        "--dup-output",
        type=str,
        default="data/duplicate_entries.jsonl",
        help="Where to write duplicate entries",
    )
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="Reject low-quality heuristic failures instead of only keeping schema-valid items",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files",
    )
    return parser.parse_args()


def ensure_can_write(path: str | Path, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. Use --overwrite to allow."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def print_summary(
    total: int,
    kept: int,
    bad: int,
    dup: int,
    output: str,
    bad_output: str,
    dup_output: str,
) -> None:
    print("=" * 72)
    print("Cleaning complete")
    print(f"total read:        {total}")
    print(f"kept:              {kept}")
    print(f"bad / malformed:   {bad}")
    print(f"duplicates:        {dup}")
    print(f"clean output:      {output}")
    print(f"bad output:        {bad_output}")
    print(f"duplicate output:  {dup_output}")
    print("=" * 72)


def main() -> None:
    args = parse_args()

    ensure_can_write(args.output, overwrite=args.overwrite)
    ensure_can_write(args.bad_output, overwrite=args.overwrite)
    ensure_can_write(args.dup_output, overwrite=args.overwrite)

    # Truncate outputs at start of run
    Path(args.output).write_text("", encoding="utf-8")
    Path(args.bad_output).write_text("", encoding="utf-8")
    Path(args.dup_output).write_text("", encoding="utf-8")

    seen_fingerprints: set[str] = set()

    total = 0
    kept = 0
    bad = 0
    dup = 0

    for entry in iter_jsonl(args.input, skip_bad=False):
        total += 1
        category, payload = classify_entry(
            entry=entry,
            seen_fingerprints=seen_fingerprints,
            strict_quality=args.strict_quality,
        )

        if category == "keep":
            append_jsonl(args.output, payload)
            kept += 1

        elif category == "bad":
            append_jsonl(args.bad_output, payload)
            bad += 1
            reason = payload.get("reason", "bad")
            detail = truncate(" | ".join(payload.get("errors", [])), 200)
            print(f"[BAD] #{total} {reason}: {detail}")

        elif category == "duplicate":
            append_jsonl(args.dup_output, payload)
            dup += 1
            print(f"[DUP] #{total} fingerprint={payload.get('fingerprint', '')}")

        else:
            # Defensive fallback
            append_jsonl(
                args.bad_output,
                {
                    "reason": "unknown_classification",
                    "entry": entry,
                },
            )
            bad += 1

    print_summary(
        total=total,
        kept=kept,
        bad=bad,
        dup=dup,
        output=args.output,
        bad_output=args.bad_output,
        dup_output=args.dup_output,
    )


if __name__ == "__main__":
    main()