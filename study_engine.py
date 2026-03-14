"""
study_engine.py

Terminal-based adaptive MCAT study engine.

Reads:
- mcat_topics.json
- cleaned question bank JSONL
- user_progress.json

Writes:
- updated user_progress.json after each completed topic/session item

Design goals:
- zero inference at study time
- adaptive topic selection using:
    - adaptive_score
    - spacing / last_seen
- supports both science and CARS
- avoids immediate topic repeats when possible
- saves progress after each completed topic
- clean terminal UX

Scoring:
- y = correct
- p = partial
- n = wrong
- q = quit

Adaptive score update logic is implemented in utils.py:
- correct => divide by 3
- partial => mild increase
- wrong => multiply by 2

Typical usage:
    python study_engine.py

    python study_engine.py --mcat-topics mcat_topics.json \
        --bank data/question_bank_clean.jsonl \
        --progress data/user_progress.json

    python study_engine.py --mode science --num-rounds 20
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from typing import Any, Optional

from utils import (
    apply_progress_update,
    clamp,
    compute_topic_weight,
    is_cars_topic,
    iter_jsonl,
    load_progress,
    load_topics,
    save_progress,
    truncate,
    weighted_choice,
)


# ---------------------------------------------------------------------------
# Loading / indexing
# ---------------------------------------------------------------------------


def load_bank_index(
    bank_path: str,
) -> dict[str, list[dict[str, Any]]]:
    """
    Load cleaned bank JSONL and index by topic_id.

    Returns:
        {
            "TOPIC_ID": [entry1, entry2, ...],
            ...
        }
    """
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0

    for entry in iter_jsonl(bank_path, skip_bad=False):
        topic_id = str(entry.get("topic_id", "")).strip()
        if not topic_id:
            continue
        by_topic[topic_id].append(entry)
        count += 1

    if count == 0:
        raise ValueError(
            f"No usable entries found in bank file: {bank_path}. "
            "Run generate_bank.py and clean_bank.py first."
        )

    return dict(by_topic)


def ensure_progress_covers_topics(
    topics: list[dict[str, Any]],
    progress: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Ensure every topic has a progress record.

    This lets the study engine recover gracefully if new topics were added
    after the progress file was created.
    """
    out = dict(progress)
    for topic in topics:
        topic_id = str(topic["topic_id"])
        if topic_id not in out:
            initial_score = topic.get("adaptive_score", 1.0)
            try:
                initial_score = float(initial_score)
            except (TypeError, ValueError):
                initial_score = 1.0

            out[topic_id] = {
                "topic_id": topic_id,
                "adaptive_score": initial_score,
                "last_seen": None,
                "times_seen": 0,
                "times_correct": 0,
                "times_partial": 0,
                "times_wrong": 0,
            }
    return out


# ---------------------------------------------------------------------------
# Topic / item selection
# ---------------------------------------------------------------------------


def filter_topics_for_mode(
    topics: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """Restrict topics by requested study mode."""
    mode = mode.strip().lower()
    if mode == "all":
        return topics
    if mode == "science":
        return [t for t in topics if not is_cars_topic(t)]
    if mode == "cars":
        return [t for t in topics if is_cars_topic(t)]
    raise ValueError(f"Unsupported mode: {mode}")


def filter_topics_with_bank(
    topics: list[dict[str, Any]],
    bank_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Keep only topics that actually have at least one bank entry."""
    return [t for t in topics if str(t["topic_id"]) in bank_index and bank_index[str(t["topic_id"])]]


def select_topic(
    topics: list[dict[str, Any]],
    progress: dict[str, dict[str, Any]],
    previous_topic_id: Optional[str],
    rng: random.Random,
) -> dict[str, Any]:
    """
    Select a topic adaptively using progress weights.

    Avoid immediate repetition when possible.
    """
    candidate_topics = list(topics)

    if previous_topic_id is not None and len(candidate_topics) > 1:
        non_repeat = [t for t in candidate_topics if str(t["topic_id"]) != previous_topic_id]
        if non_repeat:
            candidate_topics = non_repeat

    weights: list[float] = []
    for topic in candidate_topics:
        topic_id = str(topic["topic_id"])
        record = progress[topic_id]
        weights.append(compute_topic_weight(record))

    chosen = weighted_choice(candidate_topics, weights, rng=rng)
    return chosen


def select_entry_for_topic(
    topic_id: str,
    bank_index: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    previous_entry_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Choose one bank entry for a topic.

    Avoid repeating the immediately previous entry if possible.
    """
    entries = list(bank_index[topic_id])
    if not entries:
        raise ValueError(f"No bank entries found for topic_id={topic_id}")

    def entry_key(entry: dict[str, Any]) -> str:
        if entry.get("mode") == "science":
            return str(entry.get("question_id", ""))
        return str(entry.get("question_set_id", ""))

    if previous_entry_key is not None and len(entries) > 1:
        filtered = [e for e in entries if entry_key(e) != previous_entry_key]
        if filtered:
            entries = filtered

    return rng.choice(entries)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def hr(width: int = 72) -> str:
    return "=" * width


def print_topic_header(topic: dict[str, Any], record: dict[str, Any], round_num: int) -> None:
    """Display a concise topic header."""
    print()
    print(hr())
    print(f"Round {round_num}")
    print(hr())
    print(f"Topic ID:       {topic.get('topic_id', '')}")
    print(f"Category:       {topic.get('category', '')}")
    print(f"Subcategory:    {topic.get('subcategory', '')}")
    print(f"Title:          {topic.get('title', '')}")
    print(f"Adaptive score: {record.get('adaptive_score', 1.0):.3f}")
    print(
        "Stats:          "
        f"seen={record.get('times_seen', 0)} | "
        f"correct={record.get('times_correct', 0)} | "
        f"partial={record.get('times_partial', 0)} | "
        f"wrong={record.get('times_wrong', 0)}"
    )
    print(hr())


def print_science_entry(entry: dict[str, Any]) -> None:
    """Display one science question."""
    print(f"Difficulty: {entry.get('difficulty', '')}")
    print()
    print(entry.get("question", "").strip())
    print()
    options = entry.get("options", {})
    for key in ["A", "B", "C", "D"]:
        print(f"{key}. {options.get(key, '')}")
    print()


def print_cars_entry(entry: dict[str, Any]) -> None:
    """Display one CARS passage and its questions."""
    print(f"Difficulty: {entry.get('difficulty', '')}")
    print()
    print("PASSAGE")
    print("-" * 72)
    print(entry.get("passage", "").strip())
    print("-" * 72)
    print()


def print_explanation(title: str, explanation: str) -> None:
    print(f"{title}")
    print("-" * len(title))
    print(explanation.strip())
    print()


def normalize_grade_input(value: str) -> str:
    """Map user input to y / p / n / q if possible."""
    value = value.strip().lower()
    if value in {"y", "yes", "c", "correct"}:
        return "y"
    if value in {"p", "partial"}:
        return "p"
    if value in {"n", "no", "wrong", "w"}:
        return "n"
    if value in {"q", "quit", "exit"}:
        return "q"
    return ""


def prompt_for_mc_answer() -> str:
    """Ask user for A/B/C/D or q."""
    while True:
        value = input("Your answer (A/B/C/D, or q to quit): ").strip().upper()
        if value in {"A", "B", "C", "D", "Q"}:
            return value
        print("Please enter A, B, C, D, or q.")


def prompt_for_grade() -> str:
    """Ask user to self-grade y/p/n/q."""
    while True:
        value = input("Grade yourself: [y] correct / [p] partial / [n] wrong / [q] quit: ")
        normalized = normalize_grade_input(value)
        if normalized:
            return normalized
        print("Please enter y, p, n, or q.")


# ---------------------------------------------------------------------------
# Science study flow
# ---------------------------------------------------------------------------


def run_science_round(entry: dict[str, Any]) -> str:
    """
    Run one science question and return y/p/n/q.
    """
    print_science_entry(entry)

    user_answer = prompt_for_mc_answer()
    if user_answer == "Q":
        return "q"

    correct = str(entry.get("correct_answer", "")).upper().strip()
    options = entry.get("options", {})

    print()
    if user_answer == correct:
        print(f"You chose {user_answer}. Correct.")
    else:
        print(f"You chose {user_answer}. Correct answer: {correct}")
        if correct in options:
            print(f"Correct option text: {options[correct]}")
    print()

    print_explanation("Explanation", str(entry.get("explanation", "")))

    while True:
        grade = input(
            "Final grade for this topic: [y] correct / [p] partial / [n] wrong / [q] quit: "
        )
        normalized = normalize_grade_input(grade)
        if normalized:
            return normalized
        print("Please enter y, p, n, or q.")


# ---------------------------------------------------------------------------
# CARS study flow
# ---------------------------------------------------------------------------


def cars_question_fraction_to_outcome(score_fraction: float) -> str:
    """
    Convert average CARS performance into topic-level outcome.
    """
    if score_fraction >= 0.999:
        return "y"
    if score_fraction >= 0.5:
        return "p"
    return "n"


def run_cars_round(entry: dict[str, Any]) -> str:
    """
    Run one CARS set and return topic-level y/p/n/q.

    Per-question scoring:
    - y => 1.0
    - p => 0.5
    - n => 0.0
    Final topic-level outcome is based on average.
    """
    print_cars_entry(entry)

    questions = entry.get("questions", [])
    if not isinstance(questions, list) or not questions:
        print("This CARS set is malformed: no questions found.")
        return "n"

    numeric_scores: list[float] = []

    for i, q in enumerate(questions, start=1):
        print(hr())
        print(f"CARS Question {i}/{len(questions)}")
        print(hr())
        print(q.get("question", "").strip())
        print()

        options = q.get("options", {})
        for key in ["A", "B", "C", "D"]:
            print(f"{key}. {options.get(key, '')}")
        print()

        user_answer = prompt_for_mc_answer()
        if user_answer == "Q":
            return "q"

        correct = str(q.get("correct_answer", "")).upper().strip()
        print()
        if user_answer == correct:
            print(f"You chose {user_answer}. Correct.")
        else:
            print(f"You chose {user_answer}. Correct answer: {correct}")
            if correct in options:
                print(f"Correct option text: {options[correct]}")
        print()

        print_explanation("Explanation", str(q.get("explanation", "")))

        while True:
            grade = input(
                "Grade this question: [y] correct / [p] partial / [n] wrong / [q] quit: "
            )
            normalized = normalize_grade_input(grade)
            if normalized == "q":
                return "q"
            if normalized == "y":
                numeric_scores.append(1.0)
                break
            if normalized == "p":
                numeric_scores.append(0.5)
                break
            if normalized == "n":
                numeric_scores.append(0.0)
                break
            print("Please enter y, p, n, or q.")

    avg = sum(numeric_scores) / len(numeric_scores)
    final_outcome = cars_question_fraction_to_outcome(avg)

    print(hr())
    print(f"CARS set summary: average score = {avg:.2f} -> topic outcome = {final_outcome}")
    print(hr())

    return final_outcome


# ---------------------------------------------------------------------------
# Session stats
# ---------------------------------------------------------------------------


def empty_session_stats() -> dict[str, Any]:
    return {
        "rounds_completed": 0,
        "topics_seen": 0,
        "science_rounds": 0,
        "cars_rounds": 0,
        "correct": 0,
        "partial": 0,
        "wrong": 0,
    }


def update_session_stats(stats: dict[str, Any], entry_mode: str, outcome: str) -> None:
    stats["rounds_completed"] += 1
    stats["topics_seen"] += 1

    if entry_mode == "science":
        stats["science_rounds"] += 1
    elif entry_mode == "cars":
        stats["cars_rounds"] += 1

    if outcome == "y":
        stats["correct"] += 1
    elif outcome == "p":
        stats["partial"] += 1
    elif outcome == "n":
        stats["wrong"] += 1


def print_session_summary(stats: dict[str, Any]) -> None:
    print()
    print(hr())
    print("Session summary")
    print(hr())
    print(f"Rounds completed: {stats['rounds_completed']}")
    print(f"Science rounds:   {stats['science_rounds']}")
    print(f"CARS rounds:      {stats['cars_rounds']}")
    print(f"Correct:          {stats['correct']}")
    print(f"Partial:          {stats['partial']}")
    print(f"Wrong:            {stats['wrong']}")
    print(hr())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive terminal study engine for a pre-generated MCAT question bank."
    )
    parser.add_argument("--mcat-topics", type=str, default="mcat_topics.json", help="Path to mcat_topics.json")
    parser.add_argument(
        "--bank",
        type=str,
        default="data/question_bank_clean.jsonl",
        help="Path to cleaned question bank JSONL",
    )
    parser.add_argument(
        "--progress",
        type=str,
        default="data/user_progress.json",
        help="Path to user progress JSON",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "science", "cars"],
        help="Restrict session to science, cars, or all",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Maximum number of rounds to run this session",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for topic/item selection",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    topics = load_topics(args.mcat_topics)
    bank_index = load_bank_index(args.bank)
    progress = load_progress(args.progress)
    progress = ensure_progress_covers_topics(topics, progress)

    filtered_topics = filter_topics_for_mode(topics, args.mode)
    filtered_topics = filter_topics_with_bank(filtered_topics, bank_index)

    if not filtered_topics:
        raise ValueError(
            "No topics available for study after applying mode filter and bank availability."
        )

    print(hr())
    print("Adaptive MCAT Study Engine")
    print(hr())
    print(f"Topics loaded:     {len(topics)}")
    print(f"Study candidates:  {len(filtered_topics)}")
    print(f"Mode:              {args.mode}")
    print(f"Max rounds:        {args.num_rounds}")
    print(hr())

    stats = empty_session_stats()
    previous_topic_id: Optional[str] = None
    previous_entry_key: Optional[str] = None

    for round_num in range(1, args.num_rounds + 1):
        topic = select_topic(
            topics=filtered_topics,
            progress=progress,
            previous_topic_id=previous_topic_id,
            rng=rng,
        )
        topic_id = str(topic["topic_id"])
        record = progress[topic_id]

        entry = select_entry_for_topic(
            topic_id=topic_id,
            bank_index=bank_index,
            rng=rng,
            previous_entry_key=previous_entry_key,
        )

        print_topic_header(topic, record, round_num=round_num)

        mode = str(entry.get("mode", "")).strip().lower()
        if mode == "science":
            outcome = run_science_round(entry)
            previous_entry_key = str(entry.get("question_id", ""))
        elif mode == "cars":
            outcome = run_cars_round(entry)
            previous_entry_key = str(entry.get("question_set_id", ""))
        else:
            print(f"Skipping malformed entry with unsupported mode: {mode}")
            continue

        if outcome == "q":
            print("\nExiting study session.")
            break

        progress[topic_id] = apply_progress_update(record, outcome=outcome)
        save_progress(args.progress, progress)

        new_score = float(progress[topic_id].get("adaptive_score", 1.0))
        print(
            f"Saved progress for {topic_id} | outcome={outcome} | new adaptive_score={new_score:.3f}"
        )

        update_session_stats(stats, entry_mode=mode, outcome=outcome)
        previous_topic_id = topic_id

    print_session_summary(stats)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)