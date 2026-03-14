"""
init_progress.py

Initialize user progress for the adaptive MCAT study engine.

Reads:
- mcat_topics.json

Writes:
- user_progress.json

Design goals:
- one progress record per topic_id
- safe atomic save
- preserve existing progress unless --overwrite is used
- optionally merge in new topics without wiping old stats

Default progress fields:
- topic_id
- adaptive_score
- last_seen
- times_seen
- times_correct
- times_partial
- times_wrong

Typical usage:
    python init_progress.py

    python init_progress.py --mcat-topics mcat_topics.json --output data/user_progress.json

    python init_progress.py --merge
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import (
    atomic_write_json,
    load_topics,
    load_progress,
    make_default_progress_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize or merge user progress from mcat_topics.json."
    )
    parser.add_argument(
        "--mcat-topics",
        type=str,
        default="mcat_topics.json",
        help="Path to mcat_topics.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/user_progress.json",
        help="Path to user progress JSON",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing progress file completely",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge missing topics into an existing progress file without resetting old records",
    )
    return parser.parse_args()


def build_progress_from_topics(topics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Create a fresh progress mapping keyed by topic_id."""
    progress: dict[str, dict[str, Any]] = {}
    for topic in topics:
        topic_id = str(topic["topic_id"])
        progress[topic_id] = make_default_progress_record(topic)
    return progress


def merge_progress(
    existing: dict[str, dict[str, Any]],
    topics: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int]:
    """
    Merge new topics into existing progress without altering existing stats.

    Returns:
        (merged_progress, num_added)
    """
    merged = dict(existing)
    num_added = 0

    for topic in topics:
        topic_id = str(topic["topic_id"])
        if topic_id not in merged:
            merged[topic_id] = make_default_progress_record(topic)
            num_added += 1

    return merged, num_added


def main() -> None:
    args = parse_args()

    topics = load_topics(args.mcat_topics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite and not args.merge:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it or --merge to add missing topics."
        )

    if args.overwrite or not output_path.exists():
        progress = build_progress_from_topics(topics)
        atomic_write_json(args.output, progress, indent=2)
        print("=" * 72)
        print("Initialized progress file")
        print(f"topics loaded:   {len(topics)}")
        print(f"records written: {len(progress)}")
        print(f"output:          {args.output}")
        print("=" * 72)
        return

    # merge path
    existing = load_progress(args.output)
    merged, num_added = merge_progress(existing=existing, topics=topics)
    atomic_write_json(args.output, merged, indent=2)

    print("=" * 72)
    print("Merged progress file")
    print(f"topics loaded:        {len(topics)}")
    print(f"existing records:     {len(existing)}")
    print(f"new topics added:     {num_added}")
    print(f"final record count:   {len(merged)}")
    print(f"output:               {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()