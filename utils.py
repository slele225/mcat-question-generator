"""
utils.py

Shared helpers for the MCAT adaptive question bank project.

Design goals:
- Minimal dependencies (standard library only)
- Safe JSON / JSONL file handling
- Reusable validation helpers
- Topic and progress utilities
- Stable IDs and text normalization for deduplication
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


# ---------------------------------------------------------------------------
# Paths / file IO
# ---------------------------------------------------------------------------


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for a file path if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """
    Safely write JSON to disk using a temp file + atomic replace.

    This reduces the chance of corrupting state files like user_progress.json
    if the process is interrupted mid-write.
    """
    path = Path(path)
    ensure_parent_dir(path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=indent)
        tmp.write("\n")
        tmp_path = tmp.name

    os.replace(tmp_path, path)


def load_json(path: str | Path, default: Any = None) -> Any:
    """Load JSON from disk, optionally returning a default if the file is missing."""
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a JSONL file from scratch."""
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, skip_bad: bool = False) -> list[dict[str, Any]]:
    """
    Read a JSONL file into memory.

    If skip_bad=True, malformed lines are ignored.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                elif not skip_bad:
                    raise ValueError(
                        f"Expected JSON object on line {line_number}, got {type(obj).__name__}"
                    )
            except Exception:
                if not skip_bad:
                    raise
    return rows


def iter_jsonl(path: str | Path, skip_bad: bool = False) -> Iterator[dict[str, Any]]:
    """
    Stream a JSONL file one object at a time.

    Useful for large banks.
    """
    path = Path(path)
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
                elif not skip_bad:
                    raise ValueError(
                        f"Expected JSON object on line {line_number}, got {type(obj).__name__}"
                    )
            except Exception:
                if not skip_bad:
                    raise


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO datetime string, returning None on blank input."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Text normalization / hashing
# ---------------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison / deduplication.

    Steps:
    - Unicode normalization
    - lowercase
    - collapse whitespace
    - strip
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def slugify(text: str, max_length: int = 80) -> str:
    """Create a simple slug from text."""
    text = normalize_text(text)
    text = _NON_ALNUM_RE.sub("-", text)
    text = text.strip("-")
    if len(text) > max_length:
        text = text[:max_length].rstrip("-")
    return text or "item"


def stable_hash(payload: str, length: int = 12) -> str:
    """Return a stable short hash for a string payload."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def fingerprint_question_entry(entry: dict[str, Any]) -> str:
    """
    Build a deduplication fingerprint for a bank entry.

    For science:
    - based mainly on topic_id + question text + options

    For CARS:
    - based mainly on topic_id + passage + question texts
    """
    mode = str(entry.get("mode", "")).strip().lower()
    topic_id = str(entry.get("topic_id", "")).strip()

    if mode == "science":
        question = normalize_text(str(entry.get("question", "")))
        options = entry.get("options", {})
        option_blob = "||".join(
            normalize_text(str(options.get(k, ""))) for k in ["A", "B", "C", "D"]
        )
        raw = f"science::{topic_id}::{question}::{option_blob}"
        return stable_hash(raw, length=20)

    if mode == "cars":
        passage = normalize_text(str(entry.get("passage", "")))
        questions = entry.get("questions", [])
        q_blob_parts: list[str] = []
        if isinstance(questions, list):
            for q in questions:
                if isinstance(q, dict):
                    q_text = normalize_text(str(q.get("question", "")))
                    opts = q.get("options", {})
                    opt_blob = "||".join(
                        normalize_text(str(opts.get(k, ""))) for k in ["A", "B", "C", "D"]
                    )
                    q_blob_parts.append(f"{q_text}::{opt_blob}")
        raw = f"cars::{topic_id}::{passage}::{'###'.join(q_blob_parts)}"
        return stable_hash(raw, length=20)

    return stable_hash(json.dumps(entry, sort_keys=True, ensure_ascii=False), length=20)


# ---------------------------------------------------------------------------
# Topic helpers
# ---------------------------------------------------------------------------


REQUIRED_TOPIC_FIELDS = {
    "topic_id",
    "category",
    "subcategory",
    "title",
    "content_to_test",
    "tags",
}


def validate_topic(topic: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a topic entry from topics.json."""
    errors: list[str] = []

    for field in REQUIRED_TOPIC_FIELDS:
        if field not in topic:
            errors.append(f"Missing required field: {field}")

    if "topic_id" in topic and not isinstance(topic["topic_id"], str):
        errors.append("topic_id must be a string")

    if "tags" in topic and not isinstance(topic["tags"], list):
        errors.append("tags must be a list")

    return (len(errors) == 0, errors)


def is_cars_topic(topic: dict[str, Any]) -> bool:
    """
    Heuristic to identify CARS topics.

    Uses category and tags so it works even if formatting varies slightly.
    """
    category = normalize_text(str(topic.get("category", "")))
    title = normalize_text(str(topic.get("title", "")))
    tags = [normalize_text(str(x)) for x in topic.get("tags", [])]

    if "critical analysis and reasoning skills" in category:
        return True
    if "cars" in tags:
        return True
    if title.startswith("cars"):
        return True
    return False


def is_science_topic(topic: dict[str, Any]) -> bool:
    """Inverse helper for non-CARS topics."""
    return not is_cars_topic(topic)


def load_topics(topics_path: str | Path) -> list[dict[str, Any]]:
    """
    Load topics.json and return a list of validated topic dicts.

    Supports either:
    - a top-level list of topics
    - or a dict with a `topics` key
    """
    raw = load_json(topics_path)

    if isinstance(raw, list):
        topics = raw
    elif isinstance(raw, dict) and isinstance(raw.get("topics"), list):
        topics = raw["topics"]
    else:
        raise ValueError(
            "topics.json must be either a list of topic objects or a dict with a 'topics' list"
        )

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, topic in enumerate(topics):
        if not isinstance(topic, dict):
            raise ValueError(f"Topic at index {idx} is not a JSON object")

        ok, errors = validate_topic(topic)
        if not ok:
            joined = "; ".join(errors)
            raise ValueError(f"Invalid topic {topic.get('topic_id', idx)}: {joined}")

        topic_id = topic["topic_id"]
        if topic_id in seen_ids:
            raise ValueError(f"Duplicate topic_id found in topics.json: {topic_id}")
        seen_ids.add(topic_id)
        validated.append(topic)

    return validated


def sample_topics(
    topics: list[dict[str, Any]],
    n: int,
    seed: int = 42,
    stratify_cars: bool = True,
) -> list[dict[str, Any]]:
    """
    Sample topics for prompt optimization or debugging.

    If stratify_cars=True, tries to include both science and CARS topics when possible.
    """
    if n <= 0:
        return []

    if n >= len(topics):
        return list(topics)

    rng = random.Random(seed)

    if not stratify_cars:
        return rng.sample(topics, n)

    cars = [t for t in topics if is_cars_topic(t)]
    science = [t for t in topics if not is_cars_topic(t)]

    if not cars or not science:
        return rng.sample(topics, n)

    target_cars = max(1, round(n * len(cars) / len(topics)))
    target_cars = min(target_cars, len(cars))
    target_science = min(n - target_cars, len(science))

    sampled = rng.sample(cars, target_cars) + rng.sample(science, target_science)

    remaining_needed = n - len(sampled)
    if remaining_needed > 0:
        chosen_ids = {t["topic_id"] for t in sampled}
        remainder = [t for t in topics if t["topic_id"] not in chosen_ids]
        sampled.extend(rng.sample(remainder, remaining_needed))

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def make_question_id(topic_id: str, question_text: str) -> str:
    """Create a stable science question ID."""
    base = f"{topic_id}::{normalize_text(question_text)}"
    return f"q_{stable_hash(base, length=16)}"


def make_question_set_id(topic_id: str, passage_text: str) -> str:
    """Create a stable CARS question-set ID."""
    base = f"{topic_id}::{normalize_text(passage_text)}"
    return f"qs_{stable_hash(base, length=16)}"


# ---------------------------------------------------------------------------
# Validation helpers for generated bank entries
# ---------------------------------------------------------------------------


VALID_ANSWER_KEYS = {"A", "B", "C", "D"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]


def _validate_options(options: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(options, dict):
        return ["options must be an object with keys A, B, C, D"]

    keys = set(options.keys())
    if keys != VALID_ANSWER_KEYS:
        errors.append("options must contain exactly keys A, B, C, D")

    for k in VALID_ANSWER_KEYS:
        if not isinstance(options.get(k), str) or not options.get(k, "").strip():
            errors.append(f"option {k} must be a non-empty string")
    return errors


def validate_science_entry(entry: dict[str, Any]) -> ValidationResult:
    """Validate one science question entry."""
    errors: list[str] = []

    required = [
        "question_id",
        "topic_id",
        "mode",
        "question",
        "options",
        "correct_answer",
        "explanation",
        "difficulty",
        "model",
        "prompt_version",
    ]
    for field in required:
        if field not in entry:
            errors.append(f"Missing required field: {field}")

    if entry.get("mode") != "science":
        errors.append("mode must be 'science'")

    if not isinstance(entry.get("question"), str) or not entry.get("question", "").strip():
        errors.append("question must be a non-empty string")

    errors.extend(_validate_options(entry.get("options")))

    correct_answer = entry.get("correct_answer")
    if correct_answer not in VALID_ANSWER_KEYS:
        errors.append("correct_answer must be one of A, B, C, D")

    if not isinstance(entry.get("explanation"), str) or not entry.get("explanation", "").strip():
        errors.append("explanation must be a non-empty string")

    if entry.get("difficulty") not in VALID_DIFFICULTIES:
        errors.append("difficulty must be one of easy, medium, hard")

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def validate_cars_entry(entry: dict[str, Any]) -> ValidationResult:
    """Validate one CARS question set entry."""
    errors: list[str] = []

    required = [
        "question_set_id",
        "topic_id",
        "mode",
        "passage",
        "questions",
        "difficulty",
        "model",
        "prompt_version",
    ]
    for field in required:
        if field not in entry:
            errors.append(f"Missing required field: {field}")

    if entry.get("mode") != "cars":
        errors.append("mode must be 'cars'")

    if not isinstance(entry.get("passage"), str) or not entry.get("passage", "").strip():
        errors.append("passage must be a non-empty string")

    questions = entry.get("questions")
    if not isinstance(questions, list) or not questions:
        errors.append("questions must be a non-empty list")
    else:
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                errors.append(f"questions[{i}] must be an object")
                continue

            if not isinstance(q.get("question"), str) or not q.get("question", "").strip():
                errors.append(f"questions[{i}].question must be a non-empty string")

            option_errors = _validate_options(q.get("options"))
            errors.extend([f"questions[{i}].{e}" for e in option_errors])

            if q.get("correct_answer") not in VALID_ANSWER_KEYS:
                errors.append(f"questions[{i}].correct_answer must be one of A, B, C, D")

            if not isinstance(q.get("explanation"), str) or not q.get("explanation", "").strip():
                errors.append(f"questions[{i}].explanation must be a non-empty string")

    if entry.get("difficulty") not in VALID_DIFFICULTIES:
        errors.append("difficulty must be one of easy, medium, hard")

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def validate_bank_entry(entry: dict[str, Any]) -> ValidationResult:
    """Validate either a science or CARS bank entry."""
    mode = entry.get("mode")
    if mode == "science":
        return validate_science_entry(entry)
    if mode == "cars":
        return validate_cars_entry(entry)
    return ValidationResult(ok=False, errors=["mode must be either 'science' or 'cars'"])


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def make_default_progress_record(topic: dict[str, Any]) -> dict[str, Any]:
    """
    Build the initial progress record for a topic.

    adaptive_score defaults to topic adaptive_score if present, else 1.0.
    """
    initial_score = topic.get("adaptive_score", 1.0)
    try:
        initial_score = float(initial_score)
    except (TypeError, ValueError):
        initial_score = 1.0

    return {
        "topic_id": topic["topic_id"],
        "adaptive_score": initial_score,
        "last_seen": None,
        "times_seen": 0,
        "times_correct": 0,
        "times_partial": 0,
        "times_wrong": 0,
    }


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into [minimum, maximum]."""
    return max(minimum, min(maximum, value))


def update_adaptive_score(
    current_score: float,
    outcome: str,
    min_score: float = 0.05,
    max_score: float = 100.0,
) -> float:
    """
    Update adaptive score using the requested behavior.

    Rules:
    - correct: divide by 3
    - partial: mild increase
    - wrong: multiply by 2
    """
    outcome = outcome.lower().strip()

    if outcome == "y":
        new_score = current_score / 3.0
    elif outcome == "p":
        new_score = current_score * 1.15
    elif outcome == "n":
        new_score = current_score * 2.0
    else:
        new_score = current_score

    return clamp(new_score, min_score, max_score)


def apply_progress_update(
    record: dict[str, Any],
    outcome: str,
    seen_at: Optional[str] = None,
    min_score: float = 0.05,
    max_score: float = 100.0,
) -> dict[str, Any]:
    """
    Return an updated copy of a topic progress record.
    """
    updated = dict(record)
    updated["times_seen"] = int(updated.get("times_seen", 0)) + 1

    outcome = outcome.lower().strip()
    if outcome == "y":
        updated["times_correct"] = int(updated.get("times_correct", 0)) + 1
    elif outcome == "p":
        updated["times_partial"] = int(updated.get("times_partial", 0)) + 1
    elif outcome == "n":
        updated["times_wrong"] = int(updated.get("times_wrong", 0)) + 1

    current_score = float(updated.get("adaptive_score", 1.0))
    updated["adaptive_score"] = update_adaptive_score(
        current_score=current_score,
        outcome=outcome,
        min_score=min_score,
        max_score=max_score,
    )
    updated["last_seen"] = seen_at or utc_now_iso()
    return updated


def load_progress(progress_path: str | Path) -> dict[str, dict[str, Any]]:
    """
    Load user progress as a dict keyed by topic_id.

    Supports either:
    - {"TOPIC_ID": {...}, ...}
    - {"topics": {"TOPIC_ID": {...}, ...}}
    """
    raw = load_json(progress_path, default={})

    if isinstance(raw, dict) and "topics" in raw and isinstance(raw["topics"], dict):
        raw = raw["topics"]

    if not isinstance(raw, dict):
        raise ValueError("user_progress.json must be a dict or a dict containing a 'topics' dict")

    out: dict[str, dict[str, Any]] = {}
    for topic_id, record in raw.items():
        if isinstance(record, dict):
            out[str(topic_id)] = record
    return out


def save_progress(progress_path: str | Path, progress: dict[str, dict[str, Any]]) -> None:
    """Save user progress atomically."""
    atomic_write_json(progress_path, progress, indent=2)


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def hours_since_last_seen(last_seen: Optional[str]) -> float:
    """
    Return hours since last seen.

    If never seen or unparsable, returns a large number so the topic gets some
    spacing priority.
    """
    dt = parse_iso_datetime(last_seen)
    if dt is None:
        return 1e6

    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.utcnow()
    delta = now - dt.replace(tzinfo=None) if dt.tzinfo is None else now - dt
    return max(delta.total_seconds() / 3600.0, 0.0)


def compute_topic_weight(
    progress_record: dict[str, Any],
    spacing_strength: float = 0.35,
) -> float:
    """
    Compute a sampling weight for adaptive selection.

    Higher adaptive_score => higher probability of being chosen.
    More time since last_seen => modest increase via spacing term.
    """
    score = float(progress_record.get("adaptive_score", 1.0))
    hours = hours_since_last_seen(progress_record.get("last_seen"))
    spacing_factor = 1.0 + spacing_strength * min(hours / 24.0, 14.0)
    return max(score * spacing_factor, 1e-6)


def weighted_choice(
    items: list[Any],
    weights: list[float],
    rng: Optional[random.Random] = None,
) -> Any:
    """Choose one item according to weights."""
    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")
    if not items:
        raise ValueError("Cannot choose from an empty list")

    rng = rng or random
    total = sum(max(w, 0.0) for w in weights)
    if total <= 0:
        return rng.choice(items)

    threshold = rng.uniform(0, total)
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += max(weight, 0.0)
        if cumulative >= threshold:
            return item
    return items[-1]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def truncate(text: str, max_len: int = 120) -> str:
    """Truncate text for logs."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def safe_get(d: dict[str, Any], key: str, expected_type: type, default: Any = None) -> Any:
    """Typed dict access helper."""
    value = d.get(key, default)
    return value if isinstance(value, expected_type) else default