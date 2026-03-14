"""
backend.py

Backend abstraction for MCAT question generation.

Goals:
- Keep generate_bank.py independent from any one inference provider
- Support easy swapping between:
    - Mock backend (for local testing)
    - OpenAI-compatible endpoints
    - Future custom backends (vLLM server, Ollama, etc.)
- Return parsed Python dicts whenever possible
- Keep dependencies minimal

Notes:
- This file only includes standard-library code.
- The OpenAICompatibleBackend uses urllib so requirements can stay light.
- For local pipeline testing, use MockBackend first.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from utils import (
    make_question_id,
    make_question_set_id,
    normalize_text,
    stable_hash,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BackendError(Exception):
    """Base error for generation backend failures."""


class BackendResponseError(BackendError):
    """Raised when the backend returns malformed or unusable output."""


class BackendRequestError(BackendError):
    """Raised when the request to a remote backend fails."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """
    Shared generation parameters.

    These are generic enough to apply across most backends.
    """

    temperature: float = 0.7
    max_tokens: int = 1200
    top_p: float = 0.95
    timeout_seconds: int = 180
    retries: int = 2
    retry_backoff_seconds: float = 2.0


@dataclass
class GenerationResult:
    """
    Parsed result from a backend generation call.

    raw_text:
        The raw model output before JSON parsing / cleanup.

    parsed:
        Parsed dict/list if parsing succeeded, else None.

    model:
        Backend/model identifier.

    success:
        Whether the generation succeeded in a usable way.

    error:
        Error message on failure.
    """

    raw_text: str
    parsed: Any
    model: str
    success: bool
    error: Optional[str] = None
    created_at: str = ""


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class GenerationBackend(ABC):
    """
    Abstract interface for all question-generation backends.

    generate_json():
        Generate one JSON-shaped output for one prompt.

    generate_json_batch():
        Generate one JSON-shaped output per prompt.
        Default implementation loops over generate_json(), but subclasses may
        override this with true batching for throughput.
    """

    def __init__(self, model_name: str, config: Optional[GenerationConfig] = None) -> None:
        self.model_name = model_name
        self.config = config or GenerationConfig()

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        schema_hint: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate one JSON-like response for one prompt.
        """
        raise NotImplementedError

    def generate_json_batch(
        self,
        prompts: list[str],
        schema_hint: Optional[str] = None,
    ) -> list[GenerationResult]:
        """
        Default batch implementation: loop serially.

        Subclasses can override for better performance.
        """
        return [self.generate_json(prompt, schema_hint=schema_hint) for prompt in prompts]


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def extract_json_candidate(text: str) -> str:
    """
    Try to extract a JSON object or array from raw model text.

    Handles common cases like:
    - fenced code blocks
    - extra commentary before/after JSON
    - raw JSON already present

    This is intentionally simple and conservative.
    """
    text = text.strip()

    if not text:
        return text

    # Remove fenced markdown if present
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            # Drop first and last fence lines if they look like fences
            if lines[0].startswith("```") and lines[-1].startswith("```"):
                text = "\n".join(lines[1:-1]).strip()

    # Try full text first
    if looks_like_json(text):
        return text

    # Try to extract first top-level object or array span
    start_positions = []
    obj_start = text.find("{")
    arr_start = text.find("[")
    if obj_start != -1:
        start_positions.append(obj_start)
    if arr_start != -1:
        start_positions.append(arr_start)

    if not start_positions:
        return text

    start = min(start_positions)

    end_obj = text.rfind("}")
    end_arr = text.rfind("]")
    end_candidates = [x for x in [end_obj, end_arr] if x != -1]
    if not end_candidates:
        return text[start:]

    end = max(end_candidates)
    if end >= start:
        return text[start : end + 1].strip()

    return text


def looks_like_json(text: str) -> bool:
    """Cheap heuristic for whether text resembles JSON."""
    text = text.strip()
    return (
        (text.startswith("{") and text.endswith("}"))
        or (text.startswith("[") and text.endswith("]"))
    )


def parse_json_response(text: str) -> Any:
    """
    Parse model output into Python data.

    Raises BackendResponseError if parsing fails.
    """
    candidate = extract_json_candidate(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise BackendResponseError(f"Failed to parse JSON response: {exc}") from exc


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockBackend(GenerationBackend):
    """
    Deterministic fake backend for local testing.

    It fabricates plausible science and CARS outputs using the prompt text.
    This lets the rest of the pipeline be tested without model inference.
    """

    def __init__(
        self,
        model_name: str = "mock-backend",
        config: Optional[GenerationConfig] = None,
        seed: int = 123,
    ) -> None:
        super().__init__(model_name=model_name, config=config)
        self.rng = random.Random(seed)

    def generate_json(
        self,
        prompt: str,
        schema_hint: Optional[str] = None,
    ) -> GenerationResult:
        created_at = utc_now_iso()
        try:
            lower = normalize_text(prompt)

            if "mode\":\"cars" in lower or "mode: cars" in lower or "cars" in lower:
                obj = self._mock_cars_output(prompt)
            else:
                obj = self._mock_science_output(prompt)

            raw_text = json.dumps(obj, ensure_ascii=False)
            parsed = obj
            return GenerationResult(
                raw_text=raw_text,
                parsed=parsed,
                model=self.model_name,
                success=True,
                error=None,
                created_at=created_at,
            )
        except Exception as exc:
            return GenerationResult(
                raw_text="",
                parsed=None,
                model=self.model_name,
                success=False,
                error=str(exc),
                created_at=created_at,
            )

    def _extract_topic_id(self, prompt: str) -> str:
        """
        Pull topic_id from prompt text if present.

        Falls back to a stable synthetic ID.
        """
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("topic_id"):
                parts = stripped.split(":", 1)
                if len(parts) == 2 and parts[1].strip():
                    return parts[1].strip()

        return f"TOPIC_{stable_hash(prompt, length=8)}"

    def _extract_title(self, prompt: str) -> str:
        """Pull title from prompt text if present."""
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("title"):
                parts = stripped.split(":", 1)
                if len(parts) == 2 and parts[1].strip():
                    return parts[1].strip()
        return "Untitled Topic"

    def _mock_science_output(self, prompt: str) -> dict[str, Any]:
        topic_id = self._extract_topic_id(prompt)
        title = self._extract_title(prompt)

        question = f"Which statement best reflects the core concept in {title}?"
        options = {
            "A": "A correct interpretation of the tested concept.",
            "B": "A distractor based on a common conceptual confusion.",
            "C": "A distractor that overgeneralizes a related idea.",
            "D": "A distractor that reverses cause and effect.",
        }
        correct_answer = "A"
        explanation = (
            "Choice A is correct because it matches the core principle being tested. "
            "The other options reflect common MCAT-style errors such as overgeneralization, "
            "confusing correlation with mechanism, or reversing the relevant relationship."
        )

        return {
            "question_id": make_question_id(topic_id, question),
            "topic_id": topic_id,
            "mode": "science",
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "difficulty": self.rng.choice(["easy", "medium", "hard"]),
            "model": self.model_name,
            "prompt_version": "mock_v1",
        }

    def _mock_cars_output(self, prompt: str) -> dict[str, Any]:
        topic_id = self._extract_topic_id(prompt)
        title = self._extract_title(prompt)

        passage = (
            f"The essay on {title} argues that readers often mistake confidence for clarity. "
            f"Although sweeping claims may appear persuasive at first glance, the passage suggests "
            f"that intellectual discipline requires attending to nuance, ambiguity, and the limits "
            f"of any single framework. In this way, the author treats interpretation not as passive "
            f"absorption but as an active confrontation with uncertainty."
        )

        questions = [
            {
                "question": "What is the author's central claim?",
                "options": {
                    "A": "Interpretation requires active engagement with nuance and uncertainty.",
                    "B": "Clear writing should eliminate all ambiguity from serious argument.",
                    "C": "Readers generally prefer arguments that contain technical vocabulary.",
                    "D": "Confidence in tone is the best indicator of intellectual rigor.",
                },
                "correct_answer": "A",
                "explanation": (
                    "The passage emphasizes nuance, uncertainty, and active interpretation. "
                    "The other choices either exaggerate, distort tone, or focus on secondary points."
                ),
            },
            {
                "question": "Which mistaken reading would the author be most likely to reject?",
                "options": {
                    "A": "Nuance can coexist with strong argument.",
                    "B": "A forceful tone guarantees the truth of an argument.",
                    "C": "Interpretation involves judgment from the reader.",
                    "D": "Ambiguity can reveal the limits of a framework.",
                },
                "correct_answer": "B",
                "explanation": (
                    "The author explicitly warns against equating confidence with clarity or rigor."
                ),
            },
        ]

        return {
            "question_set_id": make_question_set_id(topic_id, passage),
            "topic_id": topic_id,
            "mode": "cars",
            "passage": passage,
            "questions": questions,
            "difficulty": self.rng.choice(["easy", "medium", "hard"]),
            "model": self.model_name,
            "prompt_version": "mock_v1",
        }


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------


class OpenAICompatibleBackend(GenerationBackend):
    """
    Backend for OpenAI-style chat-completions endpoints.

    This is meant to work with:
    - vLLM serve
    - OpenAI-compatible self-hosted servers
    - possibly OpenAI-compatible proxies

    Endpoint format assumed:
        POST {base_url}/chat/completions

    Expected response shape:
        {
          "choices": [
            {
              "message": {
                "content": "..."
              }
            }
          ]
        }

    Notes:
    - Uses only urllib from the stdlib.
    - For some providers, you may need an API key.
    - For local vLLM servers, api_key can often be omitted.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_name=model_name, config=config)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.system_prompt = system_prompt or (
            "You are a careful generator that returns valid JSON only."
        )
        self.extra_body = extra_body or {}

    def generate_json(
        self,
        prompt: str,
        schema_hint: Optional[str] = None,
    ) -> GenerationResult:
        created_at = utc_now_iso()
        attempt = 0
        last_error: Optional[str] = None

        while attempt <= self.config.retries:
            try:
                raw_text = self._request_chat_completion(prompt, schema_hint=schema_hint)
                parsed = parse_json_response(raw_text)

                return GenerationResult(
                    raw_text=raw_text,
                    parsed=parsed,
                    model=self.model_name,
                    success=True,
                    error=None,
                    created_at=created_at,
                )

            except Exception as exc:
                last_error = str(exc)
                if attempt >= self.config.retries:
                    break
                sleep_for = self.config.retry_backoff_seconds * (attempt + 1)
                time.sleep(sleep_for)
                attempt += 1

        return GenerationResult(
            raw_text="",
            parsed=None,
            model=self.model_name,
            success=False,
            error=last_error or "Unknown backend failure",
            created_at=created_at,
        )

    def generate_json_batch(
        self,
        prompts: list[str],
        schema_hint: Optional[str] = None,
    ) -> list[GenerationResult]:
        """
        Default serial batching over prompts.

        This keeps behavior predictable. If later you want real concurrent
        requests, you can add a threaded/async subclass without changing the
        rest of the codebase.
        """
        return [self.generate_json(prompt, schema_hint=schema_hint) for prompt in prompts]

    def _request_chat_completion(
        self,
        prompt: str,
        schema_hint: Optional[str] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        user_content = prompt
        if schema_hint:
            user_content = (
                f"{prompt}\n\n"
                f"Return valid JSON only.\n"
                f"Schema reminder:\n{schema_hint}\n"
            )

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }

        body.update(self.extra_body)

        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=payload,
            method="POST",
            headers=self._build_headers(),
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise BackendRequestError(
                f"HTTP {exc.code} from backend: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise BackendRequestError(f"Failed to reach backend: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackendResponseError(
                f"Backend returned non-JSON response envelope: {raw[:500]}"
            ) from exc

        try:
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendResponseError(
                f"Unexpected chat-completions response format: {parsed}"
            ) from exc

        if not isinstance(content, str):
            raise BackendResponseError("Backend message content is not a string")

        return content

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def build_backend(
    backend_type: str,
    model_name: str,
    *,
    config: Optional[GenerationConfig] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    extra_body: Optional[dict[str, Any]] = None,
    seed: int = 123,
) -> GenerationBackend:
    """
    Factory for constructing backends from CLI/config values.

    Supported backend_type values:
    - "mock"
    - "openai_compat"

    Example:
        backend = build_backend(
            backend_type="openai_compat",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            base_url="http://127.0.0.1:8000/v1",
        )
    """
    backend_type = backend_type.strip().lower()

    if backend_type == "mock":
        return MockBackend(model_name=model_name or "mock-backend", config=config, seed=seed)

    if backend_type in {"openai_compat", "openai-compatible", "openai"}:
        if not base_url:
            raise ValueError("base_url is required for openai_compat backend")
        return OpenAICompatibleBackend(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            system_prompt=system_prompt,
            extra_body=extra_body,
        )

    raise ValueError(f"Unsupported backend_type: {backend_type}")


# ---------------------------------------------------------------------------
# Optional normalization helpers for generation outputs
# ---------------------------------------------------------------------------


def normalize_science_output(obj: dict[str, Any], model_name: str, prompt_version: str) -> dict[str, Any]:
    """
    Normalize a science output object into canonical bank format.

    Useful when models omit some fields but include enough information to
    recover them.
    """
    topic_id = str(obj.get("topic_id", "")).strip()
    question = str(obj.get("question", "")).strip()

    out = dict(obj)
    out["topic_id"] = topic_id
    out["mode"] = "science"
    out["question"] = question
    out["question_id"] = str(out.get("question_id") or make_question_id(topic_id, question))
    out["model"] = str(out.get("model") or model_name)
    out["prompt_version"] = str(out.get("prompt_version") or prompt_version)
    return out


def normalize_cars_output(obj: dict[str, Any], model_name: str, prompt_version: str) -> dict[str, Any]:
    """
    Normalize a CARS output object into canonical bank format.
    """
    topic_id = str(obj.get("topic_id", "")).strip()
    passage = str(obj.get("passage", "")).strip()

    out = dict(obj)
    out["topic_id"] = topic_id
    out["mode"] = "cars"
    out["passage"] = passage
    out["question_set_id"] = str(
        out.get("question_set_id") or make_question_set_id(topic_id, passage)
    )
    out["model"] = str(out.get("model") or model_name)
    out["prompt_version"] = str(out.get("prompt_version") or prompt_version)
    return out