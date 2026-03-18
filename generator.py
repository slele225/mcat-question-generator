"""
MCAT question generation module.
Handles science and CARS generation using prompt templates.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from prompt_templates import (
    CARS_PROMPT_VERSION,
    SCIENCE_PROMPT_VERSION,
    build_cars_generation_prompt,
    build_science_generation_prompt,
)
from llm_client import get_client


logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Raised when generation fails."""


def generate_science_batch(
    topic: Dict[str, Any],
    difficulty: str = "medium",
    num_questions: int = 5,
) -> Optional[List[Dict[str, Any]]]:
    """
    Generate a batch of science questions for a topic.

    Args:
        topic: Topic dictionary from topics.json
        difficulty: Desired difficulty level
        num_questions: Number of questions to generate

    Returns:
        List of question dictionaries or None if failed
    """
    prompt = build_science_generation_prompt(
        topic=topic,
        difficulty=difficulty,
        num_questions=num_questions,
    )

    client = get_client()
    result = client.generate_json(prompt)

    if not result:
        logger.error(
            "Failed to generate science batch for topic %s",
            topic.get("topic_id"),
        )
        return None

    # Validate basic schema
    if not isinstance(result, dict) or "items" not in result:
        logger.error("Invalid response format: missing 'items' key")
        return None

    items = result["items"]
    if not isinstance(items, list):
        logger.error("Invalid response format: 'items' is not a list")
        return None

    # Add metadata to each item
    for item in items:
        if "prompt_version" not in item:
            item["prompt_version"] = SCIENCE_PROMPT_VERSION
        if "validation_score" not in item:
            item["validation_score"] = 0

    logger.info("Generated %d science questions", len(items))
    return items


def generate_cars_set(
    topic: Dict[str, Any],
    difficulty: str = "hard",
    num_questions: int = 8,
    passage_word_target: int = 600,
) -> Optional[Dict[str, Any]]:
    """
    Generate a CARS passage with questions.

    Args:
        topic: Topic dictionary from topics.json
        difficulty: Desired difficulty level
        num_questions: Number of questions to generate
        passage_word_target: Target passage length in words

    Returns:
        CARS set dictionary or None if failed
    """
    prompt = build_cars_generation_prompt(
        topic=topic,
        difficulty=difficulty,
        num_questions=num_questions,
        passage_word_target=passage_word_target,
    )

    client = get_client()
    result = client.generate_json(prompt)

    if not result:
        logger.error(
            "Failed to generate CARS set for topic %s",
            topic.get("topic_id"),
        )
        return None

    if not isinstance(result, dict):
        logger.error("Invalid CARS response format: result is not a dict")
        return None

    # Add metadata
    if "prompt_version" not in result:
        result["prompt_version"] = CARS_PROMPT_VERSION
    if "validation_score" not in result:
        result["validation_score"] = 0

    logger.info(
        "Generated CARS set with %d questions",
        len(result.get("questions", [])),
    )
    return result


def generate_with_retry(
    generator_func,
    max_attempts: int = 3,
    **kwargs,
) -> Optional[Any]:
    """
    Retry generation with simple exponential backoff.

    Args:
        generator_func: Function that generates content
        max_attempts: Maximum number of attempts
        **kwargs: Arguments to pass to generator_func

    Returns:
        Generated content or None if all attempts fail
    """
    for attempt in range(max_attempts):
        try:
            result = generator_func(**kwargs)
            if result:
                return result

            if attempt < max_attempts - 1:
                wait_time = 2**attempt
                logger.debug(
                    "Generation attempt %d failed, waiting %ds",
                    attempt + 1,
                    wait_time,
                )
                time.sleep(wait_time)

        except Exception as e:
            logger.error("Generation error on attempt %d: %s", attempt + 1, e)
            if attempt < max_attempts - 1:
                wait_time = 2**attempt
                time.sleep(wait_time)

    logger.error("All %d generation attempts failed", max_attempts)
    return None


def generate_science_batch_with_retry(
    topic: Dict[str, Any],
    difficulty: str = "medium",
    num_questions: int = 5,
    max_attempts: int = 3,
) -> Optional[List[Dict[str, Any]]]:
    """Generate a science batch with retries."""
    return generate_with_retry(
        generate_science_batch,
        max_attempts=max_attempts,
        topic=topic,
        difficulty=difficulty,
        num_questions=num_questions,
    )


def generate_cars_set_with_retry(
    topic: Dict[str, Any],
    difficulty: str = "hard",
    num_questions: int = 8,
    passage_word_target: int = 600,
    max_attempts: int = 3,
) -> Optional[Dict[str, Any]]:
    """Generate a CARS set with retries."""
    return generate_with_retry(
        generate_cars_set,
        max_attempts=max_attempts,
        topic=topic,
        difficulty=difficulty,
        num_questions=num_questions,
        passage_word_target=passage_word_target,
    )