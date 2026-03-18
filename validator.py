"""
MCAT question validation module.
Validates generated items using validator prompts.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple

from prompt_templates import (
    build_science_validator_prompt,
    build_cars_validator_prompt,
    SCIENCE_VALIDATION_SCHEMA,
    CARS_VALIDATION_SCHEMA,
)
from llm_client import get_client
import config
import schemas

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_science_item(
    topic: Dict[str, Any],
    item: Dict[str, Any],
    prior_items: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate a single science item using the validator prompt.
    
    Args:
        topic: Topic dictionary from topics.json
        item: The generated science item to validate
        prior_items: Previously accepted items for duplication checking
    
    Returns:
        Validator feedback dictionary or None if failed
    """
    prompt = build_science_validator_prompt(
        topic=topic,
        generated_item=item,
        prior_items=prior_items or [],
    )
    
    client = get_client()
    result = client.generate_json(prompt)
    
    if not result:
        logger.error(f"Failed to validate science item")
        return None
    
    # Check schema compliance
    valid, errors = schemas.validate_against_schema(result, SCIENCE_VALIDATION_SCHEMA, "validator")
    if not valid:
        logger.error(f"Validator returned invalid schema: {errors}")
        return None
    
    return result


def validate_cars_set(
    topic: Dict[str, Any],
    cars_set: Dict[str, Any],
    prior_sets: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate a CARS set using the validator prompt.
    
    Args:
        topic: Topic dictionary from topics.json
        cars_set: The generated CARS set to validate
        prior_sets: Previously accepted CARS sets for duplication checking
    
    Returns:
        Validator feedback dictionary or None if failed
    """
    prompt = build_cars_validator_prompt(
        topic=topic,
        generated_set=cars_set,
        prior_sets=prior_sets or [],
    )
    
    client = get_client()
    result = client.generate_json(prompt)
    
    if not result:
        logger.error(f"Failed to validate CARS set")
        return None
    
    # Check schema compliance
    valid, errors = schemas.validate_against_schema(result, CARS_VALIDATION_SCHEMA, "validator")
    if not valid:
        logger.error(f"Validator returned invalid schema: {errors}")
        return None
    
    return result


def should_accept(validator_feedback: Dict[str, Any]) -> bool:
    """
    Determine if an item should be accepted based on validator feedback.
    
    Args:
        validator_feedback: The validator's feedback dictionary
    
    Returns:
        True if the item should be accepted
    """
    # Check verdict
    verdict = validator_feedback.get("verdict")
    if verdict == "keep":
        return True
    
    # Check score if available
    score = validator_feedback.get("overall_score", 0)
    if score >= config.MIN_ACCEPTABLE_SCORE:
        return True
    
    return False


def should_revise(validator_feedback: Dict[str, Any]) -> bool:
    """
    Determine if an item should be revised based on validator feedback.
    
    Args:
        validator_feedback: The validator's feedback dictionary
    
    Returns:
        True if the item should be revised
    """
    verdict = validator_feedback.get("verdict")
    return verdict == "revise"


def should_reject(validator_feedback: Dict[str, Any]) -> bool:
    """
    Determine if an item should be rejected based on validator feedback.
    
    Args:
        validator_feedback: The validator's feedback dictionary
    
    Returns:
        True if the item should be rejected
    """
    verdict = validator_feedback.get("verdict")
    if verdict == "reject":
        return True
    
    score = validator_feedback.get("overall_score", 10)
    if score < config.MIN_ACCEPTABLE_SCORE:
        return True
    
    return False


def extract_required_fixes(validator_feedback: Dict[str, Any]) -> List[str]:
    """
    Extract required fixes from validator feedback.
    
    Args:
        validator_feedback: The validator's feedback dictionary
    
    Returns:
        List of required fix descriptions
    """
    return validator_feedback.get("required_fixes", [])


def get_duplication_risk(validator_feedback: Dict[str, Any]) -> str:
    """
    Get duplication risk level from validator feedback.
    
    Args:
        validator_feedback: The validator's feedback dictionary
    
    Returns:
        "low", "medium", or "high"
    """
    return validator_feedback.get("duplication_risk", "low")


def validate_with_retry(
    validator_func,
    max_attempts: int = 2,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Retry validation with exponential backoff.
    
    Args:
        validator_func: Function that performs validation
        max_attempts: Maximum number of attempts
        **kwargs: Arguments to pass to validator_func
    
    Returns:
        Validator feedback or None if all attempts fail
    """
    import time
    
    for attempt in range(max_attempts):
        try:
            result = validator_func(**kwargs)
            if result:
                return result
            
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt
                logger.debug(f"Validation attempt {attempt + 1} failed, waiting {wait_time}s")
                time.sleep(wait_time)
        
        except Exception as e:
            logger.error(f"Validation error on attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
    
    logger.error(f"All {max_attempts} validation attempts failed")
    return None