"""
MCAT question repair module.
Repairs items flagged as "revise" by the validator.
"""

import logging
from typing import Dict, Any, Optional

from prompt_templates import (
    build_science_repair_prompt,
    build_cars_repair_prompt,
    SCIENCE_REPAIR_OUTPUT_SCHEMA,
    CARS_REPAIR_OUTPUT_SCHEMA,
)
from llm_client import get_client
import config
import schemas

logger = logging.getLogger(__name__)


class RepairError(Exception):
    """Raised when repair fails."""
    pass


def repair_science_item(
    topic: Dict[str, Any],
    item: Dict[str, Any],
    validator_feedback: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Repair a science item based on validator feedback.
    
    Args:
        topic: Topic dictionary from topics.json
        item: The original generated item
        validator_feedback: Validator feedback with required fixes
    
    Returns:
        Repaired item or None if failed
    """
    prompt = build_science_repair_prompt(
        topic=topic,
        generated_item=item,
        validator_feedback=validator_feedback,
    )
    
    client = get_client()
    result = client.generate_json(prompt)
    
    if not result:
        logger.error("Failed to repair science item")
        return None
    
    # Check schema compliance
    valid, errors = schemas.validate_against_schema(result, SCIENCE_REPAIR_OUTPUT_SCHEMA, "repaired")
    if not valid:
        logger.error(f"Repaired item has schema errors: {errors}")
        return None
    
    # Preserve original metadata if needed
    if "prompt_version" not in result:
        result["prompt_version"] = item.get("prompt_version", "")
    if "validation_score" not in result:
        result["validation_score"] = 0
    
    logger.info("Successfully repaired science item")
    return result


def repair_cars_set(
    topic: Dict[str, Any],
    cars_set: Dict[str, Any],
    validator_feedback: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Repair a CARS set based on validator feedback.
    
    Args:
        topic: Topic dictionary from topics.json
        cars_set: The original generated CARS set
        validator_feedback: Validator feedback with required fixes
    
    Returns:
        Repaired CARS set or None if failed
    """
    prompt = build_cars_repair_prompt(
        topic=topic,
        generated_set=cars_set,
        validator_feedback=validator_feedback,
    )
    
    client = get_client()
    result = client.generate_json(prompt)
    
    if not result:
        logger.error("Failed to repair CARS set")
        return None
    
    # Check schema compliance
    valid, errors = schemas.validate_against_schema(result, CARS_REPAIR_OUTPUT_SCHEMA, "repaired")
    if not valid:
        logger.error(f"Repaired CARS set has schema errors: {errors}")
        return None
    
    # Preserve original metadata
    if "prompt_version" not in result:
        result["prompt_version"] = cars_set.get("prompt_version", "")
    if "validation_score" not in result:
        result["validation_score"] = 0
    
    logger.info("Successfully repaired CARS set")
    return result


def repair_with_retry(
    repair_func,
    max_attempts: int = 2,
    **kwargs
) -> Optional[Any]:
    """
    Retry repair with exponential backoff.
    
    Args:
        repair_func: Function that performs repair
        max_attempts: Maximum number of attempts
        **kwargs: Arguments to pass to repair_func
    
    Returns:
        Repaired item or None if all attempts fail
    """
    import time
    
    for attempt in range(max_attempts):
        try:
            result = repair_func(**kwargs)
            if result:
                return result
            
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt
                logger.debug(f"Repair attempt {attempt + 1} failed, waiting {wait_time}s")
                time.sleep(wait_time)
        
        except Exception as e:
            logger.error(f"Repair error on attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
    
    logger.error(f"All {max_attempts} repair attempts failed")
    return None


def is_repairable(validator_feedback: Dict[str, Any]) -> bool:
    """
    Check if an item is repairable based on validator feedback.
    
    Args:
        validator_feedback: Validator feedback dictionary
    
    Returns:
        True if the item should be repaired
    """
    from validator import should_revise
    
    if not should_revise(validator_feedback):
        return False
    
    # Check if there are specific required fixes
    fixes = validator_feedback.get("required_fixes", [])
    if not fixes:
        logger.warning("Validator says 'revise' but no required_fixes provided")
        return True  # Still try to repair
    
    return True