"""
I/O utilities for MCAT question generation.
Handles loading topics, saving results, and file management.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import jsonlines

import config

logger = logging.getLogger(__name__)


def load_topics(file_path: str = config.TOPICS_FILE) -> List[Dict[str, Any]]:
    """
    Load topics from JSON file.
    
    Args:
        file_path: Path to topics.json
    
    Returns:
        List of topic dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            topics = json.load(f)
        
        logger.info(f"Loaded {len(topics)} topics from {file_path}")
        return topics
    
    except FileNotFoundError:
        logger.error(f"Topics file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in topics file: {e}")
        return []


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(config.SCIENCE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CARS_OUTPUT_DIR, exist_ok=True)
    
    if config.SAVE_FAILED_ITEMS:
        failed_dir = os.path.join(config.OUTPUT_DIR, "failed")
        os.makedirs(failed_dir, exist_ok=True)
    
    logger.debug(f"Output directories created")


def save_science_items(
    items: List[Dict[str, Any]],
    topic_id: str,
    append: bool = True,
) -> str:
    """
    Save science items to JSONL file.
    
    Args:
        items: List of science items to save
        topic_id: Topic ID for filename
        append: If True, append to existing file; otherwise overwrite
    
    Returns:
        Path to saved file
    """
    filename = f"{topic_id}_science.jsonl"
    filepath = os.path.join(config.SCIENCE_OUTPUT_DIR, filename)
    
    mode = 'a' if append else 'w'
    
    with jsonlines.open(filepath, mode=mode) as writer:
        writer.write_all(items)
    
    logger.info(f"Saved {len(items)} science items to {filepath}")
    return filepath


def save_cars_sets(
    sets: List[Dict[str, Any]],
    topic_id: str,
    append: bool = True,
) -> str:
    """
    Save CARS sets to JSONL file.
    
    Args:
        sets: List of CARS sets to save
        topic_id: Topic ID for filename
        append: If True, append to existing file; otherwise overwrite
    
    Returns:
        Path to saved file
    """
    filename = f"{topic_id}_cars.jsonl"
    filepath = os.path.join(config.CARS_OUTPUT_DIR, filename)
    
    mode = 'a' if append else 'w'
    
    with jsonlines.open(filepath, mode=mode) as writer:
        writer.write_all(sets)
    
    logger.info(f"Saved {len(sets)} CARS sets to {filepath}")
    return filepath


def save_failed_item(
    item: Dict[str, Any],
    topic_id: str,
    reason: str,
    mode: str,
):
    """
    Save a failed/rejected item for inspection.
    
    Args:
        item: The failed item
        topic_id: Topic ID
        reason: Reason for failure (e.g., "reject", "repair_failed")
        mode: "science" or "cars"
    """
    if not config.SAVE_FAILED_ITEMS:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{topic_id}_{mode}_failed_{timestamp}.json"
    filepath = os.path.join(config.OUTPUT_DIR, "failed", filename)
    
    # Add failure metadata
    item_with_meta = item.copy()
    item_with_meta["_failure_reason"] = reason
    item_with_meta["_failure_timestamp"] = timestamp
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(item_with_meta, f, indent=2, ensure_ascii=False)
    
    logger.debug(f"Saved failed item to {filepath}")


def load_existing_items(
    topic_id: str,
    mode: str,
) -> List[Dict[str, Any]]:
    """
    Load existing accepted items for a topic.
    
    Args:
        topic_id: Topic ID
        mode: "science" or "cars"
    
    Returns:
        List of existing items
    """
    if mode == "science":
        dir_path = config.SCIENCE_OUTPUT_DIR
        filename = f"{topic_id}_science.jsonl"
    else:
        dir_path = config.CARS_OUTPUT_DIR
        filename = f"{topic_id}_cars.jsonl"
    
    filepath = os.path.join(dir_path, filename)
    
    if not os.path.exists(filepath):
        return []
    
    items = []
    try:
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                items.append(obj)
        
        logger.debug(f"Loaded {len(items)} existing {mode} items for {topic_id}")
    
    except Exception as e:
        logger.error(f"Error loading existing items from {filepath}: {e}")
    
    return items


def count_existing_items(topic_id: str, mode: str) -> int:
    """Count existing accepted items for a topic."""
    return len(load_existing_items(topic_id, mode))


def save_checkpoint(
    topic_id: str,
    mode: str,
    accepted_count: int,
    attempts: int,
):
    """
    Save a checkpoint for resuming generation.
    
    Args:
        topic_id: Topic ID
        mode: "science" or "cars"
        accepted_count: Number of accepted items so far
        attempts: Number of attempts made
    """
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = f"{topic_id}_{mode}_checkpoint.json"
    filepath = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        "topic_id": topic_id,
        "mode": mode,
        "accepted_count": accepted_count,
        "attempts": attempts,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)
    
    logger.debug(f"Saved checkpoint: {filepath}")


def load_checkpoint(topic_id: str, mode: str) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint for a topic if it exists.
    
    Args:
        topic_id: Topic ID
        mode: "science" or "cars"
    
    Returns:
        Checkpoint dict or None
    """
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, "checkpoints")
    filename = f"{topic_id}_{mode}_checkpoint.json"
    filepath = os.path.join(checkpoint_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None