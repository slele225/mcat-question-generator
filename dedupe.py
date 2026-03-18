"""
Simple duplicate detection for MCAT questions.
Uses text similarity to identify near-duplicates.
"""

import logging
from typing import List, Dict, Any, Set, Tuple
from difflib import SequenceMatcher

import config

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score between 0 and 1
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    return SequenceMatcher(None, norm1, norm2).ratio()


def is_duplicate(
    new_item: Dict[str, Any],
    existing_items: List[Dict[str, Any]],
    threshold: float = config.DUPLICATE_SIMILARITY_THRESHOLD,
    mode: str = "science",
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Check if an item is a duplicate of any existing item.
    
    Args:
        new_item: The item to check
        existing_items: List of existing accepted items
        threshold: Similarity threshold for considering duplicates
        mode: "science" or "cars"
    
    Returns:
        (is_duplicate, max_similarity, most_similar_item)
    """
    if not existing_items:
        return False, 0.0, {}
    
    max_similarity = 0.0
    most_similar = {}
    
    if mode == "science":
        # Compare question text for science items
        new_question = new_item.get("question", "")
        
        for existing in existing_items:
            existing_question = existing.get("question", "")
            similarity = calculate_similarity(new_question, existing_question)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = existing
    
    else:  # cars
        # Compare passage for CARS sets
        new_passage = new_item.get("passage", "")
        
        for existing in existing_items:
            existing_passage = existing.get("passage", "")
            similarity = calculate_similarity(new_passage, existing_passage)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = existing
    
    is_dup = max_similarity >= threshold
    
    if is_dup:
        logger.debug(f"Duplicate detected with similarity {max_similarity:.2f}")
    
    return is_dup, max_similarity, most_similar


def deduplicate_items(
    items: List[Dict[str, Any]],
    threshold: float = config.DUPLICATE_SIMILARITY_THRESHOLD,
    mode: str = "science",
) -> List[Dict[str, Any]]:
    """
    Remove duplicates from a list of items.
    
    Args:
        items: List of items to deduplicate
        threshold: Similarity threshold
        mode: "science" or "cars"
    
    Returns:
        Deduplicated list (preserves order, keeps first occurrence)
    """
    if not items:
        return []
    
    unique_items = []
    
    for item in items:
        is_dup, _, _ = is_duplicate(item, unique_items, threshold, mode)
        if not is_dup:
            unique_items.append(item)
    
    removed = len(items) - len(unique_items)
    if removed > 0:
        logger.info(f"Removed {removed} duplicates")
    
    return unique_items


def get_duplicate_groups(
    items: List[Dict[str, Any]],
    threshold: float = config.DUPLICATE_SIMILARITY_THRESHOLD,
    mode: str = "science",
) -> List[List[Dict[str, Any]]]:
    """
    Group duplicate items together.
    
    Args:
        items: List of items to group
        threshold: Similarity threshold
        mode: "science" or "cars"
    
    Returns:
        List of groups, where each group contains similar items
    """
    if not items:
        return []
    
    groups = []
    used = set()
    
    for i, item in enumerate(items):
        if i in used:
            continue
        
        group = [item]
        used.add(i)
        
        for j, other in enumerate(items):
            if j in used or i == j:
                continue
            
            is_dup, _, _ = is_duplicate(item, [other], threshold, mode)
            if is_dup:
                group.append(other)
                used.add(j)
        
        groups.append(group)
    
    return groups


def find_best_in_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Find the best item in a duplicate group based on validation score.
    
    Args:
        group: List of duplicate items
    
    Returns:
        The item with the highest validation score
    """
    if not group:
        return {}
    
    if len(group) == 1:
        return group[0]
    
    # Sort by validation score (higher is better)
    sorted_group = sorted(
        group,
        key=lambda x: x.get("validation_score", 0),
        reverse=True
    )
    
    return sorted_group[0]