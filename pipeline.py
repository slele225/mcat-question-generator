"""
Main pipeline orchestration for MCAT question generation.
Coordinates generation, validation, repair, and deduplication.
"""

import logging
from typing import List, Dict, Any, Optional

from tqdm import tqdm

import config
import generator
import validator
import repair
import dedupe
import io_utils
from llm_client import get_client

logger = logging.getLogger(__name__)


class SciencePipeline:
    """Pipeline for generating science questions."""
    
    def __init__(self, topic: Dict[str, Any]):
        """
        Initialize science pipeline for a topic.
        
        Args:
            topic: Topic dictionary from topics.json
        """
        self.topic = topic
        self.topic_id = topic.get("topic_id", "unknown")
        self.accepted_items = []
        self.attempts = 0
        
        # Load existing items
        self.accepted_items = io_utils.load_existing_items(self.topic_id, "science")
        logger.info(f"Loaded {len(self.accepted_items)} existing science items for {self.topic_id}")
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the generation pipeline for this topic.
        
        Returns:
            List of accepted items
        """
        target = config.SCIENCE_TARGET_PER_TOPIC
        batch_size = config.SCIENCE_BATCH_SIZE
        
        logger.info(f"Starting science pipeline for {self.topic_id}, target: {target}")
        
        pbar = tqdm(total=target, desc=f"Science {self.topic_id}", unit="items")
        pbar.update(len(self.accepted_items))
        
        while len(self.accepted_items) < target and self.attempts < config.MAX_ATTEMPTS_PER_TOPIC:
            self.attempts += 1
            
            # Generate batch
            items = generator.generate_science_batch_with_retry(
                topic=self.topic,
                difficulty="medium",
                num_questions=batch_size,
            )
            
            if not items:
                logger.warning(f"Generation failed on attempt {self.attempts}")
                continue
            
            # Process each item
            for item in items:
                if len(self.accepted_items) >= target:
                    break
                
                self._process_item(item)
            
            pbar.update(len(self.accepted_items) - pbar.n)
            
            # Save checkpoint
            io_utils.save_checkpoint(
                self.topic_id,
                "science",
                len(self.accepted_items),
                self.attempts,
            )
        
        pbar.close()
        logger.info(f"Completed science pipeline for {self.topic_id}: {len(self.accepted_items)} items")
        return self.accepted_items
    
    def _process_item(self, item: Dict[str, Any]):
        """Process a single item through validation and repair."""
        
        # Skip if config says so
        if config.SKIP_VALIDATION:
            self.accepted_items.append(item)
            io_utils.save_science_items([item], self.topic_id)
            return
        
        # Validate
        feedback = validator.validate_with_retry(
            validator.validate_science_item,
            topic=self.topic,
            item=item,
            prior_items=self.accepted_items,
        )
        
        if not feedback:
            logger.debug("Validation failed, skipping item")
            io_utils.save_failed_item(item, self.topic_id, "validation_failed", "science")
            return
        
        # Check verdict
        if validator.should_accept(feedback):
            # Check for duplicates
            is_dup, sim, _ = dedupe.is_duplicate(
                item, self.accepted_items, mode="science"
            )
            
            if not is_dup:
                # Add validation score
                item["validation_score"] = feedback.get("overall_score", 5)
                self.accepted_items.append(item)
                io_utils.save_science_items([item], self.topic_id)
                logger.debug(f"Accepted item with score {item['validation_score']}")
            else:
                logger.debug(f"Rejected duplicate (similarity: {sim:.2f})")
                io_utils.save_failed_item(item, self.topic_id, f"duplicate_{sim:.2f}", "science")
        
        elif validator.should_revise(feedback) and repair.is_repairable(feedback):
            # Attempt repair
            repaired = repair.repair_with_retry(
                repair.repair_science_item,
                topic=self.topic,
                item=item,
                validator_feedback=feedback,
            )
            
            if repaired:
                # Re-validate repaired item
                self._process_item(repaired)  # Recursive call
            else:
                logger.debug("Repair failed")
                io_utils.save_failed_item(item, self.topic_id, "repair_failed", "science")
        
        else:
            # Reject
            logger.debug(f"Item rejected: {feedback.get('verdict', 'unknown')}")
            io_utils.save_failed_item(
                item,
                self.topic_id,
                f"rejected_{feedback.get('verdict', 'unknown')}",
                "science",
            )


class CARSPipeline:
    """Pipeline for generating CARS sets."""
    
    def __init__(self, topic: Dict[str, Any]):
        """
        Initialize CARS pipeline for a topic.
        
        Args:
            topic: Topic dictionary from topics.json
        """
        self.topic = topic
        self.topic_id = topic.get("topic_id", "unknown")
        self.accepted_sets = []
        self.attempts = 0
        
        # Load existing sets
        self.accepted_sets = io_utils.load_existing_items(self.topic_id, "cars")
        logger.info(f"Loaded {len(self.accepted_sets)} existing CARS sets for {self.topic_id}")
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the generation pipeline for this topic.
        
        Returns:
            List of accepted CARS sets
        """
        target = config.CARS_TARGET_PER_TOPIC
        
        logger.info(f"Starting CARS pipeline for {self.topic_id}, target: {target}")
        
        pbar = tqdm(total=target, desc=f"CARS {self.topic_id}", unit="sets")
        pbar.update(len(self.accepted_sets))
        
        while len(self.accepted_sets) < target and self.attempts < config.MAX_ATTEMPTS_PER_TOPIC:
            self.attempts += 1
            
            # Generate CARS set
            cars_set = generator.generate_cars_set_with_retry(
                topic=self.topic,
                difficulty="hard",
                num_questions=config.CARS_QUESTIONS_PER_SET,
                passage_word_target=config.CARS_PASSAGE_WORDS,
            )
            
            if not cars_set:
                logger.warning(f"Generation failed on attempt {self.attempts}")
                continue
            
            self._process_set(cars_set)
            pbar.update(len(self.accepted_sets) - pbar.n)
            
            # Save checkpoint
            io_utils.save_checkpoint(
                self.topic_id,
                "cars",
                len(self.accepted_sets),
                self.attempts,
            )
        
        pbar.close()
        logger.info(f"Completed CARS pipeline for {self.topic_id}: {len(self.accepted_sets)} sets")
        return self.accepted_sets
    
    def _process_set(self, cars_set: Dict[str, Any]):
        """Process a single CARS set through validation and repair."""
        
        if config.SKIP_VALIDATION:
            self.accepted_sets.append(cars_set)
            io_utils.save_cars_sets([cars_set], self.topic_id)
            return
        
        # Validate
        feedback = validator.validate_with_retry(
            validator.validate_cars_set,
            topic=self.topic,
            cars_set=cars_set,
            prior_sets=self.accepted_sets,
        )
        
        if not feedback:
            logger.debug("Validation failed, skipping set")
            io_utils.save_failed_item(cars_set, self.topic_id, "validation_failed", "cars")
            return
        
        # Check verdict
        if validator.should_accept(feedback):
            # Check for duplicates
            is_dup, sim, _ = dedupe.is_duplicate(
                cars_set, self.accepted_sets, mode="cars"
            )
            
            if not is_dup:
                cars_set["validation_score"] = feedback.get("overall_score", 5)
                self.accepted_sets.append(cars_set)
                io_utils.save_cars_sets([cars_set], self.topic_id)
                logger.debug(f"Accepted CARS set with score {cars_set['validation_score']}")
            else:
                logger.debug(f"Rejected duplicate (similarity: {sim:.2f})")
                io_utils.save_failed_item(cars_set, self.topic_id, f"duplicate_{sim:.2f}", "cars")
        
        elif validator.should_revise(feedback) and repair.is_repairable(feedback):
            # Attempt repair
            repaired = repair.repair_with_retry(
                repair.repair_cars_set,
                topic=self.topic,
                cars_set=cars_set,
                validator_feedback=feedback,
            )
            
            if repaired:
                self._process_set(repaired)  # Recursive call
            else:
                logger.debug("Repair failed")
                io_utils.save_failed_item(cars_set, self.topic_id, "repair_failed", "cars")
        
        else:
            logger.debug(f"Set rejected: {feedback.get('verdict', 'unknown')}")
            io_utils.save_failed_item(
                cars_set,
                self.topic_id,
                f"rejected_{feedback.get('verdict', 'unknown')}",
                "cars",
            )