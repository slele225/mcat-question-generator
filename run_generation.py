"""
Entry point for MCAT question generation pipeline.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import io_utils
from pipeline import SciencePipeline, CARSPipeline
from llm_client import get_client

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)


def filter_topics_by_mode(topics: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """
    Filter topics by mode.
    
    Args:
        topics: List of all topics
        mode: "science" or "cars"
    
    Returns:
        Filtered topics list
    """
    if mode == "science":
        # Science topics are those with BB or CP prefixes (Biochemistry/Biology/Chemistry/Physics)
        return [t for t in topics if t.get("topic_id", "").startswith(("BB_", "CP_"))]
    elif mode == "cars":
        # CARS topics are those with CARS_ prefix or PS_ (Psychology/Sociology) for CARS-like content
        return [t for t in topics if t.get("topic_id", "").startswith(("CARS_", "PS_"))]
    else:
        return []


def run_science_pipeline(topics: List[Dict[str, Any]]):
    """Run science pipeline for all relevant topics."""
    science_topics = filter_topics_by_mode(topics, "science")
    logger.info(f"Found {len(science_topics)} science topics")
    
    total_items = 0
    
    for topic in science_topics:
        topic_id = topic.get("topic_id", "unknown")
        
        # Check if we already have enough items
        existing = io_utils.count_existing_items(topic_id, "science")
        if existing >= config.SCIENCE_TARGET_PER_TOPIC:
            logger.info(f"Topic {topic_id} already has {existing} items, skipping")
            continue
        
        # Run pipeline
        pipeline = SciencePipeline(topic)
        accepted = pipeline.run()
        
        total_items += len(accepted)
        logger.info(f"Topic {topic_id}: accepted {len(accepted)} items")
    
    logger.info(f"Science pipeline complete. Total items: {total_items}")


def run_cars_pipeline(topics: List[Dict[str, Any]]):
    """Run CARS pipeline for all relevant topics."""
    cars_topics = filter_topics_by_mode(topics, "cars")
    logger.info(f"Found {len(cars_topics)} CARS topics")
    
    total_sets = 0
    
    for topic in cars_topics:
        topic_id = topic.get("topic_id", "unknown")
        
        # Check if we already have enough sets
        existing = io_utils.count_existing_items(topic_id, "cars")
        if existing >= config.CARS_TARGET_PER_TOPIC:
            logger.info(f"Topic {topic_id} already has {existing} sets, skipping")
            continue
        
        # Run pipeline
        pipeline = CARSPipeline(topic)
        accepted = pipeline.run()
        
        total_sets += len(accepted)
        logger.info(f"Topic {topic_id}: accepted {len(accepted)} sets")
    
    logger.info(f"CARS pipeline complete. Total sets: {total_sets}")


def main():
    """Main entry point."""
    logger.info("Starting MCAT question generation pipeline")
    
    # Create output directories
    io_utils.ensure_output_dirs()
    
    # Test LLM connection
    logger.info(f"Testing LLM connection with model: {config.MODEL_NAME}")
    client = get_client()
    test_response = client.generate("Say 'Connection successful' if you can read this.")
    if test_response:
        logger.info(f"LLM connection successful: {test_response[:50]}...")
    else:
        logger.warning("LLM connection test failed. Check your configuration.")
    
    # Load topics
    topics = io_utils.load_topics()
    if not topics:
        logger.error("No topics loaded. Exiting.")
        return
    
    logger.info(f"Loaded {len(topics)} topics")
    
    # Run pipelines based on config or command line args
    import argparse
    
    parser = argparse.ArgumentParser(description="MCAT Question Generation")
    parser.add_argument(
        "--mode",
        choices=["science", "cars", "both"],
        default="both",
        help="Which mode to run"
    )
    parser.add_argument(
        "--topic",
        help="Specific topic ID to process (optional)"
    )
    
    args = parser.parse_args()
    
    if args.topic:
        # Process single topic
        topic = next((t for t in topics if t.get("topic_id") == args.topic), None)
        if not topic:
            logger.error(f"Topic {args.topic} not found")
            return
        
        mode = "science" if topic.get("topic_id", "").startswith(("BB_", "CP_")) else "cars"
        logger.info(f"Processing single topic {args.topic} in {mode} mode")
        
        if mode == "science":
            pipeline = SciencePipeline(topic)
            pipeline.run()
        else:
            pipeline = CARSPipeline(topic)
            pipeline.run()
    
    else:
        # Run all topics
        if args.mode in ["science", "both"]:
            run_science_pipeline(topics)
        
        if args.mode in ["cars", "both"]:
            run_cars_pipeline(topics)
    
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()