"""
Simple configuration for MCAT question generation pipeline.
Edit these values to change model, output paths, and generation parameters.
"""

import os

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

# Backend type:
# - "vllm" for local / remote vLLM OpenAI-compatible server
# - "openai" for OpenAI API
# - "openrouter" for OpenRouter
BACKEND_TYPE = "vllm"

# Strong default Qwen model for a big GPU box
MODEL_NAME = "Qwen/Qwen3-32B"

# vLLM OpenAI-compatible endpoint
BASE_URL = "http://localhost:8000/v1"

# vLLM usually requires an API key flag; this can be any dummy string
API_KEY_ENV_VAR = "VLLM_API_KEY"

# Generation parameters
TEMPERATURE = 0.7
MAX_TOKENS = 4096
TOP_P = 0.95
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0

# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------

# Input file
TOPICS_FILE = "data/topics.json"

# Output directories
OUTPUT_DIR = "data/output"
SCIENCE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "science")
CARS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cars")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FAILED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "failed")

# Generation targets
SCIENCE_TARGET_PER_TOPIC = 20
CARS_TARGET_PER_TOPIC = 2

# Batch sizes
SCIENCE_BATCH_SIZE = 5
CARS_PASSAGE_WORDS = 600
CARS_QUESTIONS_PER_SET = 8

# Attempt limits
MAX_ATTEMPTS_PER_TOPIC = 10
MAX_REPAIR_ATTEMPTS = 1

# Duplicate detection
DUPLICATE_SIMILARITY_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Minimum validation score to accept an item (1-10)
MIN_ACCEPTABLE_SCORE = 7

# Whether to skip validation (useful only for testing)
SKIP_VALIDATION = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"
SAVE_FAILED_ITEMS = True
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")