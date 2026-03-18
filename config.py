"""
Simple configuration for MCAT question generation pipeline.
Edit these values to change model, output paths, and generation parameters.
"""

import os

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

# Choose backend: "openai", "vllm", "openrouter", or any OpenAI-compatible API
BACKEND_TYPE = "openai"  # Options: "openai", "vllm", "openrouter"

# Model name (adjust based on your backend)
MODEL_NAME = "gpt-3.5-turbo"  # For OpenAI
# MODEL_NAME = "qwen/Qwen2.5-72B-Instruct"  # For vLLM
# MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"  # For vLLM
# MODEL_NAME = "openrouter/anthropic/claude-3.5-sonnet"  # For OpenRouter

# API endpoint (adjust based on your backend)
# For OpenAI: default is fine
# For local vLLM: http://localhost:8000/v1
# For OpenRouter: https://openrouter.ai/api/v1
BASE_URL = None  # None = use default for the backend
# BASE_URL = "http://localhost:8000/v1"  # Example for local vLLM
# BASE_URL = "https://openrouter.ai/api/v1"  # Example for OpenRouter

# API key environment variable name
# Set this environment variable before running
API_KEY_ENV_VAR = "OPENAI_API_KEY"  # e.g., OPENAI_API_KEY, VLLM_API_KEY, etc.

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

# Generation targets
SCIENCE_TARGET_PER_TOPIC = 20  # Number of accepted science questions per topic
CARS_TARGET_PER_TOPIC = 2      # Number of accepted CARS sets per topic

# Batch sizes
SCIENCE_BATCH_SIZE = 5          # Generate 5 science questions at a time
CARS_PASSAGE_WORDS = 600        # Target passage length in words
CARS_QUESTIONS_PER_SET = 8      # Number of questions per CARS set

# Attempt limits
MAX_ATTEMPTS_PER_TOPIC = 10     # Max generation attempts per topic
MAX_REPAIR_ATTEMPTS = 1         # Max repair attempts per item (validator suggests 1)

# Duplicate detection
DUPLICATE_SIMILARITY_THRESHOLD = 0.8  # Threshold for near-duplicate detection

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Minimum validation score to accept an item (1-10)
MIN_ACCEPTABLE_SCORE = 7

# Whether to skip validation (for testing)
SKIP_VALIDATION = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
SAVE_FAILED_ITEMS = True  # Save rejected/revised items for inspection