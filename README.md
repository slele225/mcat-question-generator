# MCAT Question Generation Pipeline

A simple, modular pipeline for generating MCAT practice questions using LLMs.

## Overview

- **Science**: Standalone multiple-choice questions (four options + explanation).
- **CARS**: Passages with 8–10 questions (Critical Analysis and Reasoning Skills).

The pipeline uses a generate–validate–repair loop with configurable quality thresholds.

## Repo structure

```
.
├── config.py          # Model, paths, and generation parameters
├── schemas.py         # JSON schemas for validation
├── prompt_templates.py
├── llm_client.py      # OpenAI-compatible API client
├── generator.py       # Science and CARS generation
├── validator.py       # Quality validation
├── repair.py          # Repair for items needing revision
├── dedupe.py          # Duplicate detection
├── io_utils.py        # File I/O
├── pipeline.py        # Orchestration
├── run_generation.py  # Entry point
├── requirements.txt
├── data/
│   ├── topics.json    # Input: topic list (add your own)
│   └── output/        # Generated outputs (git-ignored)
│       ├── science/   # Science JSONL files
│       ├── cars/      # CARS JSONL files
│       ├── checkpoints/
│       └── failed/
└── README.md
```

## Setup with uv

```bash
uv venv
# Activate: .venv\Scripts\activate (Windows) or source .venv/bin/activate (macOS/Linux)
uv pip install -r requirements.txt
```

## Setup with pip

```bash
python -m venv .venv
# Activate the venv, then:
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set:

- **BACKEND_TYPE** – `"openai"`, `"vllm"`, or `"openrouter"`
- **MODEL_NAME** – Model identifier for your backend
- **BASE_URL** – API base URL (e.g. `http://localhost:8000/v1` for vLLM)
- **API_KEY_ENV_VAR** – Environment variable name for the API key (e.g. `OPENAI_API_KEY`)
- **TOPICS_FILE** – Path to topics JSON (default `data/topics.json`)
- **OUTPUT_DIR** – Base output directory (default `data/output`)
- Generation targets, batch sizes, and validation threshold

Set your API key in the environment before running, e.g.:

```bash
export OPENAI_API_KEY="your-key"
```

## Local usage

1. Add a topics file at `data/topics.json` (list of topic dicts with at least `topic_id` and any fields your prompts use).
2. Run:

```bash
python run_generation.py
```

Options:

- `--mode science` – Science only  
- `--mode cars` – CARS only  
- `--mode both` – Both (default)  
- `--topic TOPIC_ID` – Run for a single topic

Outputs are written under `data/output/` (see structure above).

## Output locations

| Content        | Path                          |
|----------------|-------------------------------|
| Science JSONL  | `data/output/science/`        |
| CARS JSONL     | `data/output/cars/`           |
| Checkpoints    | `data/output/checkpoints/`     |
| Failed items   | `data/output/failed/`         |

## Switching model backends

In `config.py`, set **BACKEND_TYPE** and **BASE_URL** (and **MODEL_NAME** as needed). The client in `llm_client.py` is OpenAI-compatible, so it works with OpenAI, vLLM, OpenRouter, or any compatible API.

## Google Colab

1. Upload the repo (or clone from GitHub) into the Colab environment.
2. Install dependencies in a cell:

```python
!pip install -r requirements.txt
```

3. Set your API key (e.g. in a Colab secret or plain variable):

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"  # or use a secret
```

4. Optionally mount Drive and set the working directory to the repo folder.
5. Run generation:

```python
!python run_generation.py --mode both
```

Outputs will appear under `data/output/`. To keep them after the session, copy `data/output/` to Google Drive.

For a minimal step-by-step notebook, see `colab_generate_mcat.ipynb`.
