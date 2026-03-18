#!/usr/bin/env bash
# Run the MCAT generation pipeline. Logs to data/output/logs/.
# Activate uv venv if .venv exists; ensure output dirs; run both modes.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/output/logs data/output/science data/output/cars data/output/checkpoints data/output/failed

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

export VLLM_API_KEY="${VLLM_API_KEY:-default-key}"

LOG_FILE="data/output/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"
exec python run_generation.py --mode both 2>&1 | tee "$LOG_FILE"
