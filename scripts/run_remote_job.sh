#!/usr/bin/env bash
# Run the full pipeline on a remote GPU so it survives SSH disconnect.
# Prereq: vLLM server already running (e.g. in tmux: run scripts/start_vllm.sh).
# This script starts the pipeline under nohup so you can exit SSH.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/output/logs data/output/science data/output/cars data/output/checkpoints data/output/failed

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

export VLLM_API_KEY="${VLLM_API_KEY:-default-key}"

LOG_FILE="data/output/logs/remote_job_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="data/output/logs/remote_job.pid"

nohup python run_generation.py --mode both >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Pipeline started in background. PID=$(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
echo "You can disconnect from SSH; the job will keep running."
echo "To check later: tail -f $LOG_FILE"
