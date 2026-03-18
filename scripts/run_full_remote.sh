#!/usr/bin/env bash
# One script to run everything on the GPU server: start vLLM, then the pipeline.
# Both run in the background with nohup so you can disconnect from SSH.
# Run from repo root: ./scripts/run_full_remote.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/output/logs data/output/science data/output/cars data/output/checkpoints data/output/failed

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

export VLLM_API_KEY="${VLLM_API_KEY:-default-key}"
PORT="${VLLM_PORT:-8000}"
VLLM_LOG="data/output/logs/vllm.log"
PIPELINE_LOG="data/output/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "Starting vLLM in background (log: $VLLM_LOG)..."
nohup "$REPO_ROOT/scripts/start_vllm.sh" >> "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM on 127.0.0.1:$PORT (up to ~5 min)..."
for i in {1..60}; do
  if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 200; then
    echo "vLLM is up."
    break
  fi
  if [ $i -eq 60 ]; then
    echo "Timeout waiting for vLLM. Check $VLLM_LOG"
    exit 1
  fi
  sleep 5
done

echo "Starting pipeline in background (log: $PIPELINE_LOG)..."
nohup python run_generation.py --mode both >> "$PIPELINE_LOG" 2>&1 &
PIPELINE_PID=$!
echo "Pipeline PID: $PIPELINE_PID"

echo ""
echo "Done. Both are running. You can disconnect from SSH."
echo "  vLLM:    tail -f $VLLM_LOG"
echo "  pipeline: tail -f $PIPELINE_LOG"
echo "  results: data/output/science/  data/output/cars/"
