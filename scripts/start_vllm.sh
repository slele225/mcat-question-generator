#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for this repo.
# Edit MODEL and PORT below if needed. Uses VLLM_API_KEY from env or a default.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${VLLM_MODEL:-Qwen/Qwen3-32B}"
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-default-key}"

echo "Starting vLLM: $MODEL on 127.0.0.1:$PORT"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --api-key "$API_KEY"
