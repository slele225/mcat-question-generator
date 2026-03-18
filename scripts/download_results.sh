#!/usr/bin/env bash
# Copy generated outputs from remote GPU machine to local.
# Set REMOTE_USER, REMOTE_HOST, and optionally REMOTE_PATH before running.

REMOTE_USER="${REMOTE_USER:-user}"
REMOTE_HOST="${REMOTE_HOST:-your-gpu-server}"
REMOTE_PATH="${REMOTE_PATH:-/path/to/mcat-question-generator}"
LOCAL_DIR="${LOCAL_DIR:-./mcat_results}"

mkdir -p "$LOCAL_DIR"
echo "Downloading from $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/data/output/ to $LOCAL_DIR/"
rsync -avz --progress \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/data/output/" \
  "$LOCAL_DIR/"

echo "Done. Results in $LOCAL_DIR/"
