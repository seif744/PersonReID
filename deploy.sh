#!/usr/bin/env bash
# =============================================================================
# deploy.sh -- one-command sync of this repo to the GPU box (replaces FileZilla).
#
# Uses the SSH access you already have. Set the target ONCE in a gitignored
# .deploy.env next to this script:
#
#     DEPLOY_TARGET=user@gpu-host:/home/you/Inference
#     # optional: DEPLOY_SSH_PORT=22
#
# Then just run:   ./deploy.sh
#
# It rsyncs code only (skips the local venv, git, qdrant data, videos, caches),
# so re-runs transfer just what changed -- seconds, not a FileZilla drag-drop.
# It does NOT delete anything on the server. It does NOT touch the server's
# Python env: install deps there once (python -m pip install -r requirements.txt)
# with a CUDA torch build.
# =============================================================================
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/.deploy.env" ] && set -a && source "$here/.deploy.env" && set +a
: "${DEPLOY_TARGET:?Set DEPLOY_TARGET in .deploy.env (e.g. user@gpu-host:/path/Inference)}"

port="${DEPLOY_SSH_PORT:-22}"

rsync -avz --human-readable \
  -e "ssh -p ${port}" \
  --exclude '.venv/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'qdrant_data/' \
  --exclude 'qdrant_storage/' \
  --exclude '*.avi' \
  --exclude '*.mp4' \
  --exclude 'recorded_*' \
  --exclude 'output_*' \
  --exclude '.deploy.env' \
  "$here/" "$DEPLOY_TARGET/"

echo "==> Deployed code to $DEPLOY_TARGET"
echo "    On the server:  cd <path> && python main.py --mode live --videos rtsp://..."
