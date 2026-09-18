#!/usr/bin/env bash
# Production start: conda python + uvicorn, serving frontend/dist if built.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${NAVMAP_CONSOLE_PYTHON:-/root/miniconda3/envs/opennavmap/bin/python}"
export PYTHONPATH="${HERE}/backend:${HERE}/../../python:${HERE}/../../third_party/litevloc_code/python${PYTHONPATH:+:$PYTHONPATH}"
cd "${HERE}/backend"
exec "$PY" -m uvicorn navmap_console.main:app \
  --host "${NAVMAP_CONSOLE_HOST:-0.0.0.0}" --port "${NAVMAP_CONSOLE_PORT:-8765}" "$@"
