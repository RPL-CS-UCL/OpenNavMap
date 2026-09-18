#!/usr/bin/env bash
# Backend with reload on 8765 + Vite dev server on 5173 (proxies /api and /ws).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
trap 'kill 0' EXIT
bash "${HERE}/scripts/serve.sh" --reload --reload-dir "${HERE}/backend/navmap_console" &
(cd "${HERE}/frontend" && pnpm dev) &
wait
