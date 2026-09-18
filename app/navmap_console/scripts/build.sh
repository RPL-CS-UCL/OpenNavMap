#!/usr/bin/env bash
# Install frontend deps and build frontend/dist (served by scripts/serve.sh).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "${HERE}/frontend" && pnpm install --frozen-lockfile && pnpm build
