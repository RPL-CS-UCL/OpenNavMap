#!/usr/bin/env bash
# Fail if any file under frontend/src or frontend/scripts is swallowed by the repo .gitignore
# (root has *vis*, log*, output*, paper*, cache/, tmp/, dataset/ globs).
set -euo pipefail
cd "$(dirname "$0")/.."
repo_root="$(git rev-parse --show-toplevel)"
rel="$(realpath --relative-to="$repo_root" "$PWD")"
ignored="$(cd "$repo_root" && git ls-files -oi --exclude-standard "$rel/src" "$rel/scripts" "$rel/index.html" "$rel/package.json" 2>/dev/null || true)"
if [ -n "$ignored" ]; then
  echo "ERROR: these files are ignored by .gitignore (rename them):" >&2
  echo "$ignored" >&2
  exit 1
fi
echo "check-ignored: ok"
