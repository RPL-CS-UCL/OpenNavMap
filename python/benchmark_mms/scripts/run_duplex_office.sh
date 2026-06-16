#!/usr/bin/env bash
set -euo pipefail
# Run frontier exploration benchmark on duplex_office
# Grid: 207x207 at 0.1m res, 44.0% obstacles
# Start: world (2.0, 2.5) → grid (25, 20)
# Goal:  world (19.0, 18.0) → grid (180,190)
# MUST use 0.1m: 0.2m fragments free space into 390 disconnected components

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCH="$SCRIPT_DIR/../frontier_explore_benchmark.py"
PCD="$SCRIPT_DIR/../data/duplex_office.pcd"
OUT="$SCRIPT_DIR/../output/duplex_office"

echo "=== Duplex Office Frontier Benchmark ==="
echo "  PCD: $PCD"
echo "  output: $OUT"
echo

cd "$PROJ_DIR"
exec python -u "$BENCH" \
  --pcd "$PCD" \
  --output_dir "$OUT" \
  --start_world 2 2.5 \
  --goal_world 19 18 \
  --res_m 0.1 \
  --dilate 0 \
  --k 5 \
  --seed 42 \
  "$@"
