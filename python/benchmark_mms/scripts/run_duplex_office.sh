#!/usr/bin/env bash
set -euo pipefail
# Run frontier exploration benchmark on duplex_office
# Grid: ~104x104 at 0.2m res, ~32% obstacles (XY=ground, Z=height)
# Start: world (col=1.5, row=2.5) → grid (row=12, col=7)
# Goal:  world (col=19,  row=18)  → grid (row=90, col=95)

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
  --start 1.5 2.5 \
  --goal 19 18 \
  --res_m 0.2 \
  --k 5 \
  --seed 42 \
  --temperature 2.5 \
  "$@"
