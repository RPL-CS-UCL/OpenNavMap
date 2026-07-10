#!/usr/bin/env bash
set -euo pipefail
# Run frontier exploration benchmark on tunnel
# Grid: ~1262x1635 at 0.2m res, ~6% obstacles (XY=ground, Z=height)
# Start: world (col=1,   row=-5)  → grid (row=84,   col=470)
# Goal:  world (col=201, row=220) → grid (row=1209, col=1467)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCH="$SCRIPT_DIR/../frontier_explore_benchmark.py"
PCD="$SCRIPT_DIR/../data/tunnel.pcd"
OUT="$SCRIPT_DIR/../output/tunnel"

echo "=== Tunnel Frontier Benchmark ==="
echo "  PCD: $PCD"
echo "  output: $OUT"
echo

cd "$PROJ_DIR"
exec python -u "$BENCH" \
  --pcd "$PCD" \
  --output_dir "$OUT" \
  --start -15 -5 \
  --goal 201 132 \
  --res_m 0.3 \
  --k 10 \
  --seed 42 \
  --temperature 2.5 \
  --max_steps 8000 \
  "$@"
