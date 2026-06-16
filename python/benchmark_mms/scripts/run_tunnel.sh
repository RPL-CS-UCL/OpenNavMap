#!/usr/bin/env bash
set -euo pipefail
# Run frontier exploration benchmark on tunnel
# Grid: ~1636x1262 at 0.2m res (X+Y plane, binary PCD)
# Axes: col=0 (X), row=1 (Y), height=2 (Z)
# Start: world (-7, 25) → grid (~519, 30)
# Goal:  world (227, -60) → grid (~349, 498)
# The tunnel long axis is PCD-Y, so row_axis=1

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
  --col_axis 0 \
  --row_axis 1 \
  --height_axis 2 \
  --start_world -7 25 \
  --goal_world 227 -60 \
  --res_m 0.5 \
  --dilate 0 \
  --k 10 \
  --seed 42 \
  "$@"
