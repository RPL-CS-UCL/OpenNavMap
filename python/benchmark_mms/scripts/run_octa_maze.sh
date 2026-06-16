#!/usr/bin/env bash
set -euo pipefail
# Run frontier exploration benchmark on octa_maze
# Grid: ~176x176 at 0.2m res, ~32% obstacles (XY=ground, Z=height)
# Start: world (col=2.5, row=4)   → grid (row=20, col=12)
# Goal:  world (col=30,  row=33)  → grid (row=165, col=150)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCH="$SCRIPT_DIR/../frontier_explore_benchmark.py"
PCD="$SCRIPT_DIR/../data/octa_maze.pcd"
OUT="$SCRIPT_DIR/../output/octa_maze"

echo "=== Octa Maze Frontier Benchmark ==="
echo "  PCD: $PCD"
echo "  output: $OUT"
echo

cd "$PROJ_DIR"
exec python -u "$BENCH" \
  --pcd "$PCD" \
  --output_dir "$OUT" \
  --start 2.5 4 \
  --goal 30 33 \
  --res_m 0.2 \
  --k 5 \
  --seed 42 \
  --temperature 2.5 \
  "$@"
