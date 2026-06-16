#!/usr/bin/env bash
set -euo pipefail
# Run frontier exploration benchmark on octa_maze
# Grid: ~176x176 at 0.2m res, 32.7% obstacles
# Start: (18, 12) grid → world (2.4, 3.6)
# Goal:  (155,162) grid → world (32.4, 31.0)

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
  --start 18 12 \
  --goal 155 162 \
  --res_m 0.2 \
  --k 5 \
  --seed 42 \
  "$@"
