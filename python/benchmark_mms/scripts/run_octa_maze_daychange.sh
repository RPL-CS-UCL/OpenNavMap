#!/usr/bin/env bash
set -euo pipefail
# Run day-change frontier benchmark on octa_maze.
# Day0 adds a blocking obstacle in world coordinates; day1 removes it.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCH="$SCRIPT_DIR/../frontier_explore_benchmark.py"
PCD="$SCRIPT_DIR/../data/octa_maze.pcd"
OUT="$SCRIPT_DIR/../output/octa_maze_daychange"

echo "=== Octa Maze Day-Change Frontier Benchmark ==="
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
  --max_steps 1000 \
  --obstacle_block 0.0 17.0 35.0 18.0 \
  --day_change 2 \
  "$@"
