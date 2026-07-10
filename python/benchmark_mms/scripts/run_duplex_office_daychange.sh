#!/usr/bin/env bash
set -euo pipefail
# Run day-change frontier benchmark on duplex_office.
# Day0 adds a blocking obstacle in world coordinates; day1 removes it.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCH="$SCRIPT_DIR/../frontier_explore_benchmark.py"
PCD="$SCRIPT_DIR/../data/duplex_office.pcd"
OUT="$SCRIPT_DIR/../output/duplex_office_daychange"

echo "=== Duplex Office Day-Change Frontier Benchmark ==="
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
  --max_steps 1000 \
  --obstacle_block 5.0 10.0 20.0 11.0 \
  --day_change 2 \
  "$@"
