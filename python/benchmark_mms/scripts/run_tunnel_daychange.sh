#!/usr/bin/env bash
set -euo pipefail
# Run day-change frontier benchmark on tunnel.
# Day0 adds a blocking obstacle in world coordinates; day1 removes it.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCH="$SCRIPT_DIR/../frontier_explore_benchmark.py"
PCD="$SCRIPT_DIR/../data/tunnel.pcd"
OUT="$SCRIPT_DIR/../output/tunnel_daychange"

echo "=== Tunnel Day-Change Frontier Benchmark ==="
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
  --obstacle_block 175.0 25.0 180.0 100.0 \
  --day_change 2 \
  "$@"
