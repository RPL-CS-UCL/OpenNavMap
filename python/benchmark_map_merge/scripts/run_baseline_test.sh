#!/bin/bash
# Run hloc_sfm_netvlad_splg baseline on ucl_campus_aria s00000.
#
# Usage:
#   bash run_baseline_test.sh [OPTIONS]
#
# Options:
#   --overwrite           Remove existing result dir before running
#   --max-submaps N       Limit to first N submaps (default: 2)
#   --global-ba-iter N    Global BA iterations (default: 0)
#   --order-index N       Order index 0=in, 1=r0, ... (default: 0)
#
# Examples:
#   bash run_baseline_test.sh
#   bash run_baseline_test.sh --overwrite
#   bash run_baseline_test.sh --max-submaps 3 --overwrite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON=/root/miniconda3/envs/opennavmap/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/opennavmap/lib:${LD_LIBRARY_PATH:-}

DATASET_ROOT=/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria
DATA_DIR=s00000_aria_full_data
TRAJ_EVAL_ROOT=/Titan/dataset/data_opennavmap/traj_eval_data/map_merge_eval_data
METHOD=hloc_sfm_netvlad_splg

ORDER_INDEX=0
MAX_SUBMAPS=2
GLOBAL_BA_ITER=0
OVERWRITE=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      ;;
    --max-submaps)
      MAX_SUBMAPS=$2
      shift
      ;;
    --global-ba-iter)
      GLOBAL_BA_ITER=$2
      shift
      ;;
    --order-index)
      ORDER_INDEX=$2
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

ORDER_TAGS=(in r0 r1 r2 r3 r4 r5 r6 r7 r8)
ORDER_TAG="${ORDER_TAGS[$ORDER_INDEX]}"
if [[ -n "$MAX_SUBMAPS" ]]; then
  ORDER_TAG="${ORDER_TAG}_${MAX_SUBMAPS}sub"
fi
DATA_SUFFIX="_full_data"
RESULT_DIR="${DATASET_ROOT}/s00000_results_${ORDER_TAG}_${METHOD}${DATA_SUFFIX}"

CMD=(
  "$PYTHON"
  "${PROJECT_ROOT}/python/benchmark_map_merge/run_baseline.py"
  --dataset-root "$DATASET_ROOT"
  --method "$METHOD"
  --order-index "$ORDER_INDEX"
  --data-dir "$DATA_DIR"
  --traj-eval-data-root "$TRAJ_EVAL_ROOT"
  --global-ba-iter "$GLOBAL_BA_ITER"
)
if [[ -n "$MAX_SUBMAPS" ]]; then
  CMD+=(--max-submaps "$MAX_SUBMAPS")
fi
if [[ -n "$OVERWRITE" ]]; then
  CMD+=(--overwrite)
fi

echo "=== run_baseline_test.sh ==="
echo "Result dir : ${RESULT_DIR}"
echo "Command    : ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "=== metrics/summary.json ==="
cat "${RESULT_DIR}/metrics/summary.json"
echo ""
