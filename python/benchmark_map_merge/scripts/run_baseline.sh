#!/bin/bash
# Run hloc_sfm_netvlad_splg baseline (SFM build or submap merge) on ucl_campus_aria s00000.
#
# Two-step workflow:
#   Step 1 (--mode sfm)   – build SFM for each submap independently
#   Step 2 (--mode merge) – merge pre-built SFM submaps; runs trajectory
#                           evaluation automatically via run_evaluation.sh
#
# Usage:
#   bash run_baseline.sh --mode sfm|merge [OPTIONS]
#
# Options:
#   --mode sfm|merge      Required. sfm=build only, merge=merge+eval
#   --max-submaps N       Limit to first N submaps (default: 2)
#   --sfm-ba-iter N       SfM bundle-adjustment iterations (default: 0)
#   --order-index N       Order index: 0=in, 1=r0, ..., 9=r8 (default: 0)
#   --eval-config NAME    yaml config for run_evaluation.sh
#                         (default: map_merge.yaml)
#   --overwrite           Remove existing result dir before running
#
# Examples:
#   # Step 1: build SFM for all submaps
#   bash run_baseline.sh --mode sfm
#
#   # Step 1: build SFM for first 2 submaps, overwrite
#   bash run_baseline.sh --mode sfm --max-submaps 2 --overwrite
#
#   # Step 2: merge sub0+sub1, overwrite, then evaluate against GT
#   bash run_baseline.sh --mode merge --max-submaps 2 --overwrite
#
#   # Step 2: merge with different order and evaluate with full config
#   bash run_baseline.sh --mode merge --order-index 1 --eval-config map_merge.yaml
#
#   # Step 2: merge all submaps (no --max-submaps limit)
#   bash run_baseline.sh --mode merge --overwrite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
PYTHON=/root/miniconda3/envs/opennavmap/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/opennavmap/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/third_party/pose_estimation_models/estimator:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Fixed config
# ---------------------------------------------------------------------------
DATASET_ROOT=/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria
DATA_DIR=s00000_aria_full_data
TRAJ_EVAL_ROOT=/Titan/dataset/data_opennavmap/traj_eval_data/map_merge_eval_data
METHOD=hloc_sfm_netvlad_splg

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODE=
ORDER_INDEX=0
MAX_SUBMAPS=
SFM_BA_ITER=0
EVAL_CONFIG=map_merge.yaml
OVERWRITE=

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)         MODE=$2; shift ;;
    --max-submaps)  MAX_SUBMAPS=$2; shift ;;
    --sfm-ba-iter)  SFM_BA_ITER=$2; shift ;;
    --order-index)  ORDER_INDEX=$2; shift ;;
    --eval-config)  EVAL_CONFIG=$2; shift ;;
    --overwrite)    OVERWRITE=1 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

if [[ -z "$MODE" ]]; then
  echo "Error: --mode sfm|merge is required."
  echo "Run: bash run_baseline.sh --help  (or read the header of this script)"
  exit 1
fi
if [[ "$MODE" != "sfm" && "$MODE" != "merge" ]]; then
  echo "Error: --mode must be 'sfm' or 'merge', got '$MODE'."
  exit 1
fi

# ---------------------------------------------------------------------------
# Result directory (mirrors _get_result_root() in run_baseline.py)
# ---------------------------------------------------------------------------
ORDER_TAGS=(in r0 r1 r2 r3 r4 r5 r6 r7 r8)
ORDER_TAG="${ORDER_TAGS[$ORDER_INDEX]}"
DATA_LABEL="${DATA_DIR#s00000_aria_}"   # e.g. full_data
DATA_SUFFIX="_${DATA_LABEL}"            # e.g. _full_data

if [[ "$MODE" == "sfm" ]]; then
  RESULT_DIR="${DATASET_ROOT}/s00000_sfm_${DATA_LABEL}_sba${SFM_BA_ITER}"
else
  RESULT_DIR="${DATASET_ROOT}/s00000_results_${ORDER_TAG}_${METHOD}${DATA_SUFFIX}_sba${SFM_BA_ITER}"
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD=(
  "$PYTHON"
  "${PROJECT_ROOT}/python/benchmark_map_merge/run_baseline.py"
  --dataset-root "$DATASET_ROOT"
  --method       "$METHOD"
  --order-index  "$ORDER_INDEX"
  --data-dir     "$DATA_DIR"
  --traj-eval-data-root "$TRAJ_EVAL_ROOT"
  --sfm-ba-iter  "$SFM_BA_ITER"
)
[[ "$MODE" == "sfm" ]]   && CMD+=(--submap-sfm)
[[ "$MODE" == "merge" ]] && CMD+=(--submap-merge)
[[ -n "$MAX_SUBMAPS" ]]  && CMD+=(--max-submaps "$MAX_SUBMAPS")
[[ -n "$OVERWRITE" ]]    && CMD+=(--overwrite)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "=== run_baseline.sh ==="
echo "Mode       : $MODE"
echo "Result dir : $RESULT_DIR"
echo "Command    : ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "=== metrics/summary.json ==="
cat "${RESULT_DIR}/metrics/summary.json"
echo ""

# ---------------------------------------------------------------------------
# Trajectory evaluation (merge mode only)
# ---------------------------------------------------------------------------
if [[ "$MODE" == "merge" ]]; then
  echo "=== Running trajectory evaluation ==="
  bash "${SCRIPT_DIR}/run_evaluation.sh" --config "$EVAL_CONFIG"
fi
