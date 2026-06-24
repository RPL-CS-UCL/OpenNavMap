#!/bin/bash
# Run hloc_sfm_netvlad_splg baseline (SFM build or submap merge) on map merging.
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
#   --mode sfm|merge        Required. sfm=build only, merge=merge+eval
#   --method NAME           HLoc SfM method (default: hloc_sfm_netvlad_splg)
#   --env NAME              Dataset environment: ucl_campus_aria,
#                           hkust_campus, vineyard (default: ucl_campus_aria)
#   --max-submaps N         Limit to first N submaps (default: 2)
#   --sfm-sample-dist F     SfM keyframe sampling distance in metres (default: 0.25)
#   --order-index N         Order index: 0=in, 1=r0, ..., 9=r8 (default: 0)
#   --data-dir NAME         Override data directory (default: s00000_aria_data_000)
#   --prebuilt-sfm-root DIR Pre-built submaps_sfm/ root; skips SfM rebuild in merge mode
#   --eval-config NAME      yaml config for run_evaluation.sh
#                           (default: OpenNavMap_map_merge.yaml)
#   --overwrite             Remove existing result dir before running
#   --clean-work            Delete large _work/merge_subN/ intermediates after each
#                           submap merge, and also clean safe leftovers after success
#   --dry-run               Print command and exit without running it
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
#   bash run_baseline.sh --mode merge --order-index 1
#
#   # Step 2: merge all submaps (no --max-submaps limit)
#   bash run_baseline.sh --mode merge --overwrite
#
#   # Step 2: merge _025 using pre-built SfM
#   bash run_baseline.sh --mode merge --env hkust_campus --sfm-sample-dist 0.25 \
#     --prebuilt-sfm-root /path/to/s00000_sfm_netvlad_splg_025 --overwrite
#
#   # Step 2: merge _390 with data_390 and pre-built SfM
#   bash run_baseline.sh --mode merge --sfm-sample-dist 3.90 \
#     --data-dir s00000_aria_data_390 \
#     --prebuilt-sfm-root /path/to/s00000_sfm_netvlad_splg_390 --overwrite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash run_baseline.sh --mode sfm|merge [OPTIONS]

Options:
  --mode sfm|merge        Required. sfm=build only, merge=merge+eval
  --method NAME           HLoc SfM method (default: hloc_sfm_netvlad_splg)
  --env NAME              Dataset environment: ucl_campus_aria, hkust_campus, vineyard
                           (default: ucl_campus_aria)
  --max-submaps N         Limit to first N submaps
  --sfm-sample-dist F     SfM keyframe sampling distance in metres (default: 0.25)
  --sfm-ba-iter N         BA iterations after SfM triangulation (default: 0)
  --num-retrieval N       Retrieval top-k for localization (default: 10)
  --geo-verify-min-matches N
                          Minimum geometric verification inliers (default: 150)
  --pnp-min-inliers N     Minimum PnP inliers for a successful frame (default: 50)
  --order-index N         Order index: 0=in, 1=r0, ..., 9=r8 (default: 0)
  --data-dir NAME         Override data directory (default: s00000_aria_data_000)
  --prebuilt-sfm-root DIR Pre-built submaps_sfm/ root; skips SfM rebuild in merge mode
  --eval-config NAME      yaml config for run_evaluation.sh
  --overwrite             Remove existing result dir before running
  --clean-work            Delete large _work/merge_subN/ intermediates after each
                          submap merge, and also delete _work/merge_sub* and
                          _work/*.h5 after a successful run
  --dry-run               Print command and exit without running it
  --help                  Show this help message
EOF
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
PYTHON=/root/miniconda3/envs/opennavmap/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/opennavmap/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/third_party/pose_estimation_models/estimator:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Fixed config
# ---------------------------------------------------------------------------
DATA_DIR=s00000_aria_data_000
TRAJ_EVAL_ROOT=/Titan/dataset/data_opennavmap/traj_eval_data/map_merge_eval_data
METHOD=hloc_sfm_netvlad_splg
NUM_RETRIEVAL=10
GEO_VERIFY_MIN_MATCHES=150
PNP_MIN_INLIERS=50

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODE=
ENV=ucl_campus_aria
ORDER_INDEX=0
MAX_SUBMAPS=
SFM_SAMPLE_DIST=0.25
SFM_BA_ITER=0
EVAL_CONFIG=OpenNavMap_map_merge.yaml
OVERWRITE=
PREBUILT_SFM_ROOT=
CLEAN_WORK=
DRY_RUN=

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)          usage; exit 0 ;;
    --mode)             MODE=$2; shift ;;
    --method)           METHOD=$2; shift ;;
    --env)              ENV=$2; shift ;;
    --max-submaps)      MAX_SUBMAPS=$2; shift ;;
    --sfm-sample-dist)  SFM_SAMPLE_DIST=$2; shift ;;
    --sfm-ba-iter)      SFM_BA_ITER=$2; shift ;;
    --num-retrieval)    NUM_RETRIEVAL=$2; shift ;;
    --geo-verify-min-matches) GEO_VERIFY_MIN_MATCHES=$2; shift ;;
    --pnp-min-inliers)  PNP_MIN_INLIERS=$2; shift ;;
    --order-index)      ORDER_INDEX=$2; shift ;;
    --data-dir)         DATA_DIR=$2; shift ;;
    --prebuilt-sfm-root) PREBUILT_SFM_ROOT=$2; shift ;;
    --eval-config)      EVAL_CONFIG=$2; shift ;;
    --overwrite)        OVERWRITE=1 ;;
    --clean-work)       CLEAN_WORK=1 ;;
    --dry-run)          DRY_RUN=1 ;;
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
case "$METHOD" in
  hloc_sfm_netvlad_splg|hloc_sfm_netvlad_disk_dilg) ;;
  *)
    echo "Error: --method must be one of: hloc_sfm_netvlad_splg, hloc_sfm_netvlad_disk_dilg. Got '$METHOD'."
    exit 1
    ;;
esac

case "$ENV" in
  ucl_campus_aria|hkust_campus|vineyard) ;;
  *)
    echo "Error: --env must be one of: ucl_campus_aria, hkust_campus, vineyard. Got '$ENV'."
    exit 1
    ;;
esac

DATASET_ROOT=/Titan/dataset/data_opennavmap/map_multisession_eval/${ENV}

# ---------------------------------------------------------------------------
# Result directory (mirrors _build_result_root() in run_baseline.py)
# dist_tag: sfm_sample_dist * 100, zero-padded to 3 digits; empty if 0
# ---------------------------------------------------------------------------
ORDER_TAGS=(in r0 r1 r2 r3 r4 r5 r6 r7 r8)
ORDER_TAG="${ORDER_TAGS[$ORDER_INDEX]}"
if [[ -n "$MAX_SUBMAPS" ]]; then
  ORDER_TAG="${ORDER_TAG}_${MAX_SUBMAPS}sub"
fi
DIST_INT=$(echo "$SFM_SAMPLE_DIST * 100 / 1" | bc)
if [[ "$DIST_INT" -gt 0 ]]; then
  DIST_TAG=$(printf "_%03d" "$DIST_INT")
else
  DIST_TAG=""
fi

if [[ "$MODE" == "sfm" ]]; then
  SFM_TAG="${METHOD#hloc_sfm_}"
  RESULT_DIR="${DATASET_ROOT}/s00000_sfm_${SFM_TAG}${DIST_TAG}"
else
  if [[ "$METHOD" == "hloc_sfm_netvlad_disk_dilg" && "$NUM_RETRIEVAL" == "10" && "$GEO_VERIFY_MIN_MATCHES" == "300" && "$PNP_MIN_INLIERS" == "70" ]]; then
    VALUE_TAG="value0"
  elif [[ "$METHOD" == "hloc_sfm_netvlad_disk_dilg" && "$NUM_RETRIEVAL" == "10" && "$GEO_VERIFY_MIN_MATCHES" == "400" && "$PNP_MIN_INLIERS" == "110" ]]; then
    VALUE_TAG="value1"
  elif [[ "$METHOD" == "hloc_sfm_netvlad_disk_dilg" && "$NUM_RETRIEVAL" == "10" && "$GEO_VERIFY_MIN_MATCHES" == "500" && "$PNP_MIN_INLIERS" == "150" ]]; then
    VALUE_TAG="value2"
  elif [[ "$NUM_RETRIEVAL" == "10" && "$GEO_VERIFY_MIN_MATCHES" == "100" && "$PNP_MIN_INLIERS" == "25" ]]; then
    VALUE_TAG="value0"
  elif [[ "$NUM_RETRIEVAL" == "10" && "$GEO_VERIFY_MIN_MATCHES" == "120" && "$PNP_MIN_INLIERS" == "35" ]]; then
    VALUE_TAG="value1"
  elif [[ "$NUM_RETRIEVAL" == "10" && "$GEO_VERIFY_MIN_MATCHES" == "150" && "$PNP_MIN_INLIERS" == "50" ]]; then
    VALUE_TAG="value2"
  else
    VALUE_TAG="nr${NUM_RETRIEVAL}_gv${GEO_VERIFY_MIN_MATCHES}_pnp${PNP_MIN_INLIERS}"
  fi
  RESULT_DIR="${DATASET_ROOT}/s00000_results_${ORDER_TAG}_${METHOD}${DIST_TAG}_${VALUE_TAG}"
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD=(
  "$PYTHON"
  "${PROJECT_ROOT}/python/benchmark_map_merge/run_baseline.py"
  --dataset-root    "$DATASET_ROOT"
  --method          "$METHOD"
  --order-index     "$ORDER_INDEX"
  --data-dir        "$DATA_DIR"
  --traj-eval-data-root "$TRAJ_EVAL_ROOT"
  --sfm-sample-dist "$SFM_SAMPLE_DIST"
  --sfm-ba-iter "$SFM_BA_ITER"
  --num-retrieval "$NUM_RETRIEVAL"
  --geo-verify-min-matches "$GEO_VERIFY_MIN_MATCHES"
  --pnp-min-inliers "$PNP_MIN_INLIERS"
)
[[ "$MODE" == "sfm" ]]   && CMD+=(--submap-sfm)
[[ "$MODE" == "merge" ]] && CMD+=(--submap-merge)
[[ -n "$MAX_SUBMAPS" ]]  && CMD+=(--max-submaps "$MAX_SUBMAPS")
[[ -n "$OVERWRITE" ]]    && CMD+=(--overwrite)
[[ -n "$CLEAN_WORK" ]]   && CMD+=(--clean-work)
[[ -n "$PREBUILT_SFM_ROOT" ]] && CMD+=(--prebuilt-sfm-root "$PREBUILT_SFM_ROOT")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "=== run_baseline.sh ==="
echo "Mode       : $MODE"
echo "Env        : $ENV"
echo "Result dir : $RESULT_DIR"
echo "Command    : ${CMD[*]}"
echo ""

if [[ -n "$DRY_RUN" ]]; then
  exit 0
fi

"${CMD[@]}"

if [[ -n "$CLEAN_WORK" && -d "${RESULT_DIR}/_work" ]]; then
  echo ""
  echo "=== Cleaning safe _work intermediates ==="
  find "${RESULT_DIR}/_work" -maxdepth 1 -type d -name 'merge_sub*' -prune -exec rm -rf {} +
  find "${RESULT_DIR}/_work" -maxdepth 1 -type f -name '*.h5' -delete
fi

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
