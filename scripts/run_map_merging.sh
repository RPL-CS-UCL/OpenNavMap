#!/bin/bash

# Usage:
#   bash run_map_merging.sh <SCENE> <ORDER> <METHOD> <POSE_EST> [IQA] [IG] [TD] [MAX_SUBMAPS]
#   IQA/IG/TD: 1=enabled, 0=disabled, default=1
#   MAX_SUBMAPS: default=all
#
# Environment overrides:
#   DATASET_ROOT, OUTPUT_ROOT, DATA_DIR, TRAJ_EVAL_ROOT, EVAL_CONFIG

set -euo pipefail

PROJECT_PATH="/Titan/code/robohike_ws/src/opennavmap"
DATASET_ROOT=${DATASET_ROOT:-/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria}
OUTPUT_ROOT=${OUTPUT_ROOT:-$DATASET_ROOT}
TRAJ_EVAL_ROOT=${TRAJ_EVAL_ROOT:-/Titan/dataset/data_opennavmap/traj_eval_data/test_eval_data}
DATA_DIR=${DATA_DIR:-}
EVAL_CONFIG=${EVAL_CONFIG:-OpenNavMap_map_merge.yaml}
PYTHON_OPENNAVMAP=${PYTHON_OPENNAVMAP:-/root/miniconda3/envs/opennavmap/bin/python}
EVAL_PYTHON=${EVAL_PYTHON:-/root/miniconda3/envs/traj_evaluation/bin/python}

export LD_PRELOAD="${LD_PRELOAD:-/root/miniconda3/envs/opennavmap/lib/libstdc++.so.6}"
export PYTHONPATH="${PROJECT_PATH}/python:${PROJECT_PATH}/third_party/litevloc_code/python:${PROJECT_PATH}/third_party/pose_estimation_models"
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

if [[ $# -lt 4 ]]; then
    echo "Usage: bash run_map_merging.sh <SCENE> <ORDER> <METHOD> <POSE_EST> [IQA] [IG] [TD] [MAX_SUBMAPS]" >&2
    exit 1
fi

SCENE=$1
ORDER=$2
METHOD=$3
POSE_EST=$4
USE_IQA=${5:-1}
USE_IG=${6:-1}
USE_TD=${7:-1}
MAX_SUBMAPS=${8:-}

ORDER_TAGS=("in" "r0" "r1" "r2" "r3" "r4" "r5" "r6" "r7" "r8")
ORDER_TAG="${ORDER_TAGS[$ORDER]}"

SUFFIX=""
ABLATION_FLAGS=()
if [[ "$USE_IQA" == "1" ]]; then
    SUFFIX+="iqa"
    ABLATION_FLAGS+=(--use_iqa)
fi
if [[ "$USE_IG" == "1" ]]; then
    SUFFIX+="ig"
    ABLATION_FLAGS+=(--use_ig)
fi
if [[ "$USE_TD" == "1" ]]; then
    SUFFIX+="td"
    ABLATION_FLAGS+=(--use_td)
fi
[[ -n "$SUFFIX" ]] && SUFFIX="_${SUFFIX}"

RESULT_NAME="${SCENE}_results_${ORDER_TAG}_${METHOD}${SUFFIX}"
RESULT_DIR="${OUTPUT_ROOT}/${RESULT_NAME}"
FINALMAP="${RESULT_DIR}/merge_finalmap"
DATASET_NAME="$(basename "$DATASET_ROOT")"
TUM_NAME="${DATASET_NAME}_${SCENE}_${ORDER_TAG}"
TRAJ_NAME="${METHOD}${SUFFIX}"

PIPELINE_ARGS=(
    --dataset_root "$DATASET_ROOT"
    --output_root "$OUTPUT_ROOT"
    --scene "$SCENE"
    --order_index "$ORDER"
    --method "$METHOD"
    --pose_estimation_method "$POSE_EST"
    --image_size 512 288
    --vpr_match_model vpr_dp
    --vpr_match_seq_len 10
    --viz
)
if [[ -n "$DATA_DIR" ]]; then
    PIPELINE_ARGS+=(--data_dir "$DATA_DIR")
fi
if [[ -n "$MAX_SUBMAPS" ]]; then
    PIPELINE_ARGS+=(--max_submaps "$MAX_SUBMAPS")
fi
PIPELINE_ARGS+=("${ABLATION_FLAGS[@]}")

echo "=== Step 1: Map merging ==="
"$PYTHON_OPENNAVMAP" "${PROJECT_PATH}/python/map_merge_pipeline.py" "${PIPELINE_ARGS[@]}"

echo ""
echo "=== Step 2: Convert MapFree poses to TUM ==="
CONVERT_SCRIPT="${PROJECT_PATH}/third_party/litevloc_code/python/utils/utils_convert_pose_format.py"
GT_SRC="${FINALMAP}/submap_disc_0/poses_abs_gt.txt"
EST_SRC="${FINALMAP}/submap_disc_0/poses.txt"
TS_SRC="${FINALMAP}/submap_disc_0/timestamps.txt"
GT_DST="${TRAJ_EVAL_ROOT}/groundtruth/traj/${TUM_NAME}.txt"
EST_DST="${TRAJ_EVAL_ROOT}/algorithms/${TRAJ_NAME}/laptop/traj/${TUM_NAME}.txt"

mkdir -p "$(dirname "$GT_DST")" "$(dirname "$EST_DST")"

"$PYTHON_OPENNAVMAP" "$CONVERT_SCRIPT" \
    --input_type mapfree --output_type tum \
    --input_pose "$GT_SRC" \
    --input_time "$TS_SRC" \
    --output_pose "$GT_DST"

"$PYTHON_OPENNAVMAP" "$CONVERT_SCRIPT" \
    --input_type mapfree --output_type tum \
    --input_pose "$EST_SRC" \
    --input_time "$TS_SRC" \
    --output_pose "$EST_DST"

echo "TUM GT : $GT_DST"
echo "TUM EST: $EST_DST"

echo ""
echo "=== Step 3: Trajectory evaluation ==="
TRAJ_PATH="$TRAJ_EVAL_ROOT" \
EVAL_PROJ="${PROJECT_PATH}/third_party/slam_trajectory_evaluation" \
PYTHON="$EVAL_PYTHON" \
bash "${PROJECT_PATH}/python/benchmark_map_merge/scripts/run_evaluation.sh" \
    --config "$EVAL_CONFIG" \
    --output-dir "${TRAJ_EVAL_ROOT}/report"
