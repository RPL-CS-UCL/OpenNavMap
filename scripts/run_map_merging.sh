#!/bin/bash

# Usage:
#   bash run_map_merging.sh <SCENE> <ORDER> <METHOD> <POSE_EST> [IQA] [IG] [TD] [MAX_SUBMAPS]
#   IQA/IG/TD: 1=enabled, 0=disabled, default=1
#   MAX_SUBMAPS: default=all
#
# Environment overrides:
#   DATASET_ROOT, OUTPUT_ROOT, DATA_DIR, TRAJ_EVAL_ROOT, EVAL_CONFIG
#   PGO_ROBUST, PGO_GNC_BARC_PROB, PGO_PERSISTENT_LOOPS
#   PGO_LOOP_SIGMA_TRANS, PGO_LOOP_SIGMA_ROT, PGO_LOOP_CONF_SCALING, PGO_MIN_LOOP_EDGES
#   RERUN_VIZ=1 to enable Rerun visualization recording
#   RERUN_OUTPUT, RERUN_IMAGE_FORMAT, RERUN_JPEG_QUALITY,
#   RERUN_DMATRIX_FORMAT, RERUN_AXIS_SCALE, RERUN_VIZ_DIR

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

# Robust pose graph optimization back-end (none | huber | gnc_tls | gnc_gm)
PGO_ROBUST=${PGO_ROBUST:-gnc_tls}
PGO_GNC_BARC_PROB=${PGO_GNC_BARC_PROB:-0.99}
# Keep accepted loop edges as loop factors across merge steps (0 | 1)
PGO_PERSISTENT_LOOPS=${PGO_PERSISTENT_LOOPS:-0}
# Loop factor noise, which sets the GNC outlier threshold
PGO_LOOP_SIGMA_TRANS=${PGO_LOOP_SIGMA_TRANS:-0.1}
PGO_LOOP_SIGMA_ROT=${PGO_LOOP_SIGMA_ROT:-1.0}
PGO_LOOP_CONF_SCALING=${PGO_LOOP_CONF_SCALING:-inverse}
# Below this many refined loop edges the merge is deferred (1 = disabled)
PGO_MIN_LOOP_EDGES=${PGO_MIN_LOOP_EDGES:-1}

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
    --pgo_robust "$PGO_ROBUST"
    --pgo_gnc_barc_prob "$PGO_GNC_BARC_PROB"
    --pgo_loop_sigma_trans "$PGO_LOOP_SIGMA_TRANS"
    --pgo_loop_sigma_rot "$PGO_LOOP_SIGMA_ROT"
    --pgo_loop_conf_scaling "$PGO_LOOP_CONF_SCALING"
    --pgo_min_loop_edges "$PGO_MIN_LOOP_EDGES"
    --viz
)
if [[ -n "$DATA_DIR" ]]; then
    PIPELINE_ARGS+=(--data_dir "$DATA_DIR")
fi
if [[ -n "$MAX_SUBMAPS" ]]; then
    PIPELINE_ARGS+=(--max_submaps "$MAX_SUBMAPS")
fi
if [[ "$PGO_PERSISTENT_LOOPS" == "1" ]]; then
    PIPELINE_ARGS+=(--pgo_persistent_loops)
fi
PIPELINE_ARGS+=("${ABLATION_FLAGS[@]}")

# Rerun visualization flags (optional, set RERUN_VIZ=1 to enable)
if [[ "${RERUN_VIZ:-0}" == "1" ]]; then
    RERUN_OUTPUT_PATH="${RERUN_OUTPUT:-${RESULT_DIR}/map_merge_process.rrd}"
    PIPELINE_ARGS+=(
        --rerun-viz
        --rerun-output "$RERUN_OUTPUT_PATH"
        --rerun-image-format "${RERUN_IMAGE_FORMAT:-jpg}"
        --rerun-jpeg-quality "${RERUN_JPEG_QUALITY:-85}"
        --rerun-dmatrix-format "${RERUN_DMATRIX_FORMAT:-png}"
        --rerun-axis-scale "${RERUN_AXIS_SCALE:-auto}"
    )
    if [[ -n "${RERUN_VIZ_DIR:-}" ]]; then
        PIPELINE_ARGS+=(--rerun-viz-dir "$RERUN_VIZ_DIR")
    fi
fi

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
