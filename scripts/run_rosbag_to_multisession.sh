#!/bin/bash
# Usage:
#   bash scripts/run_rosbag_to_multisession.sh <config.yaml> [--skip-convert] [--skip-merge]
#
# Rosbag -> N session submaps -> VPR descriptors + IQA + covis/trav edges -> map merging -> ATE.
# Merge settings mirror the ucl_campus_aria reference run (see CLAUDE.md); only IMAGE_SIZE
# differs because this sensor is not 16:9.
#
# Environment overrides:
#   IMAGE_SIZE ("W H", default "512 416"), METHOD (default spgo_cc_seqmatch_master_gncgm),
#   TRAJ_EVAL_ROOT, EVAL_CONFIG, PYTHON_OPENNAVMAP
set -euo pipefail

PROJECT_PATH="/Titan/code/robohike_ws/src/opennavmap"
CONFIG="${1:?usage: $0 <config.yaml> [--skip-convert] [--skip-merge]}"; shift
SKIP_CONVERT=0; SKIP_MERGE=0
for arg in "$@"; do
    case "$arg" in
        --skip-convert) SKIP_CONVERT=1 ;;
        --skip-merge) SKIP_MERGE=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

PYTHON_OPENNAVMAP=${PYTHON_OPENNAVMAP:-/root/miniconda3/envs/opennavmap/bin/python}
IMAGE_SIZE=${IMAGE_SIZE:-512 416}
METHOD=${METHOD:-spgo_cc_seqmatch_master_gncgm}
# litevloc's utils/ is added as well because gen_covis_trav_edges.py imports
# `utils_image_matching_method` as a top-level module
export PYTHONPATH="${PROJECT_PATH}/python:${PROJECT_PATH}/third_party/litevloc_code/python:${PROJECT_PATH}/third_party/litevloc_code/python/utils:${PROJECT_PATH}/third_party/vismatch"

# Read output_root / scene / data dir name from the YAML through the config class
read -r OUTPUT_ROOT SCENE DATA_DIR NUM_SESSIONS < <("$PYTHON_OPENNAVMAP" - "$CONFIG" <<'PYEOF'
import sys
from rosbag_convert.config import BagConversionConfig
c = BagConversionConfig.from_yaml(sys.argv[1])
print(c.output_root, c.scene, c.data_dir_name, c.num_sessions)
PYEOF
)
TRAJ_EVAL_ROOT=${TRAJ_EVAL_ROOT:-/Titan/dataset/data_opennavmap/traj_eval_data/$(basename "$OUTPUT_ROOT")_eval_data}
EVAL_CONFIG=${EVAL_CONFIG:-OpenNavMap_$(basename "$OUTPUT_ROOT").yaml}
SUBMAP_IDS=$(seq 0 $((NUM_SESSIONS - 1)))

if [[ "$SKIP_CONVERT" -eq 0 ]]; then
    echo "=== Step A: rosbag -> ${OUTPUT_ROOT}/${DATA_DIR} ==="
    "$PYTHON_OPENNAVMAP" "${PROJECT_PATH}/python/rosbag_convert/convert_rosbag_to_multisession.py" \
        --config "$CONFIG" --overwrite

    echo "=== Step B: VPR descriptors + IQA per submap ==="
    for id in $SUBMAP_IDS; do
        SUBMAP="${OUTPUT_ROOT}/${DATA_DIR}/${id}"
        "$PYTHON_OPENNAVMAP" "${PROJECT_PATH}/third_party/litevloc_code/python/utils/extract_vpr_descriptors.py" \
            --map_path "$SUBMAP" --method cosplace --backbone ResNet18 --descriptors_dimension 256 \
            --image_size ${IMAGE_SIZE} --device cuda --save_descriptors
        "$PYTHON_OPENNAVMAP" "${PROJECT_PATH}/third_party/litevloc_code/python/utils/extract_iqa.py" \
            --map_path "$SUBMAP" --metric musiq --device cuda --output "$SUBMAP"
        test -s "${SUBMAP}/database_descriptors.txt" && test -s "${SUBMAP}/iqa_data.txt"
    done

    echo "=== Step C: covis / trav edge enrichment ==="
    "$PYTHON_OPENNAVMAP" "${PROJECT_PATH}/python/gen_covis_trav_edges.py" \
        --dataset_dir "${OUTPUT_ROOT}/${DATA_DIR}" --scenes $SUBMAP_IDS --matcher sift-nn --image_size ${IMAGE_SIZE}
fi

if [[ "$SKIP_MERGE" -eq 0 ]]; then
    echo "=== Step D: multi-session map merging (in-order) + evaluation ==="
    DATASET_ROOT="$OUTPUT_ROOT" DATA_DIR="$DATA_DIR" IMAGE_SIZE="$IMAGE_SIZE" \
    TRAJ_EVAL_ROOT="$TRAJ_EVAL_ROOT" EVAL_CONFIG="$EVAL_CONFIG" \
        bash "${PROJECT_PATH}/scripts/run_map_merging.sh" "$SCENE" 0 "$METHOD" master 1 1 1

    FINALMAP="${OUTPUT_ROOT}/${SCENE}_results_in_${METHOD}_iqaigtd/merge_finalmap"
    N_DISC=$(ls -d "${FINALMAP}"/submap_disc_* | wc -l)
    N_NODES=$(cat "${FINALMAP}"/submap_disc_*/poses.txt | wc -l)
    echo "final map: ${N_NODES} nodes, ${N_DISC} connected component(s)"
    if [[ "$N_DISC" -ne 1 ]]; then
        echo "WARNING: sessions did not all connect (${N_DISC} components); ATE below covers only submap_disc_0" >&2
    fi
    echo "report: ${TRAJ_EVAL_ROOT}/report"
fi
