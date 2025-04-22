#!/bin/bash

###### Usage: Default order (0)
# ./run_map_merging.sh 0 kf_spgo_seqmatch
###### Usage: Specific order (e.g., 1)
# ./run_map_merging.sh 1 kf_spgo_seqmatch

set -euo pipefail  # Fail on errors and undefined variables

# --------------------------
# Configuration Section
# --------------------------
# Set your desired processing range (0-based indices)
# ucl_campus/s00000_data
#   aria: 0-54 (300m length)
#   google_street_view: 55-56 (300m-1000m length)
# hkust/s00000
#   aria: 0-7 (300m length)
#   fusionportable: 8-9 (1000m length)
#   smartphone: 10-12 (<300m length)
# vineyard/
#   aria: 0-4 (300m length)

# TODO(gogojjh): Users should change these parameters
readonly START_SUBMAP_ID=0
readonly END_SUBMAP_ID=54
readonly DATASET_NAME="ucl_campus"
readonly PATH_SUBMAP="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/${DATASET_NAME}"
readonly SCENE="s00000"

# readonly METHOD="kf_forward_spgo_seqmatch"
# readonly METHOD="kf_spgo_seqmatch"
# readonly METHOD="nokf_spgo_seqmatch"
# readonly METHOD="kf_spgo_singlematch"
# readonly METHOD="nokf_spgo_singlematch"
readonly METHOD="$2" # default: kf_spgo_seqmatch
readonly DATA_TYPES=("in" "r0" "r1" "r2" "r3" "r4" "r5" "r6" "r7" "r8")

########################
readonly PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
readonly IMAGE_SIZE="512 288"
readonly VPR_MATCH_MODEL="sequence_match_adaptive" # single_match, sequence_match_adaptive
readonly VPR_SEQ_LEN=10
readonly POSE_ESTIMATION_METHOD="master_calib_pretrain"
readonly SCENE_ORDER_FILE="${PATH_SUBMAP}/${SCENE}_orders.txt"
readonly TRAJ_EVAL_PATH="/Rocket_ssd/dataset/data_litevloc/traj_eval_data/map_merge_eval_data"

# --------------------------
# Initialization and Validation
# --------------------------
validate_dependencies() {
    command -v python >/dev/null 2>&1 || { echo >&2 "Python required but not found. Aborting."; exit 1; }
}

validate_paths() {
    [[ -d "$PROJECT_PATH" ]] || { echo "Project path not found: $PROJECT_PATH"; exit 1; }
    [[ -d "$PATH_SUBMAP" ]] || { echo "Submap path not found: $PATH_SUBMAP"; exit 1; }
    [[ -f "$SCENE_ORDER_FILE" ]] || { echo "Scene order file missing: $SCENE_ORDER_FILE"; exit 1; }
}

# --------------------------
# Main Logic
# --------------------------
load_scene_order() {
    local -i order_index=${1:-0}
    local -i total_orders=$(wc -l < "$SCENE_ORDER_FILE")  
    
    (( order_index < total_orders )) || {
        echo "Error: Order index $order_index out of range (total orders: $total_orders)"
        exit 1
    }
    
    mapfile -t SCENES < <(sed -n "$((order_index + 1))p" "$SCENE_ORDER_FILE" | tr ' ' '\n')
    echo "Loaded order [$order_index] with ${#SCENES[@]} scenes"

    DATA_TYPE="${DATA_TYPES[order_index]}"
    RESULT_NAME="${SCENE}_results_${DATA_TYPE}_${METHOD}"
    TRAJ_NAME="${METHOD}"
    echo "Save results to $RESULT_NAME"
}

merge_submaps() {
    local input_dir="${PATH_SUBMAP}/${RESULT_NAME}"
    local submap_dir="${PATH_SUBMAP}/${SCENE}_data"
    mkdir -p $input_dir

    # Validate processing range
    if (( START_SUBMAP_ID >= ${#SCENES[@]} || END_SUBMAP_ID >= ${#SCENES[@]} )); then
        echo "Error: Submap ID out of range (max index: $((${#SCENES[@]} - 1)))"
        exit 1
    fi
    if (( START_SUBMAP_ID > END_SUBMAP_ID )); then
        echo "Error: Invalid range (${START_SUBMAP_ID}-${END_SUBMAP_ID})"
        exit 1
    fi

    # Create base name from initial scenes
    local base_name="merge"
    for ((i=0; i<START_SUBMAP_ID; i++)); do
        base_name+="_${SCENES[i]}"
    done

    # Process specified range
    for ((i=START_SUBMAP_ID; i<=END_SUBMAP_ID; i++)); do
        local scene="${SCENES[i]}"
        local new_merged_name="${base_name}_${scene}"
        
        echo "ID: ${i}"
        echo "Merging: ${base_name} + ${scene} => ${new_merged_name}"
        
        python "${PROJECT_PATH}/python/map_merge_pipeline.py" \
            --input_submap_path "${input_dir}/${base_name}" "${submap_dir}/${scene}" \
            --output_map_path "${input_dir}/${new_merged_name}" \
            --image_size $IMAGE_SIZE \
            --vpr_match_model "$VPR_MATCH_MODEL" \
            --vpr_match_seq_len "$VPR_SEQ_LEN" \
            --pose_estimation_method "$POSE_ESTIMATION_METHOD" \
            --viz --prune_keyframe_forward --prune_keyframe_backward # --color_correct

        base_name="${new_merged_name}"
    done
    if [[ -L "${input_dir}/merge_finalmap" ]]; then
      rm "${input_dir}/merge_finalmap"
    fi
    ln -s "${input_dir}/${base_name}" "${input_dir}/merge_finalmap"

    # GT and EST poses
    rosrun litevloc utils_convert_pose_format.py \
        --input_type mapfree --output_type tum \
        --input_pose "${input_dir}/merge_finalmap/submap_disc_0/poses_abs_gt.txt" \
        --input_time "${input_dir}/merge_finalmap/submap_disc_0/timestamps.txt" \
        --output_pose "${TRAJ_EVAL_PATH}/groundtruth/traj/${DATASET_NAME}_${SCENE}_${DATA_TYPE}.txt"

    rosrun litevloc utils_convert_pose_format.py \
        --input_type mapfree --output_type tum \
        --input_pose "${input_dir}/merge_finalmap/submap_disc_0/poses.txt" \
        --input_time "${input_dir}/merge_finalmap/submap_disc_0/timestamps.txt" \
        --output_pose "${TRAJ_EVAL_PATH}/algorithms/${TRAJ_NAME}/laptop/traj/${DATASET_NAME}_${SCENE}_${DATA_TYPE}.txt"
    
    echo "Converted pose format to TUM format."
    echo "From: ${input_dir}/merge_finalmap/poses.txt"
    echo "To  : ${TRAJ_EVAL_PATH}/algorithms/${TRAJ_NAME}/laptop/traj/${DATASET_NAME}_${SCENE}_${DATA_TYPE}.txt"
}

# --------------------------
# Execution Flow
# --------------------------
main() {
    validate_dependencies
    validate_paths
    load_scene_order "${1:-0}"
    merge_submaps
    echo "Successfully processed scenes ${START_SUBMAP_ID}-${END_SUBMAP_ID} from order [${1:-0}]"
}

main "$@"
