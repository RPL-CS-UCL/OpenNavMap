#!/bin/bash

# Usage:
#   bash run_map_merging.sh <ORDER> <METHOD> <POSE_EST_METHOD> [SCENE] [IQA] [IG] [TD]
#   IQA/IG/TD: 1=enabled, 0=disabled, default=1
# Examples:
#   bash run_map_merging.sh s00001 0 spgo_cc_seqmatch master_calib_pretrain 1 1 1 # All factors for node culling enabled
#   bash run_map_merging.sh s00001 0 spgo_cc_seqmatch master_calib_pretrain 0 1 0 # Only IG enabled
#   bash run_map_merging.sh s00001 0 spgo_cc_seqmatch master_calib_pretrain 0 0 0 # No culling factors

set -euo pipefail  # Fail on errors and undefined variables

# --------------------------
# Configuration Section
# --------------------------
# Set your desired processing range (0-based indices)
# ucl_campus_aria/s00000_aria_data
#   aria: 0-54 (300m length)
#   google_street_view: 55-56 (300m-1000m length)
# hkust_campus/s00000_aria_data
#   aria: 0-7 (300m length)
#   fusionportable: 8-9 (1000m length)
#   smartphone: 10-12 (<300m length)
# vineyard/s00000_aria_data
#   aria: 0-4 (300m length)
# 360loc/s00000_atrium_data
#   aria: 0
#   device1: 1
#   device2: 2
#   device3: 3
#   device4: 4

# Note: Users should change these parameters according to the dataset and scene
readonly START_SUBMAP_ID=0
readonly END_SUBMAP_ID=54
readonly DATASET_NAME="ucl_campus_aria"
readonly PATH_SUBMAP="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/${DATASET_NAME}"

readonly SCENE=$1
readonly ORDER_INDEX=$2
readonly METHOD=$3 # default: spgo_cc_seqmatch
readonly POSE_EST=$4 # master_nocalib_pretrain, master_calib_pretrain
readonly DATA_TYPES=("in" "r0" "r1" "r2" "r3" "r4" "r5" "r6" "r7" "r8")

# Ablation study flags: 1=enabled, 0=disabled, default=1 (all enabled)
readonly USE_IQA=${5:-1}
readonly USE_IG=${6:-1}
readonly USE_TD=${7:-1}

########################
readonly PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
readonly IMAGE_SIZE="512 288"
readonly VPR_MATCH_MODEL="graph_search" # single_match, sequence_match, sequence_match_adaptive, graph_search
readonly VPR_SEQ_LEN=10
readonly SCENE_ORDER_FILE="${PATH_SUBMAP}/${SCENE}_orders.txt"
readonly TRAJ_EVAL_PATH="/Rocket_ssd/dataset/data_litevloc/traj_eval_data/test_eval_data"

# --------------------------
# Initialization and Validation
# --------------------------
validate_dependencies_and_paths() {
    command -v python >/dev/null 2>&1 || { echo >&2 "Python required but not found. Aborting."; exit 1; }
    [[ -d "$PROJECT_PATH" ]] || { echo "Project path not found: $PROJECT_PATH"; exit 1; }
    [[ -d "$PATH_SUBMAP" ]] || { echo "Submap path not found: $PATH_SUBMAP"; exit 1; }
    [[ -f "$SCENE_ORDER_FILE" ]] || { echo "Scene order file missing: $SCENE_ORDER_FILE"; exit 1; }
}

# --------------------------
# Main Logic
# --------------------------
load_scene_order() {
    # Load scene order from file
    local -i order_index=$ORDER_INDEX
    local -i total_orders=$(wc -l < "$SCENE_ORDER_FILE")  
    
    (( order_index <= total_orders )) || {
        echo "Error: Order index $order_index out of range (total orders: $total_orders)"
        exit 1
    }
  
    mapfile -t SCENES < <(sed -n "$((order_index + 1))p" "$SCENE_ORDER_FILE" | tr ' ' '\n')
    echo "Loaded order [$order_index] with ${#SCENES[@]} scenes"
    DATA_TYPE="${DATA_TYPES[order_index]}"
    
    # Build ablation suffix and flags in one pass
    local suffix="" flags=""
    [ "$USE_IQA" == "1" ] && { suffix+="iqa"; flags+=" --use_iqa"; }
    [ "$USE_IG" == "1" ] && { suffix+="ig"; flags+=" --use_ig"; }
    [ "$USE_TD" == "1" ] && { suffix+="td"; flags+=" --use_td"; }
    [ -n "$suffix" ] && suffix="_$suffix"
    
    RESULT_NAME="${SCENE}_results_${DATA_TYPE}_${METHOD}${suffix}"
    TRAJ_NAME="${METHOD}${suffix}"
    ABLATION_FLAG="$flags"
    
    echo "Save results to $RESULT_NAME"
    echo "=== Ablation: IQA=$USE_IQA IG=$USE_IG TD=$USE_TD ==="
}

merge_submaps() {
    local input_dir="${PATH_SUBMAP}/${RESULT_NAME}"
    local submap_dir="${PATH_SUBMAP}/${SCENE}_aria_data"
    mkdir -p $input_dir

    local output_pose_path_gt="${TRAJ_EVAL_PATH}/groundtruth/traj"
    mkdir -p $output_pose_path_gt
    local output_pose_path_alg="${TRAJ_EVAL_PATH}/algorithms/${TRAJ_NAME}/laptop/traj"
    mkdir -p $output_pose_path_alg

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
        echo "ID: ${i}"
        local scene="${SCENES[i]}"
        if [ ! -d "${submap_dir}/${scene}" ]; then
            echo "Warning: Submap directory '${submap_dir}/${scene}' does not exist. Skipping."
            break
        fi
        
        local new_merged_name="${base_name}_${scene}"       
        echo "Merging: ${base_name} + ${scene} => ${new_merged_name}"
        
        python "${PROJECT_PATH}/python/map_merge_pipeline.py" \
            --input_submap_path "${input_dir}/${base_name}" "${submap_dir}/${scene}" \
            --output_map_path "${input_dir}/${new_merged_name}" \
            --image_size $IMAGE_SIZE \
            --vpr_match_model "$VPR_MATCH_MODEL" \
            --vpr_match_seq_len "$VPR_SEQ_LEN" \
            --pose_estimation_method "$POSE_EST" \
            --viz \
            $ABLATION_FLAG

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
    validate_dependencies_and_paths
    load_scene_order
    merge_submaps
    echo "Successfully processed scenes ${START_SUBMAP_ID}-${END_SUBMAP_ID} from order [${ORDER_INDEX}]"
}

main "$@"
