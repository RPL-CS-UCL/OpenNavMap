#!/bin/bash

# Usage:
#   bash run_map_merging.sh <ORDER> <METHOD> <POSE_ESTIMATION_METHOD> [SCENE]
# Example (default order, default scene):
#   bash run_map_merging.sh 0 kf_spgo_cc_seqmatch master_calib_pretrain
# Example (default order, specify scene):
#   bash run_map_merging.sh 0 kf_spgo_cc_seqmatch master_calib_pretrain s00001_concourse
# Example (specific order):
#   bash run_map_merging.sh 1 kf_spgo_cc_seqmatch master_calib_pretrain
#
# Ablation Study Usage:
#   Control which factors are used in keyframe culling by setting environment variables.
#   Set variable to empty string to disable that factor. By default, all factors are enabled.
# Example (disable IQA in forward pass):
#   USE_IQA_FORWARD="" bash run_map_merging.sh 0 kf_spgo_cc_seqmatch master_calib_pretrain
# Example (only use information gain in forward pass):
#   USE_IQA_FORWARD="" USE_IQA_BACKWARD="" USE_IG_BACKWARD="" USE_TD="" bash run_map_merging.sh 0 kf_spgo_cc_seqmatch master_calib_pretrain
# Example (no culling factors, only use topology):
#   USE_IQA_FORWARD="" USE_IQA_BACKWARD="" USE_IG_FORWARD="" USE_IG_BACKWARD="" USE_TD="" bash run_map_merging.sh 0 kf_spgo_cc_seqmatch master_calib_pretrain
#   
# Available ablation flags:
#   USE_IQA_FORWARD  : Image quality assessment in forward pass
#   USE_IQA_BACKWARD : Image quality assessment in backward pass
#   USE_IG_FORWARD   : Information gain in forward pass
#   USE_IG_BACKWARD  : Information gain in backward pass
#   USE_TD           : Temporal difference

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
readonly END_SUBMAP_ID=15
readonly DATASET_NAME="ucl_campus_aria"
readonly PATH_SUBMAP="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/${DATASET_NAME}"
if [ -z "${4:-}" ]; then
    readonly SCENE="s00000_atrium"
else
    readonly SCENE="$4"
fi

readonly METHOD="$2" # default: kf_spgo_cc_seqmatch
readonly POSE_ESTIMATION_METHOD="$3" # master_nocalib_pretrain, master_calib_pretrain
readonly DATA_TYPES=("in" "r0" "r1" "r2" "r3" "r4" "r5" "r6" "r7" "r8")

# Ablation study flags (default: all enabled)
# Set to empty string to disable a factor
readonly USE_IQA_FORWARD="${USE_IQA_FORWARD:---use_iqa_forward}"
readonly USE_IQA_BACKWARD="${USE_IQA_BACKWARD:---use_iqa_backward}"
readonly USE_IG_FORWARD="${USE_IG_FORWARD:---use_ig_forward}"
readonly USE_IG_BACKWARD="${USE_IG_BACKWARD:---use_ig_backward}"
readonly USE_TD="${USE_TD:---use_td}"

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
    
    (( order_index <= total_orders )) || {
        echo "Error: Order index $order_index out of range (total orders: $total_orders)"
        exit 1
    }
    
    mapfile -t SCENES < <(sed -n "$((order_index + 1))p" "$SCENE_ORDER_FILE" | tr ' ' '\n')
    echo "Loaded order [$order_index] with ${#SCENES[@]} scenes"

    DATA_TYPE="${DATA_TYPES[order_index]}"
    
    # Build ablation study suffix based on enabled factors
    local ablation_suffix=""
    [[ -n "$USE_IQA_FORWARD" ]] && ablation_suffix+="iqaf"
    [[ -n "$USE_IQA_BACKWARD" ]] && ablation_suffix+="iqab"
    [[ -n "$USE_IG_FORWARD" ]] && ablation_suffix+="gf"
    [[ -n "$USE_IG_BACKWARD" ]] && ablation_suffix+="gb"
    [[ -n "$USE_TD" ]] && ablation_suffix+="t"
    [[ -n "$ablation_suffix" ]] && ablation_suffix="_${ablation_suffix}"
    
    RESULT_NAME="${SCENE}_results_${DATA_TYPE}_${METHOD}${ablation_suffix}"
    TRAJ_NAME="${METHOD}${ablation_suffix}"
    echo "Save results to $RESULT_NAME"
    
    # Print ablation study configuration
    echo "=== Ablation Study Configuration ==="
    echo "IQA Forward:     $([ -n "$USE_IQA_FORWARD" ] && echo 'ENABLED' || echo 'DISABLED')"
    echo "IQA Backward:    $([ -n "$USE_IQA_BACKWARD" ] && echo 'ENABLED' || echo 'DISABLED')"
    echo "IG Forward:      $([ -n "$USE_IG_FORWARD" ] && echo 'ENABLED' || echo 'DISABLED')"
    echo "IG Backward:     $([ -n "$USE_IG_BACKWARD" ] && echo 'ENABLED' || echo 'DISABLED')"
    echo "Temporal Diff:   $([ -n "$USE_TD" ] && echo 'ENABLED' || echo 'DISABLED')"
    echo "===================================="
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
            --pose_estimation_method "$POSE_ESTIMATION_METHOD" \
            --cull_keyframe_forward --cull_keyframe_backward \
            $USE_IQA_FORWARD $USE_IQA_BACKWARD $USE_IG_FORWARD $USE_IG_BACKWARD $USE_TD \
            --viz

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
