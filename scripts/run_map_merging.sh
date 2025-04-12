#!/bin/bash

###### Usage: Default order (0)
# ./run_map_merging.sh
###### Usage: Specific order (e.g., 1)
# ./run_map_merging.sh 1

set -euo pipefail  # Fail on errors and undefined variables

# --------------------------
# Configuration Section
# --------------------------
readonly PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
readonly PATH_SUBMAP="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_aria"
readonly IMAGE_SIZE="512 288"
readonly VPR_MATCH_MODEL="single_match"
readonly POSE_ESTIMATION_METHOD="master_calib_pretrain"
readonly SCENE_ORDER_FILE="${PATH_SUBMAP}/s00000_orders.txt"
readonly RESULT_NAMES=(
  "results_in_kf_spgo"
  "results_r0_kf_spgo"
  "results_r1_kf_spgo"
  "results_r2_kf_spgo"
  "results_r3_kf_spgo"
  "results_r4_kf_spgo"
  "results_r5_kf_spgo"
  "results_r6_kf_spgo"
  "results_r7_kf_spgo"
  "results_r8_kf_spgo"
)

# Set your desired processing range (0-based indices)
readonly START_SUBMAP_ID=16
readonly END_SUBMAP_ID=17

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

    RESULT_NAME="${RESULT_NAMES[order_index]}"
    echo "Save results to $RESULT_NAME"
}

merge_submaps() {
    local input_dir="${PATH_SUBMAP}/${RESULT_NAME}"
    local submap_dir="${PATH_SUBMAP}/s00000"
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
        
        echo "Merging: ${base_name} + ${scene} => ${new_merged_name}"
        
        python "${PROJECT_PATH}/python/map_merge_pipeline.py" \
            --input_submap_path "${input_dir}/${base_name}" "${submap_dir}/${scene}" \
            --output_map_path "${input_dir}/${new_merged_name}" \
            --image_size $IMAGE_SIZE \
            --vpr_match_model "$VPR_MATCH_MODEL" \
            --pose_estimation_method "$POSE_ESTIMATION_METHOD" \
            --viz --select_keyframe

        base_name="${new_merged_name}"
    done
    if [[ -L "${input_dir}/merge_finalmap" ]]; then
      rm "${input_dir}/merge_finalmap"
    fi
    ln -s "${input_dir}/${base_name}" "${input_dir}/merge_finalmap"
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

##################### Not used
# python $PROJECT_PATH/python/pose3slam_g2o.py \
#   --input "$PathSubmap/$Out_Map/preds/refine_pose_graph.g2o" \
#   --viz

# python /Titan/code/robohike_ws/src/pycpptools/pycpptools/src/python/utils_file/tools_convert_pose_format.py \
#   --input_pose_file "$PathSubmap/$Out_Map/preds/refine_pose_graph.g2o" \
#   --input_time_file "$PathSubmap/$Out_Map/timestamps.txt" \
#   --output_pose_file "$PathSubmap/$Out_Map/poses.txt" \
#   --input_pose_type g2o \
#   --output_pose_type mapfree
