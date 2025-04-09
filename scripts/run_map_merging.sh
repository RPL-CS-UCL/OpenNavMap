#!/bin/bash

# Configuration
PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
START_SUBMAP_ID=0
END_SUBMAP_ID=1
PathSubmap="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_aria"
image_size="512 288"
vpr_match_model="single_match"
pose_estimation_method="master_calib_pretrain"

# Loop through submap IDs to extract IQA
# for ((i=START_SUBMAP_ID; i<=END_SUBMAP_ID; i++)); do
#   submap_name="out_map${i}"
#   python $PROJECT_PATH/python/utils/extract_iqa.py \
#     --dataset_path "$PathSubmap/$submap_name" \
#     --output "$PathSubmap/$submap_name"
# done

# Loop through submap IDs to merge
base_name="out_map"
for ((i=START_SUBMAP_ID; i<=END_SUBMAP_ID; i++)); do
  submap_to_add="out_map${i}"
  current_merged_map="${base_name}_test"
  output_merged_map="${base_name}_${i}_test"

  python $PROJECT_PATH/python/map_merge_pipeline.py \
    --input_submap_path "$PathSubmap/s00002_test_result/$current_merged_map" "$PathSubmap/s00000/$submap_to_add" \
    --output_map_path "$PathSubmap/s00002_test_result/$output_merged_map" \
    --image_size $image_size \
    --vpr_match_model "$vpr_match_model" \
    --pose_estimation_method "$pose_estimation_method" \
    --viz --select_keyframe

  base_name="${base_name}_${i}"
done

# NOTE(gogojjh): Not used
# python $PROJECT_PATH/python/pose3slam_g2o.py \
#   --input "$PathSubmap/$Out_Map/preds/refine_pose_graph.g2o" \
#   --viz

# NOTE(gogojjh): Not used - GTSAM-PGO
# python /Titan/code/robohike_ws/src/pycpptools/pycpptools/src/python/utils_file/tools_convert_pose_format.py \
#   --input_pose_file "$PathSubmap/$Out_Map/preds/refine_pose_graph.g2o" \
#   --input_time_file "$PathSubmap/$Out_Map/timestamps.txt" \
#   --output_pose_file "$PathSubmap/$Out_Map/poses.txt" \
#   --input_pose_type g2o \
#   --output_pose_type mapfree
