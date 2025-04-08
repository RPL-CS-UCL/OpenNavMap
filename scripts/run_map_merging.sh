#!/bin/bash

# Define path and number of submaps
PathSubmap="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_aria/s00000"
# NUM_SUBMAP=6

In_Map=(
  "out_map0"
  "out_map1"
)
Out_Map="out_map_test"

# Merge the maps
rosrun litevloc map_merge_pipeline.py \
  --input_submap_path "$PathSubmap/${In_Map[0]}" "$PathSubmap/${In_Map[1]}" \
  --output_map_path "$PathSubmap/$Out_Map" \
  --image_size 512 288 \
  --vpr_match_model single_match \
  --pose_estimation_method master_calib_pretrain \
  --viz
  # --select_keyframe

# NOTE(gogojjh): Not used - Compute IQA
# rosrun litevloc extract_iqa.py \
#   --dataset_path /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_aria/s00000/out_map3 \
#   --output /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_aria/s00000/out_map3

# NOTE(gogojjh): Not used
# rosrun litevloc pose3slam_g2o.py \
#   --input "$PathSubmap/$Out_Map/preds/refine_pose_graph.g2o" \
#   --viz

# python /Titan/code/robohike_ws/src/pycpptools/pycpptools/src/python/utils_file/tools_convert_pose_format.py \
#   --input_pose_file "$PathSubmap/$Out_Map/preds/refine_pose_graph.g2o" \
#   --input_time_file "$PathSubmap/$Out_Map/timestamps.txt" \
#   --output_pose_file "$PathSubmap/$Out_Map/poses.txt" \
#   --input_pose_type g2o \
#   --output_pose_type mapfree
