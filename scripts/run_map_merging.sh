#!/bin/bash

# Define path and number of submaps
PATH_SUBMAP="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000"
# NUM_SUBMAP=6

IN_MAP=(
  "out_map0"
  "out_map1"
)
OUT_MAP="out_map_0_1"

# Merge the maps
rosrun litevloc map_merge_pipeline.py \
  --input_submap_path "$PATH_SUBMAP/${IN_MAP[0]}" "$PATH_SUBMAP/${IN_MAP[1]}" \
  --output_map_path "$PATH_SUBMAP/$OUT_MAP" \
  --image_size 512 288 \
  --vpr_match_model sequence_match

rosrun litevloc pose3slam_g2o.py \
  --input "$PATH_SUBMAP/$OUT_MAP/preds/initial_pose_graph.g2o" \
  -o "$PATH_SUBMAP/$OUT_MAP/preds/refine_pose_graph.g2o" \
  -p

python /Titan/code/robohike_ws/src/pycpptools/pycpptools/src/python/utils_file/tools_convert_pose_format.py \
  --input_pose_file "$PATH_SUBMAP/$OUT_MAP/preds/refine_pose_graph.g2o" \
  --input_time_file "$PATH_SUBMAP/$OUT_MAP/timestamps.txt" \
  --output_pose_file "$PATH_SUBMAP/$OUT_MAP/poses.txt" \
  --input_pose_type g2o \
  --output_pose_type mapfree
