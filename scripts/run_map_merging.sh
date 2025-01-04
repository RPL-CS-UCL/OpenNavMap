#!/bin/bash

# Define path and number of submaps
PATH_SUBMAP="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000"
# NUM_SUBMAP=6

IN_MAP=(
  "out_map_0_1_2_3"
  "out_map4"
)
OUT_MAP="out_map_0_1_2_3_4"

# python ../../pycpptools/pycpptools/src/python/utils_dataset/map_multisession/gendataset_from_files.py \
#   --in_dir /Titan/dataset/data_litevloc/matterport3d/vloc_17DRP5sb8fy/out_general \
#   --out_dir /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy/ \
#   --num_split 2 \
#   --scene_id 0 \
#   --start_indice 0 \
#   --step 30

# rosrun litevloc extract_vpr_descriptors.py \
#   --dataset_path /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map7 \
#   --method cosplace --backbone ResNet18 --descriptors_dimension 256 \
#   --num_preds_to_save 3 \
#   --image_size 512 288 \
#   --device cuda \
#   --save_descriptors

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

# Pose graph optimization
# python pose3slam_g2o.py \
#   --input /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy/out_map_0_1/preds/initial_pose_graph.g2o \
#   --output /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy/out_map_0_1/preds/refine_pose_graph.g2o \
#   --plot

# Convert the poses from g2o to mapfree format
# python ../../pycpptools/pycpptools/src/python/utils_file/tools_convert_pose_format.py \
#   --input /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy/out_map_0_1/preds/refine_pose_graph.g2o \
#   --output /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy/out_map_0_1/poses.txt \
#   --input_pose_type g2o \
#   --output_pose_type mapfree