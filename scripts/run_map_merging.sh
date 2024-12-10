#!/bin/bash

# Split dataset
python ../../pycpptools/pycpptools/src/python/utils_dataset/map_multisession/gendataset_from_files.py \
  --in_dir /Titan/dataset/data_litevloc/matterport3d/vloc_17DRP5sb8fy/out_general \
  --out_dir /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy/ \
  --num_split 2 \
  --scene_id 0 \
  --start_indice 0 \
  --step 30

rosrun litevloc extract_vpr_descriptors.py \
  --dataset_path /Rocket_ssd/dataset/data_litevloc/matterport3d/vloc_17DRP5sb8fy/out_map \
  --method cosplace --backbone ResNet18 --descriptors_dimension 256 \
  --num_preds_to_save 3 \
  --image_size 512 288 \
  --device cuda \
  --save_descriptors

# Merge the maps
rosrun litevloc map_merge_pipeline.py \
  --dataset_path /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy \
  --num_submap 2

# Pose graph optimization
python pose3slam_g2o.py \
  --input /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy/out_map_0_1/preds/initial_pose_graph.g2o \
  --output /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy/out_map_0_1/preds/refine_pose_graph.g2o \
  --plot

# Convert the poses from g2o to mapfree format
python ../../pycpptools/pycpptools/src/python/utils_file/tools_convert_pose_format.py \
  --input /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy/out_map_0_1/preds/refine_pose_graph.g2o \
  --output /Rocket_ssd/dataset/data_litevloc/matterport3d/map_multisession_eval/s17DRP5sb8fy/out_map_0_1/poses.txt \
  --input_pose_type g2o \
  --output_pose_type mapfree