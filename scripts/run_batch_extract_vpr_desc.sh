#!/bin/bash

# Define path and number of submaps
PATH_SUBMAP="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000"
NUM_SUBMAP=55

# Loop over each submap index
for ((i=0; i<$NUM_SUBMAP; i++ ))
do
    echo "Processing submap index: $i"

    # Run descriptor extraction
    rosrun litevloc extract_vpr_descriptors.py \
      --dataset_path "$PATH_SUBMAP/out_map$i" \
      --method cosplace \
      --backbone ResNet18 \
      --descriptors_dimension 256 \
      --num_preds_to_save 3 \
      --image_size 512 288 \
      --device cuda \
      --save_descriptors

    # Copy the resulting file
    cp \
      "$PATH_SUBMAP/out_map$i/output_extract_vpr_descriptors/outputs_cosplace/latest/preds/database_descriptors.txt" \
      "$PATH_SUBMAP/out_map$i"
done
