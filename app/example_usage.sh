#!/bin/bash

# Example 1: Single image query with reranking and local localization
echo "Example 1: Full pipeline with SuperGlue reranking and Mast3r pose estimation"
dataset_name="opennavmap_hkust"

for img_path in assets/${dataset_name}/*.jpg; do
    echo "Processing $img_path"
    python litevloc_altas.py \
        --database_folder /Rocket_ssd/dataset/data_vpr/${dataset_name}/images/test/database \
        --img_files "$img_path" \
        --recall_k 10 \
        --image_size 224 224 \
        --batch_size 64 \
        --num_workers 8 \
        --database_descriptors_path /Rocket_ssd/dataset/data_vpr/${dataset_name}/descriptors/test/megaloc_database_descriptors.npy \
        --device cuda \
        --matcher loftr \
        --pose_estimator mast3r_calib_pretrain \
        --output_file logs/results_full_pipeline.txt
done
