#!/bin/bash

# Example 1: Single image query with reranking and local localization
echo "Example 1: Full pipeline with SuperGlue reranking and Mast3r pose estimation"
python litevloc_altas.py \
    --database_folder /Rocket_ssd/dataset/data_vpr/opennavmap_hkust/images/test/database \
    --img_files "assets/@0218047.08@2472431.44@50@Q@022.33467@0114.26272@@@-46@003@-74@014@@@hkust_20241226_1947(462).jpg" \
    --recall_k 10 \
    --image_size 224 224 \
    --batch_size 64 \
    --num_workers 8 \
    --database_descriptors_path /Rocket_ssd/dataset/data_vpr/opennavmap_hkust/descriptors/test/megaloc_database_descriptors.npy \
    --device cuda \
    --matcher loftr \
    --pose_estimator mast3r \
    --output_file logs/results_full_pipeline.txt

# Example 2: Global localization only
# echo "Example 2: Global VPR only"
# python litevloc_altas.py \
#     --database_folder /Rocket_ssd/dataset/data_vpr/opennavmap_hkust/images/test/database \
#     --img_files "assets/@0218047.08@2472431.44@50@Q@022.33467@0114.26272@@@-46@003@-74@014@@@hkust_20241226_1947(462).jpg" \
#     --recall_k 10 \
#     --image_size 224 224 \
#     --batch_size 64 \
#     --num_workers 8 \
#     --database_descriptors_path /Rocket_ssd/dataset/data_vpr/opennavmap_hkust/descriptors/test/megaloc_database_descriptors.npy \
#     --device cuda \
#     --output_file logs/results_global_only.txt

