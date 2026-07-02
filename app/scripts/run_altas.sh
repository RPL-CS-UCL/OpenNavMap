#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: DATASET_NAME is not specified."
    echo "Usage: ./run_altas.sh <DATASET_NAME> (opennavmap_hkust, opennavmap_ucl_campus)"
    exit 1
fi

DATASET_NAME=$1

echo "Full pipeline with SuperGlue reranking and Mast3r pose estimation"
for img_path in ../assets/${DATASET_NAME}/*.jpg; do
    echo "Processing $img_path"
    python ../litevloc_altas.py \
        --database_folder /Rocket_ssd/dataset/data_vpr/${DATASET_NAME}/images/test/database \
        --img_files "$img_path" \
        --recall_k 10 \
        --image_size 224 224 \
        --batch_size 64 \
        --num_workers 8 \
        --database_descriptors_path /Rocket_ssd/dataset/data_vpr/${DATASET_NAME}/descriptors/test/megaloc_database_descriptors.npy \
        --device cuda \
        --matcher loftr \
        --pose_estimator mast3r \
        --output_file ../logs/results_full_pipeline.txt \
        --viz
done

# for img_path in /Rocket_ssd/dataset/data_vpr/${DATASET_NAME}/images/test/queries/*.jpg; do
#     echo "Processing $img_path"
#     python ../litevloc_altas.py \
#         --database_folder /Rocket_ssd/dataset/data_vpr/${DATASET_NAME}/images/test/database \
#         --img_files "$img_path" \
#         --recall_k 10 \
#         --image_size 224 224 \
#         --batch_size 64 \
#         --num_workers 8 \
#         --database_descriptors_path /Rocket_ssd/dataset/data_vpr/${DATASET_NAME}/descriptors/test/megaloc_database_descriptors.npy \
#         --device cuda \
#         --matcher loftr \
#         --pose_estimator mast3r \
#         --output_file ../logs/results_full_pipeline_${DATASET_NAME}_queries.txt
# done
