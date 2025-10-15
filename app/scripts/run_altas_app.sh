#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: DATASET_NAME is not specified."
    echo "Usage: ./run_altas_app.sh <DATASET_NAME> (opennavmap_hkust, opennavmap_ucl_campus)"
    exit 1
fi

DATASET_NAME=$1

echo "Launching the LiteVLoc Atlas App with Gradio UI"
python ../litevloc_altas_app.py \
    --dataset_name ${DATASET_NAME} \
    --database_folder /Rocket_ssd/dataset/data_vpr/${DATASET_NAME}/images/test/database \
    --database_descriptors_path /Rocket_ssd/dataset/data_vpr/${DATASET_NAME}/descriptors/test/megaloc_database_descriptors.npy \
    --image_size 224 224 \
    --device cuda \
    --recall_k 20 \
    --matcher loftr \
    --pose_estimator mast3r_calib_pretrain \
    --share

