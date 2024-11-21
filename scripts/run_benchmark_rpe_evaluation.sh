#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_NAME is not specified."
  echo "Usage: ./run_benchmark_evaluation.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree)"
  exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/$DATASET_NAME/map_free_eval/"
# models=("master" "duster" "hloc_disk_dilg" "vpr_cosplace_resnet18_512")
models=("master" "hloc_disk_dilg" "vpr_cosplace_resnet18_512")

for model in "${models[@]}"
do
  echo "Evaluate pose_estimation methods: $model"
  python $PROJECT_PATH/python/benchmark_rpe/evaluation.py \
    --submission_path $DATASET_PATH/results_rpe/$model/submission_15.zip \
    --dataset_path $DATASET_PATH \
    --split test \
    --log error
  echo ""
done
