#!/bin/bash

# Check if DATASET_PATH is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_PATH is not specified."
  echo "Usage: ./run_benchmark_vpr_submission.sh <DATASET_PATH> <DATABASE_NAME> <QUERY_NAME> <VPR_MATCH_MODEL> <IMAGE_MATCH_MODEL>"
  exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_PATH=$1
DATABASE_NAME=$2
QUERY_NAME=$3

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export DATABASE_PATH="$DATASET_PATH/database/$DATABASE_NAME"
export QUERY_PATH="$DATASET_PATH/query/$QUERY_NAME"
export OUT_DIR="$DATASET_PATH/results_vpr"

export BACKBONE="ResNet18"
export DESC_DIMENSION="256"

vpr_models=("cosplace")
vpr_match_models=("single_match" "sequence_match" "sequence_match_ransac")
image_match_models=("none" "master")

# Run the Python script
for vpr_model in "${vpr_models[@]}"
do
  for vpr_match_model in "${vpr_match_models[@]}"
  do
    for image_match_model in "${image_match_models[@]}"
    do
      setting="${vpr_model}_${vpr_match_model}_${image_match_model}"
      echo "Evaluate VPR with setting: $setting"
      python $PROJECT_PATH/python/benchmark_vpr/submission.py \
        --database_folder $DATABASE_PATH \
        --queries_folder $QUERY_PATH \
        --backbone $BACKBONE \
        --descriptors_dimension $DESC_DIMENSION \
        --vpr_model $vpr_model \
        --vpr_match_model $vpr_match_model \
        --image_match_model $image_match_model \
        --num_preds_to_save 3 \
        --image_size 512 288 \
        --device cuda \
        --out_dir $OUT_DIR
      echo ""
    done
  done
done