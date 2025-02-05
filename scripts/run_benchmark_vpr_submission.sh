#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_NAME is not specified."
  echo "Usage: ./run_benchmark_vpr_submission.sh <DATASET_PATH> <DATABASE_NAME> <QUERY_NAME>"
  exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_PATH=$1
DATABASE_NAME=$2
QUERY_NAME=$3

# Export environment variables
export PROJECT_PATH="/home/jjiao/robohike_ws/src/litevloc_private"
export DATABASE_PATH="$DATASET_PATH/database/$DATABASE_NAME"
export QUERY_PATH="$DATASET_PATH/query/$QUERY_NAME"
export OUT_DIR="$DATASET_PATH/results_vpr"

export VPR_MODEL="cosplace"
export BACKBONE="ResNet18"
export DESC_DIMENSION="256"
export MATCH_MODEL="sequence_match"

export IMAGE_MATCH_MODEL="none"

# Run the Python script
python $PROJECT_PATH/python/benchmark_vpr/submission.py \
  --database_folder $DATABASE_PATH \
  --queries_folder $QUERY_PATH \
  --method $VPR_MODEL \
  --backbone $BACKBONE \
  --descriptors_dimension $DESC_DIMENSION \
  --match_model $MATCH_MODEL \
  --image_match_model $IMAGE_MATCH_MODEL \
  --num_preds_to_save 3 \
  --image_size 512 288 \
  --device cuda \
  --out_dir $OUT_DIR \
  # --debug