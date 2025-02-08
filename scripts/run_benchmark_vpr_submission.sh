#!/bin/bash

# Check if DATASET_PATH is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_PATH is not specified."
  echo "Usage: ./run_benchmark_vpr_submission.sh <DATASET_PATH> <DATABASE_NAME> <QUERY_NAME> <VPR_MATCH_MODEL> <IMAGE_MATCH_MODEL>"
  echo "Or   : ./run_benchmark_vpr_submission.sh <DATASET_PATH>"
  exit 1
fi

DATASET_PATH=$1

for db_dir in "$DATASET_PATH"/database/*/; do
  DATABASE_NAME="$(basename "$db_dir")"
  for query_dir in "$DATASET_PATH"/query/*/; do
    QUERY_NAME="$(basename "$query_dir")"

    # Export environment variables
    export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
    export DATABASE_PATH="$DATASET_PATH/database/$DATABASE_NAME"
    export QUERY_PATH="$DATASET_PATH/query/$QUERY_NAME"
    export OUT_DIR="$DATASET_PATH/results_vpr"
    export BACKBONE="ResNet18"
    export DESC_DIMENSION="256"

    # Set evaluation methods
    VPR_MODELS="cosplace"
    VPR_MATCH_MODELS="single_match sequence_match sequence_match_ransac"
    IMAGE_MATCH_MODELS="none master"
    VPR_MATCH_SEQ_LENS="5 12 20"

    # VPR_MODELS="cosplace"
    # VPR_MATCH_MODELS="sequence_match_ransac"
    # IMAGE_MATCH_MODELS="none"

    python $PROJECT_PATH/python/benchmark_vpr/submission.py \
      --database_folder $DATABASE_PATH \
      --queries_folder $QUERY_PATH \
      --backbone $BACKBONE \
      --descriptors_dimension $DESC_DIMENSION \
      --vpr_models $VPR_MODELS \
      --vpr_match_models $VPR_MATCH_MODELS \
      --vpr_match_seq_lens $VPR_MATCH_SEQ_LENS \
      --image_match_models $IMAGE_MATCH_MODELS \
      --num_preds_to_save 3 \
      --image_size 512 288 \
      --device cuda \
      --out_dir $OUT_DIR
      echo ""

  done
done
