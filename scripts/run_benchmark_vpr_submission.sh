#!/bin/bash

# Check if DATASET_PATH is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_PATH is not specified."
  echo "Usage: ./run_benchmark_vpr_submission.sh <DATASET_PATH>"
  exit 1
fi

DATASET_PATH=$1

for db_dir in "$DATASET_PATH"/database/*/; do
  DATABASE_NAME="$(basename "$db_dir")"
  for query_dir in "$DATASET_PATH"/query/*/; do
    QUERY_NAME="$(basename "$query_dir")"
    echo "Processing $DATABASE_NAME $QUERY_NAME"

    # Export environment variables
    export PROJECT_PATH="/Titan/code/robohike_ws/src/opennavmap"
    export DATABASE_PATH="$DATASET_PATH/database/$DATABASE_NAME"
    export QUERY_PATH="$DATASET_PATH/query/$QUERY_NAME"
    export OUT_DIR="$DATASET_PATH/results_vpr"
    
    ##### Setting for baseline comparison
    STR_BACKBONES="VGG16 ResNet18 ResNet18 DINOv2 DINOv2"
    STR_DESC_DIMENSIONS="4096 256 256 49152 8448"
    STR_VPR_MODELS="netvlad cosplace eigenplaces anyloc-structured megaloc"
    VPR_MATCH_MODELS="single_match seqslam vpr_dp"
    VPR_MATCH_SEQ_LENS="20"
    IMAGE_MATCH_MODELS="none master"
    ##### Default Setting
    # STR_BACKBONES="ResNet18"
    # STR_DESC_DIMENSIONS="256"
    # STR_VPR_MODELS="cosplace"
    # VPR_MATCH_MODELS="seqslam vpr_dp"
    # VPR_MATCH_SEQ_LENS="50"
    # IMAGE_MATCH_MODELS="none master"
    ##### 
    # STR_BACKBONES="ResNet18 DINOv2 DINOv2 DINOv2"
    # STR_DESC_DIMENSIONS="512 8448 8448 49152"
    # STR_VPR_MODELS="cosplace megaloc clique-mining anyloc-unstructured" 
    # VPR_MATCH_MODELS="single_match seqslam"
    # VPR_MATCH_SEQ_LENS="20"
    # IMAGE_MATCH_MODELS="master"
    ### 
    
    python $PROJECT_PATH/python/benchmark_vpr/submission.py \
      --database_folder $DATABASE_PATH \
      --queries_folder $QUERY_PATH \
      --str_backbones $STR_BACKBONES \
      --str_descriptors_dimensions $STR_DESC_DIMENSIONS \
      --str_vpr_models $STR_VPR_MODELS \
      --vpr_match_models $VPR_MATCH_MODELS \
      --vpr_match_seq_lens $VPR_MATCH_SEQ_LENS \
      --image_match_models $IMAGE_MATCH_MODELS \
      --num_preds_to_save 3 \
      --image_size 512 288 \
      --device cuda \
      --num_workers 4 --batch_size 4 \
      --out_dir $OUT_DIR \
      --debug
      echo ""
  done
done