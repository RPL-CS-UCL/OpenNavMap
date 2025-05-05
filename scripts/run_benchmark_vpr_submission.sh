#!/bin/bash
# bash run_benchmark_vpr_submission.sh /Rocket_ssd/dataset/data_litevloc/vpr_eval/ucl_campus/s00000 && \
# bash run_benchmark_vpr_submission.sh /Rocket_ssd/dataset/data_litevloc/vpr_eval/ucl_campus/s00001 && \
# bash run_benchmark_vpr_submission.sh /Rocket_ssd/dataset/data_litevloc/vpr_eval/ucl_campus/s00002 && \
# bash run_benchmark_vpr_submission.sh /Rocket_ssd/dataset/data_litevloc/vpr_eval/ucl_campus/s00003 && 
# bash run_benchmark_vpr_submission.sh /Rocket_ssd/dataset/data_litevloc/vpr_eval/ucl_campus/s00004 && \
# bash run_benchmark_vpr_submission.sh /Rocket_ssd/dataset/data_litevloc/vpr_eval/ucl_campus/s00005

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

    # Export environment variables
    export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
    export DATABASE_PATH="$DATASET_PATH/database/$DATABASE_NAME"
    export QUERY_PATH="$DATASET_PATH/query/$QUERY_NAME"
    export OUT_DIR="$DATASET_PATH/results_vpr"
    
    ##### Setting for Academic Paper Writing
    STR_BACKBONES="VGG16 ResNet18 ResNet18 ResNet18 ResNet18 DINOv2"
    STR_DESC_DIMENSIONS="4096 128 256 512 256 49152"
    STR_VPR_MODELS="netvlad cosplace cosplace cosplace eigenplaces anyloc-structured" 
    VPR_MATCH_MODELS="single_match sequence_match sequence_match_adaptive"
    VPR_MATCH_SEQ_LENS="10"
    IMAGE_MATCH_MODELS="master"
    ##### Default Setting
    # STR_BACKBONES="ResNet18"
    # STR_DESC_DIMENSIONS="256"
    # STR_VPR_MODELS="cosplace"
    # VPR_MATCH_MODELS="single_match"
    # VPR_MATCH_SEQ_LENS="1"
    # IMAGE_MATCH_MODELS="none"
    ##### 

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
      --out_dir $OUT_DIR
      echo ""
  
  done
done