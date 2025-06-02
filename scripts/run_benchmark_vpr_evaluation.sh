#!/bin/bash

# Check if DATASET_PATH is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_PATH is not specified."
  echo "Usage: ./run_benchmark_vpr_evaluation.sh <DATASET_PATH>"
  exit 1
fi

DATASET_PATH=$1

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
METHODS="
netvlad_VGG16_4096_single_match_1_none 
netvlad_VGG16_4096_single_match_1_master 
netvlad_VGG16_4096_sequence_match_10_none 
netvlad_VGG16_4096_sequence_match_10_master 
netvlad_VGG16_4096_sequence_match_adaptive_10_none 
netvlad_VGG16_4096_sequence_match_adaptive_10_master 
cosplace_ResNet18_256_single_match_1_none 
cosplace_ResNet18_256_sequence_match_10_none 
cosplace_ResNet18_256_sequence_match_adaptive_10_none 
cosplace_ResNet18_256_single_match_1_master 
cosplace_ResNet18_256_sequence_match_10_master 
cosplace_ResNet18_256_sequence_match_adaptive_10_master 
eigenplaces_ResNet18_256_single_match_1_none 
eigenplaces_ResNet18_256_sequence_match_10_none 
eigenplaces_ResNet18_256_sequence_match_adaptive_10_none 
eigenplaces_ResNet18_256_single_match_1_master 
eigenplaces_ResNet18_256_sequence_match_10_master 
eigenplaces_ResNet18_256_sequence_match_adaptive_10_master 
anyloc-structured_DINOv2_49152_single_match_1_none 
anyloc-structured_DINOv2_49152_sequence_match_10_none 
anyloc-structured_DINOv2_49152_sequence_match_adaptive_10_none 
anyloc-structured_DINOv2_49152_single_match_1_master 
anyloc-structured_DINOv2_49152_sequence_match_10_master 
anyloc-structured_DINOv2_49152_sequence_match_adaptive_10_master
"

# METHODS="
# cosplace_ResNet18_256_sequence_match_20_none 
# "

# Evaluation and generate report_evaluation.txt
echo "Evaluate VPR methods: "
python $PROJECT_PATH/python/benchmark_vpr/evaluation.py \
  --result_dir $DATASET_PATH/results_vpr \
  --methods $METHODS \
  --dataset_path $DATASET_PATH \
  --trans_threshold 7.5 \
  --ori_threshold 75.0 \
  --log warning \
  --option eval
echo ""

# Evaluation and summarize report_evaluation.csv and runtime_results.csv
python $PROJECT_PATH/python/benchmark_vpr/evaluation.py \
  --result_dir $DATASET_PATH/results_vpr \
  --dataset_path $DATASET_PATH \
  --log warning \
  --option summ

