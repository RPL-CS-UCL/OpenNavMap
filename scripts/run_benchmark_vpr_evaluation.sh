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
METHODS=(
         "cosplace_single_match_1_none" 
         "cosplace_single_match_1_master"
         "cosplace_sequence_match_2_none"
         "cosplace_sequence_match_2_master"
         "cosplace_sequence_match_3_none"
         "cosplace_sequence_match_3_master"
         "cosplace_sequence_match_4_none"
         "cosplace_sequence_match_4_master"
         "cosplace_sequence_match_5_none"
         "cosplace_sequence_match_5_master"
         "cosplace_sequence_match_12_none"
         "cosplace_sequence_match_12_master"
         "cosplace_sequence_match_20_none"
         "cosplace_sequence_match_20_master"
         "cosplace_sequence_match_ransac_2_none"
         "cosplace_sequence_match_ransac_2_master"
         "cosplace_sequence_match_ransac_3_none"
         "cosplace_sequence_match_ransac_3_master"
         "cosplace_sequence_match_ransac_4_none"
         "cosplace_sequence_match_ransac_4_master"
         "cosplace_sequence_match_ransac_5_none"
         "cosplace_sequence_match_ransac_5_master"
         "cosplace_sequence_match_ransac_12_none"
         "cosplace_sequence_match_ransac_12_master"
         "cosplace_sequence_match_ransac_20_none"
         "cosplace_sequence_match_ransac_20_master"
        )

# METHODS=(
#          "cosplace_sequence_match_12_master"
#         )

# Evaluation and generate report_evaluation.txt
for method in "${METHODS[@]}"
do
  echo "Evaluate VPR methods: $method"
  python $PROJECT_PATH/python/benchmark_vpr/evaluation.py \
    --result_dir $DATASET_PATH/results_vpr/$method \
    --dataset_path $DATASET_PATH \
    --trans_threshold 7.5 \
    --ori_threshold 75.0 \
    --log warning \
    --option eval
  echo ""
done

# Evaluation and summarize report_evaluation.csv and runtime_results.csv
python $PROJECT_PATH/python/benchmark_vpr/evaluation.py \
  --result_dir $DATASET_PATH/results_vpr \
  --dataset_path $DATASET_PATH \
  --log warning \
  --option summ

