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
methods=("cosplace_single_match_none" 
         "cosplace_single_match_master"
         "cosplace_sequence_match_none"
         "cosplace_sequence_match_master"
         "cosplace_sequence_match_ransac_none"
         "cosplace_sequence_match_ransac_master"
        )

# methods=("cosplace_single_match_none" 
#          "cosplace_single_match_master"
#         )

# Evaluation and generate report_evaluation.txt
for method in "${methods[@]}"
do
  echo "Evaluate VPR methods: $method"
  python $PROJECT_PATH/python/benchmark_vpr/evaluation.py \
    --result_dir $DATASET_PATH/results_vpr/$method \
    --dataset_path $DATASET_PATH \
    --tsl_thre 7.5 \
    --ang_thre 75.0 \
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

