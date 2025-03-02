#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
	echo "Error: DATASET_NAME is not specified."
	echo "Usage: ./run_benchmark_rpe_depth_generation.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree, hkust_aria, ucl_campus_aria) 
	<N_QUERY> <OUT_DIR>" 
	exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1
N_QUERY=$2
OUT_DIR=$3

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export TOP_K=2

export MODEL="duster_calib_pretrain"

# Run the Python script
python $PROJECT_PATH/python/benchmark_rpe/pseudo_depth_generator.py --config $CONFIG_FILE --model $MODEL \
  --out_dir $OUT_DIR --n_query $N_QUERY --top_k $TOP_K \
  --pseudo_gt_thre 1.5 --save_gtpair \
	--device cuda