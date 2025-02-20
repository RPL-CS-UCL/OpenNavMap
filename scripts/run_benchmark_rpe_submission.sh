#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
	echo "Error: DATASET_NAME is not specified."
	echo "Usage: ./run_benchmark_rpe_submission.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree)"
	exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export OUT_DIR="/Rocket_ssd/dataset/data_litevloc/map_free_eval/$DATASET_NAME/map_free_eval/results_rpe"
export N_QUERY=10
export TOP_K=2

# export MODELS="master hloc_disk_dilg vpr_cosplace_resnet18_512"
# export MODELS="duster duster_calib"
export MODELS="duster_lora"

# Run the Python script
python $PROJECT_PATH/python/benchmark_rpe/submission.py --config $CONFIG_FILE --models $MODELS \
  --out_dir $OUT_DIR --n_query $N_QUERY --top_k $TOP_K \
  --split test --debug

# Unzip files
export TMP_MODELS=($MODELS)
for MODEL in "${TMP_MODELS[@]}"; do
		mkdir "$OUT_DIR/$MODEL/submission_$TOP_K"
		if [ -f "$OUT_DIR/$MODEL/submission_$TOP_K.zip" ]; then
				unzip -o "$OUT_DIR/$MODEL/submission_$TOP_K.zip" -d "$OUT_DIR/$MODEL/submission_$TOP_K"
		else
				echo "Error: submission_$TOP_K.zip not found for model $MODEL"
		fi
done