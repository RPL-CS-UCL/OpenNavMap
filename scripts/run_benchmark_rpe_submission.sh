#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
	echo "Error: DATASET_NAME is not specified."
	echo "Usage: ./run_benchmark_rpe_submission.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree, hkust_aria) 
	<SPLIT> (train, val, test)"
	exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1
SPLIT=$2

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/map_free_eval"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export OUT_DIR="$DATASET_PATH/$DATASET_NAME/map_free_eval/results_rpe"
export N_QUERY=30
export TOP_K=2

# export MODELS="master hloc_disk_dilg vpr_cosplace_resnet18_512"
export MODELS=(
	"master"
	"duster_nocalib_pretrain"
	"duster_nocalib_ftlora_pdepth"
	"duster_nocalib_ftlora_gtdepth"
)

export LORA_PATHS=(
	"none"
	"none"	
	"$DATASET_PATH/$DATASET_NAME/map_free_eval/finetune/weights/duster_lora_s00000_pdepth.pt"
	"$DATASET_PATH/$DATASET_NAME/map_free_eval/finetune/weights/duster_lora_s00000_gtdepth.pt"
)

# Run the Python script
for i in "${!MODELS[@]}"; do
		MODEL="${MODELS[$i]}"
		LORA_PATH="${LORA_PATHS[$i]}"
		echo "Processing model: $MODEL with LoRA weight: $LORA_PATH"

		python $PROJECT_PATH/python/benchmark_rpe/submission.py --config $CONFIG_FILE --models $MODEL \
			--out_dir $OUT_DIR --n_query $N_QUERY --top_k $TOP_K \
			--lora_path $LORA_PATH --split $SPLIT 
			#  --debug --viz
		echo ""
done			

# Unzip files
for MODEL in "${MODELS[@]}"; do
		mkdir "$OUT_DIR/$MODEL/submission_$TOP_K"
		if [ -f "$OUT_DIR/$MODEL/submission_$TOP_K.zip" ]; then
				unzip -o "$OUT_DIR/$MODEL/submission_$TOP_K.zip" -d "$OUT_DIR/$MODEL/submission_$TOP_K"
		else
				echo "Error: submission_$TOP_K.zip not found for model $MODEL"
		fi
done