#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_NAME is not specified."
  echo "Usage: ./run_benchmark_evaluation.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree, hkust_aria)
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
export N_QUERY=20 # same to submission
export TOP_K=2

# models=("master" "duster" "hloc_disk_dilg" "vpr_cosplace_resnet18_512")
export MODELS=(
  "master"
	"duster_nocalib_pretrain"
	"duster_calib_pretrain"
  "duster_nocalib_ftlora_20pdepth"
  "duster_nocalib_ftlora_20gtdepth"
	# "duster_calib_ftlora_12pdepth"
	# "duster_calib_ftlora_12gtdepth"
	# "duster_calib_ftlora_16pdepth"
	# "duster_calib_ftlora_16gtdepth"
	# "duster_calib_ftlora_20pdepth"
	# "duster_calib_ftlora_20gtdepth"
)

# Run the Python script
for model in "${MODELS[@]}"
do
  echo "Evaluate pose_estimation methods: $model"
  
  python $PROJECT_PATH/python/benchmark_rpe/evaluation.py \
    --config $CONFIG_FILE \
    --submission_path $DATASET_PATH/$DATASET_NAME/map_free_eval/results_rpe/$model/submission_$TOP_K.zip \
    --dataset_path $DATASET_PATH/$DATASET_NAME/map_free_eval \
    --n_query $N_QUERY \
    --split $SPLIT \
    --log warning
  echo ""

done
