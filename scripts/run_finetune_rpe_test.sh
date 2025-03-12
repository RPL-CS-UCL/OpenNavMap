#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
	echo "Error: DATASET_NAME is not specified."
	echo "Usage: ./run_finetune_rpe_test.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree, hkust_aria, ucl_campus_aria)"
	exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1
export LITEVLOC_PROJ_DIR="/Titan/code/robohike_ws/src/litevloc"
export ESTIMATOR_PROJ_DIR="/Titan/code/robohike_ws/src/pose_estimation_models/estimator/third_party/duster"
export PYCPPTOOLS_PROJ_DIR="/Titan/code/robohike_ws/src/pycpptools"
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/map_free_eval/$DATASET_NAME/map_free_eval"

# Run Depth Generation
echo "Run Depth Generation"
rosrun litevloc run_benchmark_rpe_depth_generation.sh \
  $DATASET_NAME 1 "$DATASET_PATH"/finetune

# python "$PYCPPTOOLS_PROJ_DIR"/pycpptools/src/python/utils_dataset/map_free_reloc/merge_pair_name.py \
#   --dataset_dir "$DATASET_PATH" --depth_suffix pdepth

# python "$PYCPPTOOLS_PROJ_DIR"/pycpptools/src/python/utils_dataset/map_free_reloc/merge_pair_name.py \
#   --dataset_dir "$DATASET_PATH" --depth_suffix gtdepth

# python "$PYCPPTOOLS_PROJ_DIR"/pycpptools/src/python/utils_dataset/map_free_reloc/merge_pair_name.py \
#   --dataset_dir "$DATASET_PATH" --depth_suffix m3ddepth

# find "$DATASET_PATH"/test/ -name "*.pdepth.png" -delete
# cp -r "$DATASET_PATH"/finetune/pairs/s* "$DATASET_PATH"/train/

# # Run Training
# echo "Run Training"
# cd $ESTIMATOR_PROJ_DIR
# ./test_lora/scripts/run_train_finetune_lora.sh "$DATASET_PATH" $DATASET_NAME
# cd $LITEVLOC_PROJ_DIR

# # Run RPE Evaluation
# echo "Run RPE Evaluation"
# rosrun litevloc run_benchmark_rpe_submission.sh $DATASET_NAME test
# rosrun litevloc run_benchmark_rpe_evaluation.sh $DATASET_NAME test