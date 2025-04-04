#!/bin/bash

##### Feature matcher configuration
# Dense matching: roma tiny-roma duster master
# Semi-dense matching: loftr eloftr matchformer xfeat-star
# Sparse matching: sift-lg superpoint-lg gim-lg xfeat-lg sift-nn orb-nn gim-dkm xfeat

# Configuration
NUM_PARALLEL=2  # Set desired parallelism level here

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_NAME is not specified."
  echo "Usage: ./run_benchmark_kf_submission.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree)"
  exit 1
fi

DATASET_NAME=$1

# Set pose solver based on dataset
if [ "$DATASET_NAME" = "matterport3d" ] || [ "$DATASET_NAME" = "hkustgz_campus" ]; then
  POSE_SOLVER="pnp"
else
  POSE_SOLVER="essentialmatrix"
fi

export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/map_free_eval/$DATASET_NAME/map_free_eval/"
export KEYFRAME_PATH="/Rocket_ssd/dataset/data_litevloc/keyframe_selection_eval/$DATASET_NAME/keyframe_selection_eval"
export OUT_DIR="/Rocket_ssd/dataset/data_litevloc/map_free_eval/$DATASET_NAME/map_free_eval/results_kf"
export MODELS="master"

KF_SELECTORS=(
  "full_kf"
  "pose_density"
  "feature"
  "landmark"
)

# Function to process each selector
process_selector() {
  local kf_selector=$1
  echo "Processing with $DATASET_NAME and $kf_selector with $MODELS and $POSE_SOLVER"
  python $PROJECT_PATH/python/benchmark_kf_selection/submission.py \
    --config $CONFIG_FILE \
    --dataset_dir $DATASET_PATH \
    --keyframe_dir $KEYFRAME_PATH \
    --keyframe_selector $kf_selector \
    --image_match_models $MODELS --pose_solver $POSE_SOLVER \
    --image_size 512 288 \
    --out_dir $OUT_DIR \
    --split test #--debug
  sleep 3
}

# Export function and variables for parallel processing
export -f process_selector
export DATASET_NAME CONFIG_FILE DATASET_PATH KEYFRAME_PATH OUT_DIR MODELS POSE_SOLVER PROJECT_PATH

# Run processing in parallel
printf "%s\n" "${KF_SELECTORS[@]}" | xargs -P $NUM_PARALLEL -I {} bash -c 'process_selector "$@"' _ {}
echo "All processing completed"

