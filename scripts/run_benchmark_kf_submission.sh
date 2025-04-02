#!/bin/bash

# Dense matching: 
#   roma tiny-roma duster master
# Semi-dense matching:
#   loftr eloftr matchformer xfeat-star
# Sparse matching:
#   sift-lg superpoint-lg gim-lg xfeat-lg sift-nn orb-nn gim-dkm xfeat

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_NAME is not specified."
  echo "Usage: ./run_benchmark_kf_submission.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree)"
  exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1

export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/map_free_eval/$DATASET_NAME/map_free_eval/"
export KEYFRAME_PATH="/Rocket_ssd/dataset/data_litevloc/keyframe_selection_eval/$DATASET_NAME/keyframe_selection_eval"
export OUT_DIR="/Rocket_ssd/dataset/data_litevloc/map_free_eval/$DATASET_NAME/map_free_eval/results_kf"
# export MODELS="roma tiny-roma duster master loftr eloftr matchformer xfeat-star sift-lg superpoint-lg gim-lg xfeat-lg sift-nn orb-nn gim-dkm xfeat"
export MODELS="master"

echo "Processing with $DATASET_NAME"
python $PROJECT_PATH/python/benchmark_kf_selection/submission.py \
  --config $CONFIG_FILE \
  --dataset_dir $DATASET_PATH \
  --keyframe_dir $KEYFRAME_PATH \
  --image_match_models $MODELS --pose_solver pnp \
  --image_size 512 288 \
  --out_dir $OUT_DIR \
  --split test --debug
