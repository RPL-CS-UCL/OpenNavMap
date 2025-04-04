#!/bin/bash

##### Feature matcher configuration
# Dense matching: roma tiny-roma duster master
# Semi-dense matching: loftr eloftr matchformer xfeat-star
# Sparse matching: sift-lg superpoint-lg gim-lg xfeat-lg sift-nn orb-nn gim-dkm xfeat

# Configuration
NUM_PARALLEL=4 # Set desired parallelism level

# Validate input
if [ -z "$1" ]; then
  echo "Error: DATASET_NAME is not specified."
  echo "Usage: ./script.sh <DATASET_NAME> (e.g., matterport3d, hkustgz_campus, ucl_campus_aria)"
  exit 1
fi

DATASET_NAME=$1
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/map_free_eval/$DATASET_NAME/map_free_eval/test"
export KF_PATH="/Rocket_ssd/dataset/data_litevloc/keyframe_selection_eval/$DATASET_NAME/keyframe_selection_eval/test"
export ESTIMATOR="master"
export MATCHER="loftr"

# Get scenes with 4-digit numbering and sort numerically
SCENES=($(python -c "import os; print('\n'.join([d for d in sorted(os.listdir('$DATASET_PATH')) if os.path.isdir(os.path.join('$DATASET_PATH', d))]))"))

export KF_SELECTORS=(
  "full_kf"
  "pose_density"
  "feature"
  "landmark"
)

# Processing functions
process_precompute() {
  local scene=$1
  echo "======= Pre-Computing $scene ======="
  python python/benchmark_kf_selection/keyframe_selection.py \
    --keyframe_path "$KF_PATH" \
    --scenes "$scene" \
    --matcher "$MATCHER" \
    --estimator "$ESTIMATOR"
}

process_combination() {
  local combination=$1
  IFS=':' read -r selector scene <<< "$combination"
  echo "======= Processing $scene with $selector ======="
  python python/benchmark_kf_selection/keyframe_selection.py \
    --keyframe_path "$KF_PATH" \
    --method "$selector" \
    --scenes "$scene" \
    --matcher "$MATCHER" \
    --estimator "$ESTIMATOR"
}

# Export functions and variables
export -f process_precompute process_combination

##### Precomputation
printf "%s\n" "${SCENES[@]}" | xargs -P $NUM_PARALLEL -I {} bash -c 'process_precompute "$@"' _ {}

##### Keyframe selection
combinations=()
for selector in "${KF_SELECTORS[@]}"; do
  for scene in "${SCENES[@]}"; do
    combinations+=("${selector}:${scene}")
  done
done
printf "%s\n" "${combinations[@]}" | xargs -P $NUM_PARALLEL -I {} bash -c 'process_combination "$@"' _ {}
echo "All parallel processing completed"

##### Count number of keyframes
for scene in "${SCENES[@]}"; do
  echo "======= Counting keyframes for $scene ======="
  for selector in "${KF_SELECTORS[@]}"; do
    cat "$KF_PATH/$scene/keyframes_$selector.txt" | wc -l
  done
done
