#!/bin/bash

# Check if DATASET_NAME is provided
if [ -z "$1" ]; then
  echo "Error: DATASET_NAME is not specified."
  echo "Usage: ./script.sh <DATASET_NAME> (e.g., hkustgz_campus, ucl_campus)"
  exit 1
fi

DATASET_NAME=$1
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/keyframe_selection_eval/$DATASET_NAME/keyframe_selection_eval/test"

# Get scenes, format to 4-digit numbering, and sort numerically
SCENES=($(python -c "import os; print('\n'.join([d for d in sorted(os.listdir('$DATASET_PATH')) if os.path.isdir(os.path.join('$DATASET_PATH', d))]))"))

export KF_SELECTORS=(
  "full_kf"
  "pose_density"
  "feature"
  "landmark"
)

##### Process all scenes for precomputing
for scene in "${SCENES[@]}"; do
  echo "======= Pre-Computing $scene ======="
  python python/benchmark_kf_selection/keyframe_selection.py \
    --dataset_path "$DATASET_PATH" \
    --scenes "$scene"
done

##### Process all scenes with all keyframe selectors
for kf_selector in "${KF_SELECTORS[@]}"; do
  for scene in "${SCENES[@]}"; do
    echo "======= Processing $scene with $kf_selector ======="
    python python/benchmark_kf_selection/keyframe_selection.py \
      --dataset_path "$DATASET_PATH" \
      --method "$kf_selector" \
      --scenes "$scene"
  done
done
