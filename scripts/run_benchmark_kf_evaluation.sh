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
  echo "Usage: ./run_benchmark_kf_evaluation.sh <DATASET_NAME> (matterport3d, hkustgz_campus, ucl_campus, mapfree)"
  exit 1
fi

# Set the DATASET_NAME variable from the first argument
DATASET_NAME=$1

# Export environment variables
export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/$DATASET_NAME.yaml"
export DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/map_free_eval/$DATASET_NAME/map_free_eval"
export EVAL_CONFIGS=("config_025_5") # Optional: config_005_5, config_025_5, config_05_10, config_1_10, config_2_20

export MODELS=(
  "master_pnp"
  "master_essentialmatrix"
  # "roma_pnp"
  # "tiny-roma_pnp"
  # "duster_pnp"
  # "master_pnp"
  # "loftr_pnp"
  # "eloftr_pnp"
  # "matchformer_pnp"
  # "xfeat-star_pnp"
  # "sift-lg_pnp"
  # "superpoint-lg_pnp"
  # "gim-lg_pnp"
  # "xfeat-lg_pnp"
  # "sift-nn_pnp"
  # "orb-nn_pnp"
  # "gim-dkm_pnp"
  # "xfeat_pnp"
)

export KF_SELECTORS=(
  "full_kf"
  "pose_density"
  "feature"
  "landmark"
)

# Set evaluation based on dataset
for EVAL_CONFIG in "${EVAL_CONFIGS[@]}"
do
  for kf_selector in "${KF_SELECTORS[@]}"
  do
    for model in "${MODELS[@]}"
    do
      if [ "$DATASET_NAME" = "matterport3d" ] || [ "$DATASET_NAME" = "hkustgz_campus" ]; then
        echo "Evaluate image matching methods with pose solver: $model and $kf_selector with scale"
        python $PROJECT_PATH/python/benchmark_kf_selection/evaluation.py \
          --submission_path $DATASET_PATH/results_kf/"$model"_"$kf_selector"/submission.zip \
          --dataset_path $DATASET_PATH \
          --eval_config $EVAL_CONFIG \
          --enable_scale \
          --split test \
          --log error
        echo ""
      else
        echo "Evaluate image matching methods with pose solver: $model and $kf_selector without scale"
        python $PROJECT_PATH/python/benchmark_kf_selection/evaluation.py \
          --submission_path $DATASET_PATH/results_kf/"$model"_"$kf_selector"/submission.zip \
          --dataset_path $DATASET_PATH \
          --eval_config $EVAL_CONFIG \
          --split test \
          --log error
        echo ""
      fi
    done
  done
done

