#!/bin/bash

# Example script to run the lifelong map merging visualization
# This script demonstrates how to compare multiple methods
#
# To add more methods, simply add entries to both arrays:
#   RESULT_DIRS=("method1" "method2" "method3")
#   LABELS=("Label 1" "Label 2" "Label 3")
#
# The number of RESULT_DIRS must match the number of LABELS

BASE_DIR="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_aria"

# Use bash arrays to store multiple result directories and labels
# Add or remove entries as needed - just keep the arrays the same length
RESULT_DIRS=(
    "s00003_exp_culling_results_in_spgo_cc_seqmatch_master"
    "s00003_exp_culling_results_in_spgo_cc_seqmatch_master_iqa"
    "s00003_exp_culling_results_in_spgo_cc_seqmatch_master_iqaig"
    "s00003_exp_culling_results_in_spgo_cc_seqmatch_master_iqaigtd"
)

LABELS=(
    "W.O. Culling"
    "Culling with Factors: IQA"
    "Culling with Factors: IQA+IG"
    "Culling with Factors: IQA+IG+TD"
)

SUBMAP_DATA_FOLDER="s00003_exp_culling_aria_data"

cd "$(dirname "$0")/../python"

# Use "${array[@]}" to expand array elements as separate arguments
python3 viz_mapmerging_lifelong.py \
    --base_dir "$BASE_DIR" \
    --result_dirs "${RESULT_DIRS[@]}" \
    --labels "${LABELS[@]}" \
    --submap_data_folder "$SUBMAP_DATA_FOLDER" \
    --analyze_culling \
    --trajectory_merge_folder "merge_finalmap" \
    --time_gap_threshold 30.0 \
    --viz_trajectory

