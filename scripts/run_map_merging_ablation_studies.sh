#!/bin/bash

# Automated Ablation Study Script for Keyframe Culling in Map Merging
# This script runs multiple configurations to evaluate the impact of different factors
# 
# Usage:
#   bash run_ablation_studies.sh

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration from command line
ORDER=0
METHOD=spgo_cc_seqmatch
POSE_ESTIMATION_METHOD=master_calib_pretrain
SCENE=s00003_exp_culling

echo "=========================================="
echo "Starting Ablation Study for Node Culling"
echo "=========================================="
echo "Order: $ORDER"
echo "Method: $METHOD"
echo "Pose Estimation: $POSE_ESTIMATION_METHOD"
echo "Scene: ${SCENE:-default}"
echo "=========================================="
echo ""

# Define all ablation configurations based on TABLE VI
# Each configuration: "IQA-F IQA-B TD IG Description"
# (1=enabled, 0=disabled)
declare -a CONFIGS=(
    # "0 0 0 0 0 WO_Node_Culling"
    # "1 0 1 0 0 IQA_Forward_and_IQA_Backward"
    # "1 1 1 1 0 IQA_Forward_and_IG_Forward_and_IQA_Backward_and_IG_Backward"
    "1 1 1 1 1 ALL_factors"
)

# Run each configuration
for config in "${CONFIGS[@]}"; do
    read -r iqa_f iqa_b ig_f ig_b desc <<< "$config"
    
    echo "=========================================="
    echo "Running Configuration: $desc"
    echo "IQA-F: $iqa_f | IQA-B: $iqa_b | IG-F: $ig_f | IG-B: $ig_b"
    echo "=========================================="
    
    # Set environment variables based on configuration
    export USE_IQA_FORWARD=$([ "$iqa_f" -eq 1 ] && echo "--use_iqa_forward" || echo "")
    export USE_IQA_BACKWARD=$([ "$iqa_b" -eq 1 ] && echo "--use_iqa_backward" || echo "")
    export USE_IG_FORWARD=$([ "$ig_f" -eq 1 ] && echo "--use_ig_forward" || echo "")
    export USE_IG_BACKWARD=$([ "$ig_b" -eq 1 ] && echo "--use_ig_backward" || echo "")
    
    # Run the main script
    if [ -n "$SCENE" ]; then
        bash "${SCRIPT_DIR}/run_map_merging.sh" "$ORDER" "$METHOD" "$POSE_ESTIMATION_METHOD" "$SCENE"
    else
        bash "${SCRIPT_DIR}/run_map_merging.sh" "$ORDER" "$METHOD" "$POSE_ESTIMATION_METHOD"
    fi
    
    echo "Completed: $desc"
    echo ""
done

echo "=========================================="
echo "All Ablation Studies Completed!"
echo "=========================================="
echo ""
echo "Results are saved in separate directories with suffixes:"
echo "  _iqaf : IQA Forward"
echo "  _iqab : IQA Backward"
echo "  _gf   : Information Gain Forward"
echo "  _gb   : Information Gain Backward"
echo "  _iqaf_iqab_gf_gb_t : All factors enabled"
echo "  (none): No factors enabled"
echo ""
