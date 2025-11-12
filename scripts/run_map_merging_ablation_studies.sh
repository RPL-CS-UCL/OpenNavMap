#!/bin/bash
# Automated Ablation Study for Keyframe Culling
# Usage: bash run_map_merging_ablation_studies.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
SCENE=s00003_exp_culling
ORDER=0
METHOD=spgo_cc_seqmatch_master
POSE_EST=master_calib_pretrain

echo "=== Ablation Study: $SCENE ==="

# Each configuration: "IQA IG TD Description"
# (1=enabled, 0=disabled)
declare -a CONFIGS=(
    "0 0 0 WO_Node_Culling"
    # "1 0 0 IQA"
    # "1 1 0 IQA + IG"
    # "1 1 1 IQA + IG + TD"
)

for cfg in "${CONFIGS[@]}"; do
    read -r iqa ig td desc <<< "$cfg"
    echo ">>> Running: $desc (IQA=$iqa IG=$ig TD=$td)"
    bash "${SCRIPT_DIR}/run_map_merging.sh" "$SCENE" "$ORDER" "$METHOD" "$POSE_EST" "$iqa" "$ig" "$td"
    echo ""
done

echo "=== All Configurations Completed ==="
echo "Results: (no suffix)=no factors, _iqa=IQA, _ig=IG, _td=TD, _iqaigtd=all"
