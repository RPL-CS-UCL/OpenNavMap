#!/bin/bash
# Automated Ablation Study for Keyframe Culling (Parallel Version)
# Usage: bash run_map_merging_ablation_studies.sh

NUM_PARALLEL=2

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
SCENE=s00000
ORDER=0
METHOD=spgo_cc_seqmatch_master
POSE_EST=master_calib_pretrain

echo "=== Ablation Study: $SCENE ==="
# Each configuration: "IQA IG TD Description"
# (1=enabled, 0=disabled)
declare -a CONFIGS=(
    # "0 0 0 W.O. Node Culling"
    # "1 0 0 Node Culling-IQA"
    # "1 1 0 Node Culling-IQA + IG"
    "1 1 1 Node Culling-IQA + IG + TD"
)

run_config() {
    local cfg="$1"
    read -r iqa ig td desc <<< "$cfg"
    echo ">>> Running: $desc (IQA=$iqa IG=$ig TD=$td)"
    bash "${SCRIPT_DIR}/run_map_merging.sh" "$SCENE" "$ORDER" "$METHOD" "$POSE_EST" "$iqa" "$ig" "$td"
    echo ""
}

export -f run_config
export SCRIPT_DIR SCENE ORDER METHOD POSE_EST

printf "%s\n" "${CONFIGS[@]}" | xargs -P $NUM_PARALLEL -I {} bash -c 'run_config "$@"' _ {}

echo "=== All Configurations Completed ==="
echo "Results: (no suffix)=no factors, _iqa=IQA, _ig=IG, _td=TD, _iqaigtd=all"
