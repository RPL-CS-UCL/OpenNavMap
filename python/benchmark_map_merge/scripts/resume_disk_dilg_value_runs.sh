#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <env> <sfm_root> <value0_result_dir> <value1_log> <value2_log> <extra_run_baseline_args...>"
  exit 1
fi

ENV_NAME=$1
SFM_ROOT=$2
VALUE0_RESULT_DIR=$3
VALUE1_LOG=$4
VALUE2_LOG=$5
shift 5
EXTRA_ARGS=("$@")

VALUE0_SUMMARY="${VALUE0_RESULT_DIR}/metrics/summary.json"

while [[ ! -f "$VALUE0_SUMMARY" ]]; do
  sleep 30
done

bash "${SCRIPT_DIR}/run_baseline.sh" \
  --mode merge \
  --method hloc_sfm_netvlad_disk_dilg \
  --env "$ENV_NAME" \
  --sfm-sample-dist 0.25 \
  --prebuilt-sfm-root "$SFM_ROOT" \
  --num-retrieval 10 \
  --geo-verify-min-matches 400 \
  --pnp-min-inliers 110 \
  --overwrite \
  --clean-work \
  "${EXTRA_ARGS[@]}" >> "$VALUE1_LOG" 2>&1

bash "${SCRIPT_DIR}/run_baseline.sh" \
  --mode merge \
  --method hloc_sfm_netvlad_disk_dilg \
  --env "$ENV_NAME" \
  --sfm-sample-dist 0.25 \
  --prebuilt-sfm-root "$SFM_ROOT" \
  --num-retrieval 10 \
  --geo-verify-min-matches 500 \
  --pnp-min-inliers 150 \
  --overwrite \
  --clean-work \
  "${EXTRA_ARGS[@]}" >> "$VALUE2_LOG" 2>&1
