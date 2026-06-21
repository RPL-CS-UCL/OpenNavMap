#!/bin/bash
# One-off script: build SFM for sub0+sub1 with sfm_ba_iter=10.
# Outputs to ${DATASET_ROOT}_tmp/s00000_sfm_netvlad_splg_025 via symlink trick,
# leaving the original ${DATASET_ROOT}/s00000_sfm_netvlad_splg_025 untouched.
#
# Usage:
#   bash run_sfm_tmp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON=/root/miniconda3/envs/opennavmap/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/opennavmap/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/third_party/pose_estimation_models/estimator:${PYTHONPATH:-}"

DATASET_ROOT=/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria
TMP_ROOT="${DATASET_ROOT}_tmp"
DATA_DIR=s00000_aria_data_000

# ---------------------------------------------------------------------------
# Set up tmp root with symlinks to real data (no data is copied)
# ---------------------------------------------------------------------------
mkdir -p "$TMP_ROOT"

DATA_LINK="${TMP_ROOT}/${DATA_DIR}"
ORDERS_LINK="${TMP_ROOT}/s00000_orders.txt"

[[ -L "$DATA_LINK" ]]   && rm "$DATA_LINK"
[[ -L "$ORDERS_LINK" ]] && rm "$ORDERS_LINK"

ln -s "${DATASET_ROOT}/${DATA_DIR}" "$DATA_LINK"
ln -s "${DATASET_ROOT}/s00000_orders.txt" "$ORDERS_LINK"

echo "=== run_sfm_tmp.sh ==="
echo "Tmp root   : $TMP_ROOT"
echo "Data link  : $DATA_LINK -> ${DATASET_ROOT}/${DATA_DIR}"
echo "Orders link: $ORDERS_LINK"
echo ""

# ---------------------------------------------------------------------------
# Run SFM (sub0 + sub1 only, sfm_ba_iter=10)
# Result lands in: ${TMP_ROOT}/s00000_sfm_netvlad_splg_025
# ---------------------------------------------------------------------------
"$PYTHON" "${PROJECT_ROOT}/python/benchmark_map_merge/run_baseline.py" \
  --dataset-root    "$TMP_ROOT" \
  --method          hloc_sfm_netvlad_splg \
  --order-index     0 \
  --data-dir        "$DATA_DIR" \
  --sfm-sample-dist 0.25 \
  --sfm-ba-iter     10 \
  --max-submaps     2 \
  --submap-sfm

# ---------------------------------------------------------------------------
# Clean up symlinks (result directory is kept)
# ---------------------------------------------------------------------------
[[ -L "$DATA_LINK" ]]   && rm "$DATA_LINK"
[[ -L "$ORDERS_LINK" ]] && rm "$ORDERS_LINK"

echo ""
echo "Done. Results at: ${TMP_ROOT}/s00000_sfm_netvlad_splg_025"
