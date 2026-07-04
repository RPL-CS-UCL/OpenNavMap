#!/bin/bash
# Trajectory evaluation using slam_trajectory_evaluation.
#
# Runs two steps:
#   1. add_eval_cfg_recursive.py  – ensure eval_cfg.yaml (se3, all frames) exists
#   2. analyze_trajectories_FusionPortable_dataset.py – compute ATE, RPE, plots
#
# Usage:
#   bash run_evaluation.sh [OPTIONS]
#
# Options:
#   --config NAME    yaml config under analyze_trajectories_config/
#                    (default: OpenNavMap_map_merge.yaml)
#   --output-dir DIR Override report output directory
#                    (default: /Titan/dataset/data_opennavmap/traj_eval_data/map_merge_eval_data/report)
#   --recalculate    Force recalculate errors, ignore cached results
#                    (default: always on; pass --no-recalculate to disable)
#   --no-recalculate Use cached results if available
#
# Examples:
#   bash run_evaluation.sh
#   bash run_evaluation.sh --config OpenNavMap_map_merge.yaml
#   bash run_evaluation.sh --config OpenNavMap_map_merge.yaml --no-recalculate
#
# This script can also be called from run_baseline.sh (--mode merge).

set -euo pipefail

export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON=${PYTHON:-/root/miniconda3/envs/traj_evaluation/bin/python}
TRAJ_PATH=${TRAJ_PATH:-/Titan/dataset/data_opennavmap/traj_eval_data/map_merge_eval_data}
EVAL_PROJ=${EVAL_PROJ:-"$PROJECT_PATH/third_party/slam_trajectory_evaluation"}
EVAL_SCRIPT_PATH="$EVAL_PROJ/evaluation/rpg_trajectory_evaluation"
REPORT_DIR="$TRAJ_PATH/report"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONFIG=OpenNavMap_map_merge.yaml
RECALCULATE=1   # on by default to avoid stale cached results

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG=$2; shift ;;
    --output-dir)
      REPORT_DIR=$2; shift ;;
    --recalculate)
      RECALCULATE=1 ;;
    --no-recalculate)
      RECALCULATE= ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Step 1: ensure eval_cfg.yaml (align_type: se3, align_num_frames: -1)
#         exists under every algorithms/<alg>/laptop/traj/ directory
# ---------------------------------------------------------------------------
"$PYTHON" "$EVAL_SCRIPT_PATH/scripts/add_eval_cfg_recursive.py" \
  "$TRAJ_PATH/algorithms/" se3 -1

# ---------------------------------------------------------------------------
# Step 2: run evaluation
# ---------------------------------------------------------------------------
EVAL_CMD=(
  "$PYTHON" "$EVAL_SCRIPT_PATH/scripts/analyze_trajectories_FusionPortable_dataset.py"
  --groundtruth_dir="$TRAJ_PATH/groundtruth"
  --results_dir="$TRAJ_PATH/algorithms"
  --output_dir="$REPORT_DIR"
  --computer=laptop
  --mul_trials=0
  --overall_odometry_error
  --odometry_error_per_dataset
  --rmse_boxplot
  --rmse_table
  --rmse_table_alg_col
  --plot_trajectories
  --write_time_statistics
  --no_sort_names
)
[[ -n "$RECALCULATE" ]] && EVAL_CMD+=(--recalculate_errors)
EVAL_CMD+=("$CONFIG")

echo "=== run_evaluation.sh ==="
echo "Config     : $CONFIG"
echo "Recalculate: $( [[ -n "$RECALCULATE" ]] && echo yes || echo no )"
echo "Report dir : $REPORT_DIR"
echo "Command    : ${EVAL_CMD[*]}"
echo ""

"${EVAL_CMD[@]}"

echo ""
echo "=== Evaluation complete. Report: $REPORT_DIR ==="
