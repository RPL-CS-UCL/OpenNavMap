#!/bin/bash

# Example script to compare multiple map merging methods

BASE_DIR="/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_aria"
SUBMAP_DATA_FOLDER="s00003_exp_culling_aria_data"

cd "$(dirname "$0")/../python"

# Compare multiple methods
python3 viz_mapmerging_lifelong.py \
    --base_dir "$BASE_DIR" \
    --result_dirs \
        "s00003_exp_culling_results_in_spgo_cc_seqmatch_master" \
        "s00003_exp_culling_results_in_kf_spgo_cc_seqmatch_master" \
    --labels \
        "SPGO+CC+SeqMatch" \
        "KF+SPGO+CC+SeqMatch" \
    --submap_data_folder "$SUBMAP_DATA_FOLDER"

