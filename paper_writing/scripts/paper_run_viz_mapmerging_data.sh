python ../python/viz_mapmerging_data.py \
  --data_folders \
    /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/360loc/s00000_atrium_aria_data \
    /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/360loc/s00001_concourse_aria_data \
    /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/360loc/s00002_hall_aria_data \
    /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/360loc/s00003_piatrium_aria_data \
  --labels "Atrium" "Concourse" "Hall" "Piatrium" \
  --output_folder /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/360loc/figures/ \
  --result_folders \
    /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/360loc/s00000_atrium_results_in_kf_spgo_cc_seqmatch_master \
    /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/360loc/s00001_concourse_results_in_kf_spgo_cc_seqmatch_master \
    /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/360loc/s00002_hall_results_in_kf_spgo_cc_seqmatch_master \
    /Rocket_ssd/dataset/data_litevloc/map_multisession_eval/360loc/s00003_piatrium_results_in_kf_spgo_cc_seqmatch_master \
  --translation_threshold 5.0 \
  --plot_legend \
  --plot_start_end_points

# python ../python/viz_mapmerging_data.py \
#   --data_folders \
#     /Titan/dataset/data_litevloc/data_tro2025/map_multisession_eval/vineyard/s00000_aria_data \
#     /Titan/dataset/data_litevloc/data_tro2025/map_multisession_eval/hkust_campus_aria/s00000_aria_data \
#     /Titan/dataset/data_litevloc/data_tro2025/map_multisession_eval/ucl_campus_aria/s00000_aria_data \
#   --labels "G0" "G1" "G2" \
#   --output_folder /Titan/dataset/data_litevloc/data_tro2025/map_multisession_eval/figures/ \
#   --result_folders \
#     /Titan/dataset/data_litevloc/data_tro2025/map_multisession_eval/vineyard/s00000_results_in_kf_spgo_cc_seqmatch \
#     /Titan/dataset/data_litevloc/data_tro2025/map_multisession_eval/hkust_campus_aria/s00000_results_in_kf_spgo_cc_seqmatch \
#     /Titan/dataset/data_litevloc/data_tro2025/map_multisession_eval/ucl_campus_aria/s00000_results_r4_kf_spgo_cc_seqmatch \
#   --translation_threshold 5.0