#!/usr/bin/env bash

DATASET_PATH="/Titan/dataset/data_opennavmap/vpr_eval/ucl_campus"

python ../python/viz_vpr_data.py \
    --database_folder ${DATASET_PATH}/s00000/database/out_map_20241128_1105 \
    --queries_folder  ${DATASET_PATH}/s00000/query/out_map_20240904_0835 \
                      ${DATASET_PATH}/s00000/query/out_map_20241127_1722 \
                      ${DATASET_PATH}/s00000/query/out_map_20241202_1741 \
                      ${DATASET_PATH}/s00000/query/out_map_20241204_1411 \
                      ${DATASET_PATH}/s00000/query/out_map_20241204_1651 \
                      ${DATASET_PATH}/s00000/query/out_map_20241204_1718 \
                      ${DATASET_PATH}/s00000/query/out_map_20241205_1018 \
                      ${DATASET_PATH}/s00000/query/out_map_20241127_1716 \
                      ${DATASET_PATH}/s00000/query/out_map_20241202_1415 \
                      ${DATASET_PATH}/s00000/query/out_map_20241202_1746 \
                      ${DATASET_PATH}/s00000/query/out_map_20241204_1420 \
                      ${DATASET_PATH}/s00000/query/out_map_20241204_1704 \
                      ${DATASET_PATH}/s00000/query/out_map_20241205_1009 \
                      ${DATASET_PATH}/s00000/query/out_map_20241223_1728 \
    --trans_thresh 7.5 --rot_thresh 75.0 \
    --dmatrix_dir   ${DATASET_PATH}/s00000/results_vpr \
    --singlematch_dir ${DATASET_PATH}/s00000/results_vpr/cosplace_ResNet18_256_single_match_1_none \
    --seqmatch_dir  ${DATASET_PATH}/s00000/results_vpr/cosplace_ResNet18_256_sequence_match_20_none \
    --graph_dir     ${DATASET_PATH}/s00000/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_none_dir ${DATASET_PATH}/s00000/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_master_dir ${DATASET_PATH}/s00000/results_vpr/cosplace_ResNet18_256_graph_search_1_master \
    --output_path ${DATASET_PATH}/s00000/scene_stat

python ../python/viz_vpr_data.py \
    --database_folder ${DATASET_PATH}/s00001/database/out_map_20241129_1145 \
    --queries_folder  ${DATASET_PATH}/s00001/query/out_map_20241204_1432 \
                      ${DATASET_PATH}/s00001/query/out_map_20241204_1659 \
                      ${DATASET_PATH}/s00001/query/out_map_20241223_1851 \
                      ${DATASET_PATH}/s00001/query/out_map_20241223_1856 \
    --trans_thresh 7.5 --rot_thresh 75.0 \
    --dmatrix_dir   ${DATASET_PATH}/s00001/results_vpr \
    --singlematch_dir ${DATASET_PATH}/s00001/results_vpr/cosplace_ResNet18_256_single_match_1_none \
    --seqmatch_dir  ${DATASET_PATH}/s00001/results_vpr/cosplace_ResNet18_256_sequence_match_20_none \
    --graph_dir     ${DATASET_PATH}/s00001/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_none_dir ${DATASET_PATH}/s00001/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_master_dir ${DATASET_PATH}/s00001/results_vpr/cosplace_ResNet18_256_graph_search_1_master \
    --output_path ${DATASET_PATH}/s00001/scene_stat

python ../python/viz_vpr_data.py \
    --database_folder ${DATASET_PATH}/s00002/database/out_map_20241204_1439 \
    --queries_folder  ${DATASET_PATH}/s00002/query/out_map_20241204_1700 \
                      ${DATASET_PATH}/s00002/query/out_map_20241204_1707 \
                      ${DATASET_PATH}/s00002/query/out_map_20241204_1711 \
                      ${DATASET_PATH}/s00002/query/out_map_20241223_1723 \
                      ${DATASET_PATH}/s00002/query/out_map_20241223_1847 \
    --trans_thresh 7.5 --rot_thresh 75.0 \
    --dmatrix_dir   ${DATASET_PATH}/s00002/results_vpr \
    --singlematch_dir ${DATASET_PATH}/s00002/results_vpr/cosplace_ResNet18_256_single_match_1_none \
    --seqmatch_dir  ${DATASET_PATH}/s00002/results_vpr/cosplace_ResNet18_256_sequence_match_20_none \
    --graph_dir     ${DATASET_PATH}/s00002/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_none_dir ${DATASET_PATH}/s00002/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_master_dir ${DATASET_PATH}/s00002/results_vpr/cosplace_ResNet18_256_graph_search_1_master \
    --output_path ${DATASET_PATH}/s00002/scene_stat

python ../python/viz_vpr_data.py \
    --database_folder ${DATASET_PATH}/s00003/database/out_map_20241222_1243 \
    --queries_folder  ${DATASET_PATH}/s00003/query/out_map_20241221_1729 \
                      ${DATASET_PATH}/s00003/query/out_map_20241223_1733 \
                      ${DATASET_PATH}/s00003/query/out_map_20241223_1900 \
    --trans_thresh 7.5 --rot_thresh 75.0 \
    --dmatrix_dir   ${DATASET_PATH}/s00003/results_vpr \
    --singlematch_dir ${DATASET_PATH}/s00003/results_vpr/cosplace_ResNet18_256_single_match_1_none \
    --seqmatch_dir  ${DATASET_PATH}/s00003/results_vpr/cosplace_ResNet18_256_sequence_match_20_none \
    --graph_dir     ${DATASET_PATH}/s00003/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_none_dir ${DATASET_PATH}/s00003/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_master_dir ${DATASET_PATH}/s00003/results_vpr/cosplace_ResNet18_256_graph_search_1_master \
    --output_path ${DATASET_PATH}/s00003/scene_stat

python ../python/viz_vpr_data.py \
    --database_folder ${DATASET_PATH}/s00004/database/out_map_20241222_1545 \
    --queries_folder ${DATASET_PATH}/s00004/query/out_map_20241222_1549 \
                     ${DATASET_PATH}/s00004/query/out_map_20241223_1714 \
    --dmatrix_dir   ${DATASET_PATH}/s00004/results_vpr \
    --singlematch_dir ${DATASET_PATH}/s00004/results_vpr/cosplace_ResNet18_256_single_match_1_none \
    --seqmatch_dir  ${DATASET_PATH}/s00004/results_vpr/cosplace_ResNet18_256_sequence_match_20_none \
    --graph_dir     ${DATASET_PATH}/s00004/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_none_dir ${DATASET_PATH}/s00004/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_master_dir ${DATASET_PATH}/s00004/results_vpr/cosplace_ResNet18_256_graph_search_1_master \
    --output_path ${DATASET_PATH}/s00004/scene_stat

python ../python/viz_vpr_data.py \
    --database_folder ${DATASET_PATH}/s00005/database/out_map_20241222_1641 \
    --queries_folder ${DATASET_PATH}/s00005/query/out_map_20241222_1637 \
                     ${DATASET_PATH}/s00005/query/out_map_20241222_1645 \
    --dmatrix_dir   ${DATASET_PATH}/s00005/results_vpr \
    --singlematch_dir ${DATASET_PATH}/s00005/results_vpr/cosplace_ResNet18_256_single_match_1_none \
    --seqmatch_dir  ${DATASET_PATH}/s00005/results_vpr/cosplace_ResNet18_256_sequence_match_20_none \
    --graph_dir     ${DATASET_PATH}/s00005/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_none_dir ${DATASET_PATH}/s00005/results_vpr/cosplace_ResNet18_256_graph_search_1_none \
    --graph_master_dir ${DATASET_PATH}/s00005/results_vpr/cosplace_ResNet18_256_graph_search_1_master \
    --output_path ${DATASET_PATH}/s00005/scene_stat

##### FusionPortable
# python ../python/viz_vpr_data.py \
#     --database_folder /Rocket_ssd/dataset/data_litevloc/vpr_eval/fusionportable/s00000_garden/database/out_map_20220216_1235 \
#     --queries_folder /Rocket_ssd/dataset/data_litevloc/vpr_eval/fusionportable/s00000_garden/query/out_map_20220215_2030 \
#     --output_path /Rocket_ssd/dataset/data_litevloc/vpr_eval/fusionportable/s00000_garden/results_vpr

##### Robocar
# DATASET_PATH="/Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar"
# python ../python/viz_vpr_data.py \
#     --dataset_name robocar \
#     --database_folder /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00000_full_overlap/database/out_map_20141125_091832 \
#     --queries_folder /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00000_full_overlap/query/out_map_20141121_160703 \
#                      /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00000_full_overlap/query/out_map_20141217_181843 \
#                      /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00000_full_overlap/query/out_map_20150203_084510 \
#     --trans_thresh 24.0 --rot_thresh 75.0 \
#     --output_path /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00000_full_overlap/results_vpr

# python ../python/viz_vpr_data.py \
#     --dataset_name robocar \
#     --database_folder /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00000_easy/database/out_map_20141125_091832 \
#     --queries_folder /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00000_easy/query/out_map_20140519_130538 \
#                      /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00000_easy/query/out_map_20140626_093118 \
#                      /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00000_easy/query/out_map_20150817_133019 \
#     --trans_thresh 24.0 --rot_thresh 75.0 \
#     --output_path /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00000_easy/results_vpr

# python ../python/viz_vpr_data.py \
#     --dataset_name robocar \
#     --database_folder /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00001_hard/database/out_map_20140626_095312 \
#     --queries_folder /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00001_hard/query/out_map_20141111_110625 \
#                      /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00001_hard/query/out_map_20141205_154207 \
#                      /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00001_hard/query/out_map_20150203_194311 \
#                      /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00001_hard/query/out_map_20150424_081507 \
#                      /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00001_hard/query/out_map_20150724_143647 \
#     --trans_thresh 24.0 --rot_thresh 75.0 \
#     --output_path /Rocket_ssd/dataset/data_litevloc/vpr_eval/robocar/s00001_hard/results_vpr
