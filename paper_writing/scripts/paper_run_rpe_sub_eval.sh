##### Submission
# Top_K = 2 5 8 11 14 17 20 30 40 50
# rosrun litevloc run_benchmark_rpe_submission.sh mapfree val;
# Top_K = 2 5 8 11 14 17 20 30
# rosrun litevloc run_benchmark_rpe_submission.sh hkustgz_campus test;
# Top_K = 2 5 8
# rosrun litevloc run_benchmark_rpe_submission.sh ucl_campus_aria test;
# Top_K = 2 5 8 11 14 17 20 30
# DEVICES=("device1" "device2" "device3" "device4")
DEVICES=("device3" "device4")
for device in "${DEVICES[@]}"; do
  rosrun litevloc run_benchmark_rpe_submission.sh 360loc_$device test;
done

##### Evaluation
# rosrun litevloc run_benchmark_rpe_evaluation.sh mapfree val > /Rocket_ssd/dataset/data_litevloc/map_free_eval/mapfree/map_free_eval/results_rpe/report_evaluation_100_10.txt;
# rosrun litevloc parse_report_table_rpe.py --txt_path /Rocket_ssd/dataset/data_litevloc/map_free_eval/mapfree/map_free_eval/results_rpe/report_evaluation_100_10.txt;

# rosrun litevloc run_benchmark_rpe_evaluation.sh hkustgz_campus test > /Rocket_ssd/dataset/data_litevloc/map_free_eval/hkustgz_campus/map_free_eval/results_rpe/report_evaluation_100_10.txt;
# rosrun litevloc parse_report_table_rpe.py --txt_path /Rocket_ssd/dataset/data_litevloc/map_free_eval/hkustgz_campus/map_free_eval/results_rpe/report_evaluation_100_10.txt;

# rosrun litevloc run_benchmark_rpe_evaluation.sh ucl_campus_aria test > /Rocket_ssd/dataset/data_litevloc/map_free_eval/ucl_campus_aria/map_free_eval/results_rpe/report_evaluation_100_10.txt;
# rosrun litevloc parse_report_table_rpe.py --txt_path /Rocket_ssd/dataset/data_litevloc/map_free_eval/ucl_campus_aria/map_free_eval/results_rpe/report_evaluation_100_10.txt;

for device in "${DEVICES[@]}"; do
  rosrun litevloc run_benchmark_rpe_evaluation.sh 360loc_$device test > /Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_$device/map_free_eval/results_rpe/report_evaluation_100_10.txt;
  rosrun litevloc parse_report_table_rpe.py --txt_path /Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_$device/map_free_eval/results_rpe/report_evaluation_100_10.txt;
  cp /Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_$device/map_free_eval/results_rpe/report_evaluation_100_10.csv \
     /Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_$device/map_free_eval/results_rpe/report_evaluation_all_metrics.csv
done

##### Visualization
# rosrun litevloc viz_rpe_results.py --csv_path /Rocket_ssd/dataset/data_litevloc/map_free_eval/mapfree/map_free_eval/results_rpe/report_evaluation_all_metrics.csv;
# rosrun litevloc viz_rpe_results.py --csv_path /Rocket_ssd/dataset/data_litevloc/map_free_eval/hkustgz_campus/map_free_eval/results_rpe/report_evaluation_all_metrics.csv;
# rosrun litevloc viz_rpe_results.py --csv_path /Rocket_ssd/dataset/data_litevloc/map_free_eval/ucl_campus_aria/map_free_eval/results_rpe/report_evaluation_all_metrics.csv;
# for device in "${DEVICES[@]}"; do
#   rosrun litevloc viz_rpe_results.py --csv_path /Rocket_ssd/dataset/data_litevloc/map_free_eval/360loc_$device/map_free_eval/results_rpe/report_evaluation_all_metrics.csv;
# done