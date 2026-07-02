##### KF Selection
# rosrun litevloc run_benchmark_kf_selection.sh mapfree;
# rosrun litevloc run_benchmark_kf_selection.sh hkustgz_campus;

##### Submission
rosrun litevloc run_benchmark_kf_submission.sh mapfree;
# rosrun litevloc run_benchmark_kf_submission.sh hkustgz_campus;

##### Evaluation
# rosrun litevloc run_benchmark_kf_evaluation.sh mapfree > /Rocket_ssd/dataset/data_litevloc/map_free_eval/mapfree/map_free_eval/results_kf/report_evaluation_025_5.txt
# rosrun litevloc parse_report_table_mf.py --path_report_eval /Rocket_ssd/dataset/data_litevloc/map_free_eval/mapfree/map_free_eval/results_kf/report_evaluation_025_5.txt