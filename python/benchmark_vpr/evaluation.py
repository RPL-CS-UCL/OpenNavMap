"""
Usage: python python/benchmark_vpr/evaluation.py \
--result_dir /Rocket_ssd/dataset/data_litevloc/vpr_eval/ucl_campus/s00000/results_vpr/cosplace_sequence_match \
--dataset_path /Rocket_ssd/dataset/data_litevloc/vpr_eval/ucl_campus/s00000
"""

#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import glob
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd

from utils.utils import *

def is_same_place(quatA, transA, quatB, transB, tsl_thre, ang_thre):
    dis_tsl, dis_angle = compute_relative_dis(transA, quatA, transB, quatB)			
    return (dis_tsl < tsl_thre and dis_angle < ang_thre) 

def compute_vpr_metrics(dataset_path, query_name, database_name, results_vpr, 
                        tsl_thre, ang_thre):
    poses_query = read_poses(os.path.join(dataset_path, 'query', 'out_map_' + query_name, 'poses_abs_gt.txt'))
    poses_db = read_poses(os.path.join(dataset_path, 'database', 'out_map_' + database_name, 'poses_abs_gt.txt'))

    # Compute the number of positive sample
    num_pos_sample = 0
    for _, pose_query in poses_query.items():
        for _, pose_db in poses_db.items():
            Tc2w = convert_vec_to_matrix(pose_query[4:], pose_query[:4], 'wxyz')
            trans_query, quat_query = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
            Tc2w = convert_vec_to_matrix(pose_db[4:], pose_db[:4], 'wxyz')
            trans_db, quat_db = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
            if is_same_place(quat_query, trans_query, quat_db, trans_db, tsl_thre, ang_thre):
                num_pos_sample += 1
                break
    logging.info(f"Number of query as valid PR: {num_pos_sample}")

    # Compute the precision and recall
    tp, fp, tn = 0, 0, 0
    confidence_scores = []
    for result in results_vpr:
        query_name, database_name, score = result[0], result[1], float(result[2])
        pose_query, pose_db = poses_query[query_name], poses_db[database_name]
        confidence_scores.append(score)
        Tc2w = convert_vec_to_matrix(pose_query[4:], pose_query[:4], 'wxyz')
        trans_query, quat_query = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
        Tc2w = convert_vec_to_matrix(pose_db[4:], pose_db[:4], 'wxyz')
        trans_db, quat_db = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
        same = is_same_place(quat_query, trans_query, quat_db, trans_db, tsl_thre, ang_thre)
        # Correct Loop detection with high confidence for acceptance
        if same and score >= 1e-3:
            tp += 1
        # Wrong Loop detection but with zero confidence for rejection
        if not same and score <= 1e-3:
            tn += 1
        # Wrong loop detection with high confidence
        elif not same:
            fp += 1
    confidence_scores = np.array(confidence_scores)

    output_metrics = dict()
    if tp + fp < 1:
        output_metrics['Precision'] = 0
    else:
        output_metrics['Precision'] = tp / (tp + fp)
    
    if num_pos_sample < 1:
        output_metrics['Recall'] = 0
    else:
        output_metrics['Recall'] = tp / num_pos_sample

    # compute Curve data
    # prec_values, recall_values, thres = precision_recall_curve(y_true, confidence_scores)
    # average_precision = average_precision_score(ground_truth, confidence_scores)
    curve_data = dict()
    # curve_data['prec_values'], curve_data['recall_values'] = prec_values, recall_values

    return output_metrics, curve_data

def eval(args):
    output_querydb_metrics = dict()
    for f in sorted(os.listdir(args.result_dir)):
        if 'submission-' in f:
            f_new = f.replace('.txt', '')
            query_name, database_name = f_new.split('-')[1], f_new.split('-')[2]

            results_vpr = np.loadtxt(os.path.join(args.result_dir, f), dtype=object)           
            output_metrics, curves_data = compute_vpr_metrics(
                args.dataset_path, query_name, database_name, results_vpr,
                args.tsl_thre, args.ang_thre
            )
            querydb_name = f"{query_name}-{database_name}"
            output_querydb_metrics[querydb_name] = output_metrics
            logging.warning(f"Evaluating Results of {querydb_name}")

    output_json = json.dumps(output_querydb_metrics, indent=2)
    with open(os.path.join(args.result_dir, 'report_evaluation.json'), 'w') as f:
        f.write(output_json)

def summ(args):
    ##### Parse results
    """ Example Output
    query-database
    Precision,Cecall
    method 1,xx,xx
    method 2,xx,xx
    method 3,xx,xx
    """    
    result_method = {}
    for method_name in sorted(os.listdir(args.result_dir)):
        method_path = os.path.join(args.result_dir, method_name)
        if os.path.isdir(method_path):
            json_file = os.path.join(method_path, 'report_evaluation.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    result_method[method_name] = json.load(f)

    # Collect all unique query databases and sort them
    all_querydbs = set()
    for method_metrics in result_method.values():
        all_querydbs.update(method_metrics.keys())
    all_querydbs = sorted(all_querydbs)

    # Sort method names
    sorted_methods = sorted(result_method.keys())
    csv_lines = []

    # Generate CSV content
    for querydb in all_querydbs:
        # Add section header for query database
        csv_lines.append(f"{querydb}")
        csv_lines.append("Method,Precision,Recall")
        
        # Add each method's metrics
        for method in sorted_methods:
            metrics = result_method.get(method, {}).get(querydb, {})
            precision = metrics.get('Precision', 0)
            recall = metrics.get('Recall', 0)
            csv_lines.append(f"{method},{precision:.3f},{recall:.3f}")
        
        # Add empty line between sections
        csv_lines.append("")

    # Remove the last empty line to avoid trailing newline
    if csv_lines and csv_lines[-1] == "":
        csv_lines.pop()

    csv_output = '\n'.join(csv_lines)
    with open(os.path.join(args.result_dir, 'report_evaluation.csv'), 'w') as f:
        f.write(csv_output)

    ##### Parse running_time
    for querydb in all_querydbs:
        path_report_runtime = os.path.join(args.result_dir, f"{querydb}-runtime_results.txt")
        with open(path_report_runtime, 'r') as file:
            lines = file.readlines()

        data = dict()
        for line in lines:
            method, runtime = line.split(': ')
            method = method.replace('(vpr_model + vpr_match_model + image_match_model) ', '')
            data[method] = float(runtime.strip()[:-1])  # Remove the 's' from the end and convert to float

        for key in data.keys():
            data[key] *= 1000
            data[key] = '{:.0f}'.format(data[key])

        df = pd.DataFrame(list(data.items()), columns=['Method', 'Runtime [ms]'])
        path_report_eval_csv = path_report_runtime.replace('.txt', '.csv')
        df.to_csv(path_report_eval_csv)
    
def main(args):
    if not os.path.exists(args.result_dir):
        return
    
    if args.option == 'eval':
        eval(args)
    elif args.option == 'summ':
        summ(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('eval', description='Evaluate submissions for the VPR dataset benchmark')
    parser.add_argument('--result_dir', type=Path, default='',
                        help='Path to the submission files')
    parser.add_argument('--dataset_path', type=Path, default=None,
                        help='Path to the dataset folder')
    parser.add_argument('--tsl_thre', type=float, default=7.5, 
                        help='Threshold (meters) to consider two poses as the same place.')
    parser.add_argument('--ang_thre', type=float, default=75.0, 
                        help='Threshold (degree) to consider two poses as the same place.')
    parser.add_argument('--log', choices=('warning', 'info', 'error'),
                        default='warning', help='Logging level. Default: warning')
    parser.add_argument('--option', choices=('eval', 'summ'), 
                        default='eval', help='Running option. Default: eval')

    args = parser.parse_args()      

    logging.basicConfig(level=args.log.upper())
    main(args)
