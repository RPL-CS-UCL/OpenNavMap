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

def is_same_place(poseA, poseB, tsl_thre, ang_thre):
    Tc2w = convert_vec_to_matrix(poseA[4:], poseA[:4], 'wxyz')
    transA, quatA = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    Tc2w = convert_vec_to_matrix(poseB[4:], poseB[:4], 'wxyz')
    transB, quatB = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    dis_tsl, dis_angle = compute_relative_dis(transA, quatA, transB, quatB)			
    return (dis_tsl < tsl_thre and dis_angle < ang_thre) 

def compute_vpr_metrics(dataset_path, query_name, database_name, results_vpr, 
                        tsl_thre, ang_thre):
    poses_query = read_poses(os.path.join(dataset_path, 'query', 'out_map_' + query_name, 'poses_abs_gt.txt'))
    poses_db = read_poses(os.path.join(dataset_path, 'database', 'out_map_' + database_name, 'poses_abs_gt.txt'))
    
    assert len(poses_query) > 0, "No query poses found"
    assert len(poses_db) > 0, "No database poses found"

    # Compute the number of positive sample
    y_true = [any(is_same_place(pose_query, pose_db, tsl_thre, ang_thre)
              for pose_db in poses_db.values()) 
              for pose_query in poses_query.values()]
    y_true = [int(value) for value in y_true]
    print(y_true)

    total_samples = int(np.sum(y_true))
    logging.info(f"Number of query as valid PR: {total_samples}")

    # Compute the precision and recall
    y_score = []
    tp, fp = 0, 0

    for result in results_vpr:
        query_name, database_name, score, acc_flag = result[0], result[1], float(result[2]), int(result[3])
        pose_query, pose_db = poses_query[query_name], poses_db[database_name]
        same_flag = is_same_place(pose_query, pose_db, tsl_thre, ang_thre)
        y_score.append(score)

        # High confidence
        if acc_flag > 0:
            # Correct Loop detection with high confidence for acceptance
            if same_flag:
                tp += 1
            # Wrong loop detection with high confidence
            else:
                fp += 1

    if tp + fp < 1:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if total_samples < 1:
        recall = 0
    else:
        recall = tp / total_samples

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

    # Compute Curve data
    prec_values, recall_values, thres = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    curves_data = dict()
    curves_data['Precision Values'], curves_data['Recall Values'] = prec_values, recall_values
    curves_data['Average Precision'] = avg_precision
    curves_data['PR Thresholds'] = thres.tolist()

    output_metrics = dict()
    output_metrics['Positive Sample Number'] = total_samples
    output_metrics['Precision'] = precision
    output_metrics['Recall'] = recall
    output_metrics['F1 Score'] = f1_score
    output_metrics['Average Precision'] = avg_precision
    return output_metrics, curves_data

def plot_prec_recall_curve(precision_curve, recall_curve, average_precision=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, 
             color='b', lw=2, 
             label=f'Precision-Recall curve')
    if average_precision is not None:
        plt.plot([], [], ' ', 
                 label=f'Average Precision = {average_precision:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.grid(True, alpha=0.3)
    plt.show()

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
            logging.warning(f"Evaluating Results of Query: {query_name} Database: {database_name}")

            plot_prec_recall_curve(curves_data['Precision Values'], curves_data['Recall Values'], output_metrics['Average Precision'])

    output_json = json.dumps(output_querydb_metrics, indent=2)
    with open(os.path.join(args.result_dir, 'report_evaluation.json'), 'w') as f:
        f.write(output_json)

def summ(args):
    ##### Parse results
    """ Example Output
    query-database
    Precision, Cecall, Positive Sample Number, Total Runtime [ms], Query Number
    method 1, xx, xx, ...
    method 2, xx, xx, ...
    method 3, xx, xx, ...
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
        csv_lines.append("Method,Precision,Recall,F1 Score,Average Precision,Positive Sample Number,Total Runtime [ms],Query Number")

        json_file = os.path.join(args.result_dir, f"runtime_results-{querydb}.json")
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Add each method's metrics
        for method in sorted_methods:
            metrics = result_method.get(method, {}).get(querydb, {})
            precision = metrics.get('Precision', 0)
            recall = metrics.get('Recall', 0)
            f1_score = metrics.get('F1 Score', 0)
            avg_prec = metrics.get('Average_Precision', 0)
            num_pos_sample = metrics.get('Positive Sample Number', 0)
            if method in json_data:
                total_runtime = json_data[method]['Total Runtime [s]'] * 1000
                num_query = json_data[method]['Query Number']
            else:
                total_runtime = float('nan')
                num_query = float('nan')
            csv_lines.append(f"{method},{precision:.3f},{recall:.3f},{f1_score:.3f},{avg_prec:.3f}," + 
                             f"{num_pos_sample},{total_runtime:.1f},{num_query}")
        
        # Add empty line between sections
        csv_lines.append("")

    # Remove the last empty line to avoid trailing newline
    if csv_lines and csv_lines[-1] == "":
        csv_lines.pop()

    csv_output = '\n'.join(csv_lines)
    with open(os.path.join(args.result_dir, 'report_evaluation.csv'), 'w') as f:
        f.write(csv_output)
    
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
