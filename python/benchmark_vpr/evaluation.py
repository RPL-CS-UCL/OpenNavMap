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
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score

from utils.utils_vpr_method import save_prec_recall_curve
from utils.utils_geom import read_poses
from utils.utils_geom import compute_pose_error, convert_vec_to_matrix, convert_matrix_to_vec

def is_same_place(poseA, poseB, trans_threshold, ori_threshold):
    Tc2w = convert_vec_to_matrix(poseA[4:], poseA[:4], 'wxyz')
    transA, quatA = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    Tc2w = convert_vec_to_matrix(poseB[4:], poseB[:4], 'wxyz')
    transB, quatB = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    dis_trans, dis_angle = compute_pose_error((transA, quatA), (transB, quatB), mode='vector')
    return (dis_trans < trans_threshold and dis_angle < ori_threshold) 

def compute_max_recall(prec_values, recall_values):
    max_recall = 0
    for i in range(len(prec_values)):
        if prec_values[i] >= 1 - 1e-4:
            max_recall = max(max_recall, recall_values[i])
    
    return max_recall

def compute_metrics(dataset_path, results_vpr, 
                    query_name, database_name, 
                    trans_threshold, ori_threshold):

    poses_query = read_poses(os.path.join(dataset_path, 'query', 'out_map_' + query_name, 'poses_abs_gt.txt'))
    poses_db = read_poses(os.path.join(dataset_path, 'database', 'out_map_' + database_name, 'poses_abs_gt.txt'))
    assert len(poses_query) > 0, "No query poses found"
    assert len(poses_db) > 0, "No database poses found"

    """
    Label Definitions:
        - y_true = 1: Query image has a valid match (corresponding database image exists).
        - y_true = 0: Query image has no valid match (no corresponding database image exists).
    Prediction Logic:
        - If the query image has a valid match (y_true=1):
            - Correct Top-1 match -> True Positive (TP) (y_pred=1)
            - Incorrect Top-1 match or No Top-1 match -> False Negative (FN) (y_pred=0)
        - If the query image has no valid match (y_true=0):
            - Any Top-1 retrieval -> False Positive (FP) (y_pred=1)
            - No Top-1 retrieval  -> True Negative (TN) (y_pred=0)
    """
    # Generate binary labels
    y_true = [0] * len(results_vpr)
    for ind, result in enumerate(results_vpr):
        query_name = result[0]
        pose_query = poses_query[query_name]
        for pose_db in poses_db.values():
            same_place = is_same_place(pose_query, pose_db, trans_threshold, ori_threshold)
            if same_place:
                y_true[ind] = 1

    logging.info(f"Number Valid Match of Queries: {np.sum(y_true)}")

    # Generate binary predictions
    y_pred, y_score = [], []
    for ind, result in enumerate(results_vpr):
        query_name, database_name, score, acc_flag = result[0], result[1], float(result[2]), int(result[3])
        pose_query, pose_db = poses_query[query_name], poses_db[database_name]
        if y_true[ind]:
            # Any retrieval
            if acc_flag > 0:
                same_flag = is_same_place(pose_query, pose_db, trans_threshold, ori_threshold)
                y_pred.append(int(same_flag))
            # No retrieval
            else:
                y_pred.append(0)
        else:
            # Any retrieval
            if acc_flag > 0:
                y_pred.append(1)
            # No retrieval
            else:
                y_pred.append(0)

        y_score.append(score)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_precision = average_precision_score(y_true, y_score)

    # Compute curve data
    prec_values, recall_values, thres = precision_recall_curve(y_true, y_score)
    max_recall = compute_max_recall(prec_values, recall_values)
    
    # Store metrics
    output_metrics = dict()
    output_metrics['Accuracy'] = accuracy
    output_metrics['Precision'] = precision
    output_metrics['Recall'] = recall
    output_metrics['F1 Score'] = f1
    output_metrics['Valid Match Number'] = int(np.sum(y_true))
    output_metrics['Query Number'] = int(len(y_true))
    output_metrics['Average Precision'] = avg_precision
    output_metrics['Maximum Recall'] = max_recall

    curves_data = dict()
    curves_data['Precision Values'] = prec_values 
    curves_data['Recall Values'] = recall_values
    curves_data['PR Thresholds'] = thres.tolist()
    curves_data['Maximum Recall'] = max_recall

    return output_metrics, curves_data, y_true, y_pred, y_score

def eval(args):
    output_metrics_methods, curve_metrics_methods = dict(), dict()
    for method in args.methods:
        logging.warning(f"Evaluating Method: {method}")
        # Metric
        output_querydb_metrics, curve_querydb_metrics = dict(), dict()
        
        # Result_path + Method
        result_dir = os.path.join(args.result_dir, method)
        y_true_all, y_pred_all, y_score_all = [], [], []

        # Traverse results for each database-query pair
        for f in sorted(os.listdir(result_dir)):
            if 'submission-' in f:
                f_new = f.replace('.txt', '')
                query_name, database_name = f_new.split('-')[1], f_new.split('-')[2]
                logging.warning(f"Evaluating Results of Query: {query_name} Database: {database_name}")

                results_vpr = np.loadtxt(os.path.join(result_dir, f), dtype=object)           
                output_metrics, curves_data, y_true, y_pred, y_score = compute_metrics(
                    args.dataset_path, results_vpr,
                    query_name, database_name,
                    args.trans_threshold, args.ori_threshold
                )

                querydb_name = f"{query_name}-{database_name}"
                output_querydb_metrics[querydb_name] = output_metrics
                curve_querydb_metrics[querydb_name] = curves_data
                y_true_all  += y_true
                y_pred_all  += y_pred
                y_score_all += y_score

        # Compute mean metrics of all samples
        accuracy = accuracy_score(y_true_all, y_pred_all)
        precision = precision_score(y_true_all, y_pred_all, zero_division=0)
        recall = recall_score(y_true_all, y_pred_all, zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
        avg_precision = average_precision_score(y_true_all, y_score_all)    

        # Compute PR curve
        prec_values, recall_values, thres = precision_recall_curve(y_true_all, y_score_all)
        max_recall = compute_max_recall(prec_values, recall_values)

        # Store metrics
        output_querydb_metrics['Accuracy'] = accuracy
        output_querydb_metrics['Precision'] = precision
        output_querydb_metrics['Recall'] = recall
        output_querydb_metrics['F1 Score'] = f1
        output_querydb_metrics['Total Valid Match Number'] = int(np.sum(y_true_all))
        output_querydb_metrics['Total Query Number'] = int(len(y_true_all))
        output_querydb_metrics['Average Precision'] = avg_precision
        output_querydb_metrics['Maximum Recall'] = max_recall
        output_metrics_methods[method] = output_querydb_metrics

        curve_querydb_metrics['Precision Values'] = prec_values
        curve_querydb_metrics['Recall Values'] = recall_values
        curve_querydb_metrics['Average Precision'] = avg_precision
        curve_querydb_metrics['PR Thresholds'] = thres.tolist()
        curve_querydb_metrics['Maximum Recall'] = max_recall
        curve_metrics_methods[method] = curve_querydb_metrics

        # Save metrics as a json file
        output_json = json.dumps(output_querydb_metrics, indent=2)
        with open(os.path.join(result_dir, 'report_evaluation.json'), 'w') as f:
            f.write(output_json)

        # Save precision-curve for the method
        print(f'Saving PR Curve to {os.path.join(args.result_dir, method)}')
        # save_prec_recall_curve(os.path.join(args.result_dir, method), {method: curve_querydb_metrics})

    # Draw the PR curve of all methods
    save_prec_recall_curve(args.result_dir, curve_metrics_methods)

def summ(args):
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
        for k in method_metrics.keys():
            if '-' in k:
                all_querydbs.add(k)
    all_querydbs = sorted(all_querydbs)

    # Sort method names
    sorted_methods = sorted(result_method.keys())
    csv_lines = []

    # Generate CSV content
    for querydb in all_querydbs:
        json_file = os.path.join(args.result_dir, f"runtime_results-{querydb}.json")
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        csv_lines.append(f"{querydb}")
        csv_lines.append("Method,Accuracy,Precision,Recall,F1 Score,Max Recall," + \
                         "Valid Match Number," + \
                         "Total Runtime [ms],Query Number")
        
        for method in sorted_methods:
            metrics = result_method.get(method, {}).get(querydb, {})

            accuracy = metrics.get('Accuracy', 0)
            precision = metrics.get('Precision', 0)
            recall = metrics.get('Recall', 0)
            f1 = metrics.get('F1 Score', 0)
            max_recall = metrics.get('Maximum Recall', 0)
            num_valid_match = metrics.get('Valid Match Number', 0)
            if method in json_data:
                total_runtime = json_data[method]['Total Runtime [s]'] * 1000
                num_query = json_data[method]['Query Number']
            else:
                total_runtime = float('nan')
                num_query = float('nan')
                
            csv_lines.append(f"{method},{accuracy:.3f},{precision:.3f},{recall:.3f},{f1:.3f},{max_recall:.1f}," + \
                             f"{num_valid_match}," + \
                             f"{total_runtime:.1f},{num_query}")
        
        # Add empty line between sections
        csv_lines.append("")

    # Output Mean Results
    csv_lines.append("Mean Results")
    csv_lines.append("Method,Accuracy,Precision,Recall,F1 Score,Average Precision,Max Recall," + 
                     "Total Valid Match Number,Total Query Number")
    for method in sorted_methods:
        metrics = result_method.get(method, {})
        accuracy = metrics.get('Accuracy')
        precision = metrics.get('Precision', 0)
        recall = metrics.get('Recall', 0)
        f1 = metrics.get('F1 Score', 0)
        avg_precision = metrics.get('Average Precision', 0)
        max_recall = metrics.get('Maximum Recall', 0)
        total_valid = metrics.get('Total Valid Match Number', 0)
        total_query = metrics.get('Total Query Number', 0)   
        csv_lines.append(f"{method},{accuracy:.3f},{precision:.3f},{recall:.3f},{f1:.3f}," + \
                         f"{avg_precision:.3f}," + f"{max_recall:.1f}," \
                         f"{total_valid},{total_query}") 

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
    parser.add_argument('--methods', type=str, nargs='+', help="Different VPR methods")
    parser.add_argument('--dataset_path', type=Path, default=None,
                        help='Path to the dataset folder')
    parser.add_argument('--trans_threshold', type=float, default=7.5, 
                        help='Threshold (meters) to consider two poses as the same place.')
    parser.add_argument('--ori_threshold', type=float, default=75.0, 
                        help='Threshold (degree) to consider two poses as the same place.')
    parser.add_argument('--log', choices=('warning', 'info', 'error'),
                        default='warning', help='Logging level. Default: warning')
    parser.add_argument('--option', choices=('eval', 'summ'), 
                        default='eval', help='Running option. Default: eval')
    
    args = parser.parse_args()
    logging.basicConfig(level=args.log.upper())

    print(args.methods)
    main(args)
