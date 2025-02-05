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
import argparse
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score

from utils.utils import *

def is_same_place(quatA, transA, quatB, transB, tsl_threshold, ang_threshold):
    dis_tsl, dis_angle = compute_relative_dis(transA, quatA, transB, quatB)			
    return (dis_tsl < tsl_threshold and dis_angle < ang_threshold) 

def compute_vpr_metrics(dataset_path, query_name, database_name, results_vpr, 
                        tsl_thre, ang_thre):
    poses_query = read_poses(
        os.path.join(dataset_path, 'query', 'out_map_' + query_seq, 'poses_abs_gt.txt')
    )
    poses_database = read_poses(
        os.path.join(dataset_path, 'database', 'out_map_' + database_seq, 'poses_abs_gt.txt')
    )

    # Compute the number of positive sample
    num_pos_sample = 0
    for _, pose_query in poses_query.items():
        for _, pose_db in poses_database.items():
            Tc2w = convert_vec_to_matrix(pose_query[4:], pose_query[:4], 'wxyz')
            trans_query, quat_query = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
            Tc2w = convert_vec_to_matrix(pose_db[4:], pose_db[:4], 'wxyz')
            trans_db, quat_db = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
            if is_same_place(quat_query, trans_query, quat_db, trans_db, tsl_thre, ang_thre):
                num_pos_sample += 1
                break
    print(f"Number of query as valid PR: {num_pos_sample}")

    # Compute the precision and recall
    tp, fp = 0, 0
    confidence_scores = []
    for result in results_vpr:
        query_name, database_name, score = result[0], result[1], result[2]
        pose_query, pose_db = poses_query[query_name], poses_database[database_name]
        confidence_scores.append(score)
        Tc2w = convert_vec_to_matrix(pose_query[4:], pose_query[:4], 'wxyz')
        trans_query, quat_query = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
        Tc2w = convert_vec_to_matrix(pose_db[4:], pose_db[:4], 'wxyz')
        trans_db, quat_db = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
        same = is_same_place(quat_query, trans_query, quat_db, trans_db, tsl_thre, ang_thre)
        if same:
            tp += 1
        # Loop detection but with zero confidence for rejection
        if not flag_same_place and score <= 1e-3:
            tn += 1
        # Wrong loop detection with high confidence
        elif not flag_same_place:
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

def main(args):
    all_results = dict()
    for f in sorted(os.listdir(args.result_dir)):
        if 'submission-' in f:
            f_new = f.replace('.txt', '')
            query_name, database_name = f_new.split('-')[1], f_new.split('-')[2]
            print(f"Query: {query_name}, Database: {database_name}")

            results_vpr = np.loadtxt(os.path.join(args.result_dir, f), dtype=object)           
            metrics, curves_data = compute_vpr_metrics(
                args.dataset_path, query_seq, database_seq, results_vpr,
                args.tsl_thre, args.ang_thre
            )
            all_results[f"{query_name}-{database_name}"] = metrics
            print(metrics)
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('eval', description='Evaluate submissions for the VPR dataset benchmark')
    parser.add_argument('--result_dir', type=Path, default='',
                        help='Path to the submission files')
    parser.add_argument('--log', choices=('warning', 'info', 'error'),
                        default='warning', help='Logging level. Default: warning')
    parser.add_argument('--dataset_path', type=Path, default=None,
                        help='Path to the dataset folder')
    parser.add_argument('--tsl_thre', type=float, default=7.5, 
                        help='Threshold (meters) to consider two poses as the same place.')
    parser.add_argument('--ang_thre', type=float, default=75.0, 
                        help='Threshold (degree) to consider two poses as the same place.')
                        
    args = parser.parse_args()      

    logging.basicConfig(level=args.log.upper())
    main(args)
