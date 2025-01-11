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

def calculate_recalls(predictions, test_ds, args):
    """Calculate and log recall values."""
    if args.use_labels:
        positives_per_query = test_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Divide by num_queries and multiply by 100, so the recalls are in percentages
        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join(
            [f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)]
        )
        logging.info(recalls_str)

def compute_vpr_metrics(dataset_path, query_name, database_name, results_vpr, 
                        tsl_thre, ang_thre):
    poses_query = read_poses(
        os.path.join(dataset_path, 'query', 'out_map_' + query_name, 'poses_abs_gt.txt')
    )
    poses_database = read_poses(
        os.path.join(dataset_path, 'database', 'out_map_' + database_name, 'poses_abs_gt.txt')
    )

    # Compute the number of positive
    y_true, confidence_scores = [], []
    for indice, result in enumerate(results_vpr):
        query_name, score = result[0], result[2]
        confidence_scores.append(score)

        pose_query = poses_query[query_name]
        for k, pose_db in poses_database.items():
            Tc2w = convert_vec_to_matrix(pose_query[4:], pose_query[:4], 'wxyz')
            trans_query, quat_query = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
            Tc2w = convert_vec_to_matrix(pose_db[4:], pose_db[:4], 'wxyz')
            trans_db, quat_db = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
            
            dis_trans, dis_angle = compute_relative_dis(trans_query, quat_query, trans_db, quat_db, 'xyzw')
            flag_same_place = (dis_trans < tsl_thre and dis_angle < ang_thre)
            if flag_same_place:
                y_true.append(1)
                break
        if indice == len(y_true):
            y_true.append(0)

    y_true = np.array(y_true)
    confidence_scores = np.array(confidence_scores)

    # Compute the precision and recall
    tp, fp = 0, 0
    for result in results_vpr:
        query_name, database_name, score = result[0], result[1], result[2]
        pose_query, pose_database = poses_query[query_name], poses_database[database_name]

        Tc2w = convert_vec_to_matrix(pose_query[4:], pose_query[:4], 'wxyz')
        trans_query, quat_query = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
        Tc2w = convert_vec_to_matrix(pose_db[4:], pose_db[:4], 'wxyz')
        trans_db, quat_db = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
        
        dis_trans, dis_angle = compute_relative_dis(trans_query, quat_query, trans_db, quat_db, 'xyzw')
        flag_same_place = (dis_trans < tsl_thre and dis_angle < ang_thre)
        if flag_same_place:
            tp += 1
        else:
            fp += 1

    output_metrics = dict()
    if tp + fp < 1:
        output_metrics['Precision'] = 0
    else:
        output_metrics['Precision'] = tp / (tp + fp)
    
    num_pos_sample = np.sum(y_true)
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
    for f in os.listdir(args.result_dir):
        if 'submission-' in f:
            f_new = f.replace('.txt', '')
            query_name, database_name = f_new.split('-')[1], f_new.split('-')[2]
            results_vpr = np.loadtxt(os.path.join(args.result_dir, f), dtype=object)
            metrics, curves_data = compute_vpr_metrics(
                args.dataset_path, query_name, database_name, results_vpr,
                args.tsl_thre, args.ang_thre
            )
            # all_results[f"{query_name}-{database_name}"] = metrics
            print(metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('eval', description='Evaluate submissions for the VPR dataset benchmark')
    parser.add_argument('--result_dir', type=Path, default='',
                        help='Path to the submission files')
    parser.add_argument('--log', choices=('warning', 'info', 'error'),
                        default='warning', help='Logging level. Default: warning')
    parser.add_argument('--dataset_path', type=Path, default=None,
                        help='Path to the dataset folder')
    parser.add_argument('--tsl_thre', type=float, default=10.0, 
                        help='Threshold (meters) to consider two poses as the same place.')
    parser.add_argument('--ang_thre', type=float, default=75.0, 
                        help='Threshold (degree) to consider two poses as the same place.')
                        
    args = parser.parse_args()      

    logging.basicConfig(level=args.log.upper())
    main(args)
    # try:
    #     main(args)
    # except Exception:
    #     logging.error("Unexpected behaviour. Exiting.")
