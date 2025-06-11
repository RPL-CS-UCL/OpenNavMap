#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../python'))

import time
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from colorama import Fore, Back, Style

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import parser
from benchmark_vpr.dataloader import TestDataset
from utils.utils_vpr_method import initialize_vpr_model, initialize_match_model
from utils.utils_setting_color_font import acquire_color_palette, acquire_marker, setting_font
from utils.utils_geom import compute_pose_error, convert_vec_to_matrix, convert_matrix_to_vec

# Configure matplotlib
# setting_font()  # Assume this configures fonts as previously discussed
PALLETE = acquire_color_palette()  # Assume returns color list
MARKERS = acquire_marker()  # Assume returns marker list

def compute_diff_matrix(db_descs, query_descs, eps=1e-8):
    dots = np.dot(db_descs, query_descs.T)
    db_norms = np.linalg.norm(db_descs, axis=1)[:, None]
    q_norms = np.linalg.norm(query_descs, axis=1)[None, :]
    sims = dots / (db_norms * q_norms + eps)  # (n_db, n_query)
    D = 1.0 - sims

    return D

def convert_pose_to_2d(pose):
    """Convert 7D pose (x,y,z + quaternion) to 2D (x,z) coordinates"""
    Tc2w = convert_vec_to_matrix(pose[4:], pose[:4], 'wxyz')
    trans, quat = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    
    return (trans[0], trans[1])

def visualize_vpr_data(test_ds, D_all, valid_pairs, output_path):
    """Visualization function that creates the dual plot"""
    # D_all is (n_query, n_db)
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Difference Matrix Plot
    ax1 = fig.add_subplot(131)
    im = ax1.imshow(D_all, cmap='Greys', aspect='auto')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    im.set_clim(0.0, 1.0)
    ax1.set_xlabel('Query Index', fontsize=14)
    ax1.set_ylabel('Database Index', fontsize=14)
    ax1.set_title("Difference Matrix", fontsize=16)
    for q_idx, db_idx in valid_pairs:
        ax1.plot(q_idx, db_idx, 'r.', markersize=4, alpha=1.0, markeredgewidth=1)
    ax1.set_aspect('equal')

    # 2. Weight Matrix Plot
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(1.0 / (D_all + 1e-8), cmap='Greys', aspect='auto')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    im.set_clim(0.0, 5.0)
    ax2.set_xlabel('Query Index', fontsize=14)
    ax2.set_ylabel('Database Index', fontsize=14)
    ax2.set_title("Weight Matrix", fontsize=16)
    for q_idx, db_idx in valid_pairs:
        ax2.plot(q_idx, db_idx, 'r.', markersize=4, alpha=1.0, markeredgewidth=1)
    ax2.set_aspect('equal')

    # 3. Trajectory Plot
    ax3 = fig.add_subplot(133)
    
    db_poses = np.array([test_ds.database_poses[test_ds.database_image_names[i]] for i in range(test_ds.num_database)])
    db_xy = np.array([convert_pose_to_2d(p) for p in db_poses])
    ax3.plot(db_xy[:,0], db_xy[:,1], c=PALLETE[1], linewidth=1, label='Database')
    
    # Plot query trajectory
    query_poses = np.array([test_ds.queries_poses[test_ds.queries_image_names[i]] for i in range(test_ds.num_queries)])
    query_xy = np.array([convert_pose_to_2d(p) for p in query_poses])
    ax3.plot(query_xy[:,0], query_xy[:,1], c=PALLETE[2], linewidth=1, label='Query')
    
    # Plot connections for valid pairs
    for q_idx, db_idx in valid_pairs:
        ax3.plot([query_xy[q_idx,0], db_xy[db_idx,0]],
                 [query_xy[q_idx,1], db_xy[db_idx,1]],
                 'g-', linewidth=2)

    ax3.set_xlabel('X [m]', fontsize=14)
    ax3.set_ylabel('Y [m]', fontsize=14)
    ax3.set_title("Trajectories", fontsize=16)
    ax3.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.05, 1.0))
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def extract_descriptors(model, test_ds, args):
    """Extract database and query descriptors with timing"""
    all_descs = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
    
    with torch.no_grad():
        # Database descriptors
        db_subset = Subset(test_ds, list(range(test_ds.num_database)))
        db_loader = DataLoader(db_subset, batch_size=args.batch_size, num_workers=4)
        for images, indices, _ in db_loader:
            descs = model(images.to(args.device)).cpu().numpy()
            all_descs[indices.numpy()] = descs

        # Query descriptors
        q_subset = Subset(test_ds, list(range(test_ds.num_database, len(test_ds))))
        q_loader = DataLoader(q_subset, batch_size=1, num_workers=4)
        for images, indices, _ in q_loader:
            descs = model(images.to(args.device)).cpu().numpy()
            all_descs[indices.numpy()] = descs

    return all_descs[:test_ds.num_database], all_descs[test_ds.num_database:]

def find_valid_matches(test_ds, trans_thresh, rot_thresh):
    """Find ground-truth valid pairs using pose thresholds"""
    valid_pairs = []
    
    for q_idx in range(test_ds.num_queries):
        query_img_name = test_ds.queries_image_names[q_idx]
        query_pose = test_ds.queries_poses[query_img_name]
        
        min_dist = float('inf')
        best_db_idx = -1
        
        # Find nearest database image by position
        for db_idx in range(test_ds.num_database):
            db_img_name = test_ds.database_image_names[db_idx]
            db_pose = test_ds.database_poses[db_img_name]

            Tc2w = convert_vec_to_matrix(query_pose[4:], query_pose[:4], 'wxyz')
            trans_query, quat_query = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
            Tc2w = convert_vec_to_matrix(db_pose[4:], db_pose[:4], 'wxyz')
            trans_db, quat_db = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
            trans_err, rot_err = compute_pose_error((trans_query, quat_query), (trans_db, quat_db), mode='vector')
            
            if trans_err <= trans_thresh and rot_err <= rot_thresh:
                err = trans_err
                if err < min_dist:
                    min_dist = err
                    best_db_idx = db_idx

        if best_db_idx != -1:
            valid_pairs.append((q_idx, best_db_idx))
            
    return valid_pairs

def evaluate_vpr_system(args):
    """Main evaluation and visualization routine"""
    for query_folder in args.queries_folder:
        # Load dataset
        test_ds = TestDataset(args.database_folder, query_folder, args.image_size)
        
        # Initialize models
        vpr_model = initialize_vpr_model(args.vpr_model, args.backbone, args.descriptors_dimension, args.device)
        
        # Extract descriptors
        db_descs, query_descs = extract_descriptors(vpr_model, test_ds, args)
        
        # Compute distance matrix
        D_all = compute_diff_matrix(db_descs, query_descs)
        
        # Find ground-truth valid matches
        valid_pairs = find_valid_matches(test_ds, args.trans_thresh, args.rot_thresh)
        
        # Create visualization
        output_dir = Path(args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        visualize_vpr_data(test_ds, D_all, valid_pairs, output_dir/f"vpr_data_{query_folder.name}.jpg")

        np.save(output_dir/f"D_all_{query_folder.name}.npy", D_all)

def parse_arguments():
    parser = argparse.ArgumentParser(description='VPR Visualization System')
    
    # Dataset parameters
    parser.add_argument('--database_folder', type=Path, required=True,
                      help='Path to database folder')
    parser.add_argument('--queries_folder', type=Path, required=True, nargs='+',
                      help='Path to query folder')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Input image size')
    
    # Model parameters
    parser.add_argument('--vpr_model', type=str, default='cosplace',
                      choices=['NetVLAD', 'CosPlace'], help='VPR architecture')
    parser.add_argument('--backbone', type=str, default='ResNet18',
                      help='Feature extractor backbone')
    parser.add_argument('--descriptors_dimension', type=int, default=256,
                      help='Descriptor dimension')
    
    # Evaluation parameters
    parser.add_argument('--trans_thresh', type=float, default=7.5,
                      help='Translation threshold (meters)')
    parser.add_argument('--rot_thresh', type=float, default=75.0,
                      help='Rotation threshold (degrees)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for descriptor extraction')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cpu', 'cuda'], help='Compute device')
    parser.add_argument('--output_path', type=Path, default='results',
                      help='Output directory for visualizations')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    evaluate_vpr_system(args)
