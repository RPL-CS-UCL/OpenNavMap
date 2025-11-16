#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

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
from PIL import Image

from python.benchmark_vpr.dataloader import TestDataset
from python.utils.utils_vpr_method import initialize_vpr_model, initialize_match_model
from python.utils.utils_setting_color_font import acquire_color_palette, acquire_marker, setting_font, acquire_linestyle
from python.utils.utils_geom import compute_pose_error, convert_vec_to_matrix, convert_matrix_to_vec

# Configure matplotlib
setting_font(fontsize=14, titlesize=14, legend_fontsize=14)
PALLETE = acquire_color_palette()
MARKERS = acquire_marker()
LINESTYLE = acquire_linestyle()

def parse_arguments():
    parser = argparse.ArgumentParser(description='VPR Visualization System')
    parser.add_argument('--dataset_name', type=str, default='ucl_campus', choices=['ucl_campus', 'robocar', 'fusionportable'])
    parser.add_argument('--database_folder', type=Path, required=True)
    parser.add_argument('--queries_folder', type=Path, required=True, nargs='+')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--vpr_model', type=str, default='cosplace', choices=['NetVLAD', 'CosPlace'])
    parser.add_argument('--backbone', type=str, default='ResNet18')
    parser.add_argument('--descriptors_dimension', type=int, default=256)
    parser.add_argument('--trans_thresh', type=float, default=7.5)
    parser.add_argument('--rot_thresh', type=float, default=75.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--output_path', type=Path, default='results')
    return parser.parse_args()   

def compute_diff_matrix(db_descs, query_descs, eps=1e-8):
    dots = np.dot(db_descs, query_descs.T)
    db_norms = np.linalg.norm(db_descs, axis=1)[:, None]
    q_norms = np.linalg.norm(query_descs, axis=1)[None, :]
    D = 1.0 - dots / (db_norms * q_norms + eps)  # (n_db, n_query)

    return 1.0 - np.dot(db_descs, query_descs.T)

def convert_pose_to_2d(pose):
    """Convert 7D pose (x,y,z + quaternion) to 2D (x,z) coordinates"""
    Tc2w = convert_vec_to_matrix(pose[4:], pose[:4], 'wxyz')
    trans, quat = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    
    return (trans[0], trans[1])

def visualize_vpr_data(db_poses, query_poses, D_all, valid_pairs, output_path):
    """Visualization function that creates the dual plot"""
    # D_all is (n_query, n_db)
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Difference Matrix Plot
    ax1 = fig.add_subplot(131)
    im = ax1.imshow(D_all, cmap='Greys', aspect='auto')
    for q_idx, db_idx in valid_pairs:
        ax1.plot(q_idx, db_idx, 'r.', markersize=4, alpha=1.0, markeredgewidth=1)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    im.set_clim(0.0, 1.0)
    ax1.set_xlabel('Query Index', fontsize=14)
    ax1.set_ylabel('Reference Index', fontsize=14)
    ax1.set_title("Difference Matrix", fontsize=16)
    ax1.set_aspect('equal')

    # 2. Weight Matrix Plot
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(1.0 / (D_all + 1e-8), cmap='Greys', aspect='auto')
    for q_idx, db_idx in valid_pairs:
        ax2.plot(q_idx, db_idx, 'r.', markersize=4, alpha=1.0, markeredgewidth=1)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    im.set_clim(0.0, 5.0)
    ax2.set_xlabel('Query Index', fontsize=14)
    ax2.set_ylabel('Reference Index', fontsize=14)
    ax2.set_title("Weight Matrix", fontsize=16)
    ax2.set_aspect('equal')

    # 3. Trajectory Plot
    ax3 = fig.add_subplot(133)
    
    db_xy = np.array([convert_pose_to_2d(p) for p in db_poses])
    ax3.plot(db_xy[:,0], db_xy[:,1], c=PALLETE[1], linewidth=2, linestyle='--', label='Reference', zorder=1)
    
    # Plot query trajectory
    query_xy = np.array([convert_pose_to_2d(p) for p in query_poses])
    ax3.plot(query_xy[:,0], query_xy[:,1], c=PALLETE[2], linewidth=2, linestyle='-', label='Query', zorder=0)
    
    # Plot connections for valid pairs
    for q_idx, db_idx in valid_pairs:
        ax3.plot([query_xy[q_idx,0], db_xy[db_idx,0]],
                 [query_xy[q_idx,1], db_xy[db_idx,1]],
                 'g-', linewidth=1, alpha=0.5)

    ax3.set_xlabel('X [m]', fontsize=14)
    ax3.set_ylabel('Y [m]', fontsize=14)
    ax3.set_title("Trajectories", fontsize=16)
    ax3.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.05, 1.0))
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_vpr_data_queries(db_poses, all_query_poses, all_valid_pairs, query_names, output_dir):
    """Visualize VPR reference-query pairs for each query separately"""
    db_xy = np.array([convert_pose_to_2d(p) for p in db_poses])
    
    for query_id, (query_poses, query_name) in enumerate(zip(all_query_poses, query_names)):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        
        ax.plot(
            db_xy[:,0], db_xy[:,1], c=PALLETE[1], marker='o', markersize=3, label='Reference', zorder=0, linewidth=0
        )
                
        query_xy = np.array([convert_pose_to_2d(p) for p in query_poses])
        ax.plot(
            query_xy[:,0], query_xy[:,1], 
            c=PALLETE[3], 
            marker='o', markersize=3,
            linewidth=0,
            label=f'Query', 
            zorder=1,
        )

        for q_idx, db_idx in all_valid_pairs[query_id]:
            ax.plot([query_xy[q_idx,0], db_xy[db_idx,0]],
                    [query_xy[q_idx,1], db_xy[db_idx,1]],
                    '-', c='k', linewidth=1.5, alpha=1.0)
        
        N_query = len(query_poses)
        N_pos = len(all_valid_pairs[query_id])

        # ax.set_xlabel('X [m]', fontsize=16)
        # ax.set_ylabel('Y [m]', fontsize=16)
        # ax.set_aspect('equal')
        ax.set_title(f'Topological Localization [{args.trans_thresh:.1f}m, {args.rot_thresh:.1f}°]: N_pos={N_pos}, N_query={N_query}')
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        output_path = str(output_dir / f"vpr_data_{query_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        output_path = output_path.replace('.png', '.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved VPR visualization for {query_name} to {output_path}")

def save_img_valid_pairs(all_test_ds, all_valid_pairs, query_names, output_dir):
    """Save matched reference-query image pairs for each sequence"""
    
    for query_id, (test_ds, valid_pairs, query_name) in enumerate(zip(all_test_ds, all_valid_pairs, query_names)):
        query_output_dir = output_dir / query_name
        query_output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Saving matched image pairs for {query_name} ({len(valid_pairs)} pairs)")
        
        for q_idx, db_idx in valid_pairs:
            query_img_path = test_ds.queries_image_paths[q_idx]
            db_img_path = test_ds.database_image_paths[db_idx]
            
            query_img = Image.open(query_img_path).convert('RGB')
            db_img = Image.open(db_img_path).convert('RGB')
            query_img = query_img.resize(
                (int(query_img.width * 0.5), int(query_img.height * 0.5)),
                resample=Image.BICUBIC
            )
            db_img = db_img.resize(
                (int(db_img.width * 0.5), int(db_img.height * 0.5)),
                resample=Image.BICUBIC
            )
            
            fig, axes = plt.subplots(2, 1, figsize=(6, 6))
            
            axes[0].imshow(query_img)
            axes[0].set_title(f'Query Image (idx={q_idx})', fontsize=16)
            axes[0].axis('off')
            
            axes[1].imshow(db_img)
            axes[1].set_title(f'Reference Image (idx={db_idx})', fontsize=16)
            axes[1].axis('off')
            
            plt.tight_layout()
            output_path = query_output_dir / f"matched_query_{q_idx:04d}_ref_{db_idx:04d}.jpg"
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        logging.info(f"Saved {len(valid_pairs)} matched image pairs to {query_output_dir}")

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

def find_valid_matches(test_ds, trans_thresh, rot_thresh, dataset_name):
    """Find ground-truth valid pairs using pose thresholds"""
    valid_pairs = []
    
    for q_idx in range(test_ds.num_queries):
        query_img_name = test_ds.queries_image_names[q_idx]
        query_pose = test_ds.queries_poses[query_img_name]
        
        # Find nearest database image by position
        best_db_idx = None
        best_err = float('inf')
        for db_idx in range(test_ds.num_database):
            db_img_name = test_ds.database_image_names[db_idx]
            db_pose = test_ds.database_poses[db_img_name]

            Tc2w = convert_vec_to_matrix(query_pose[4:], query_pose[:4], 'wxyz')
            trans_query, quat_query = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
            Tc2w = convert_vec_to_matrix(db_pose[4:], db_pose[:4], 'wxyz')
            trans_db, quat_db = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')

            trans_err, rot_err = compute_pose_error(
                (trans_query, quat_query), 
                (trans_db, quat_db), mode='vector'
            )

            if trans_err <= trans_thresh and rot_err <= rot_thresh and trans_err < best_err:
                best_db_idx = db_idx
                best_err = trans_err
        
        if best_db_idx is not None:
            valid_pairs.append((q_idx, best_db_idx))
            
    return valid_pairs

def evaluate_vpr_system(args):
    """Main evaluation and visualization routine"""
    all_db_poses = []
    all_query_poses = []
    all_valid_pairs = []
    all_test_ds = []
    query_names = []

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for query_folder in args.queries_folder:
        test_ds = TestDataset(args.database_folder, query_folder, args.image_size)
        db_poses = np.array([test_ds.database_poses[test_ds.database_image_names[i]] for i in range(test_ds.num_database)])
        query_poses = np.array([test_ds.queries_poses[test_ds.queries_image_names[i]] for i in range(test_ds.num_queries)])
        if len(all_db_poses) == 0:
            all_db_poses.append(db_poses)
        all_query_poses.append(query_poses)
        all_test_ds.append(test_ds)
        query_names.append(query_folder.name)

        dataset_name = args.dataset_name
        valid_pairs = find_valid_matches(test_ds, args.trans_thresh, args.rot_thresh, dataset_name)
        all_valid_pairs.append(valid_pairs)

        # vpr_model = initialize_vpr_model(args.vpr_model, args.backbone, args.descriptors_dimension, args.device)
        # db_descs, query_descs = extract_descriptors(vpr_model, test_ds, args)
        # D_all = compute_diff_matrix(db_descs, query_descs)
        # visualize_vpr_data(db_poses, query_poses, D_all, valid_pairs, output_dir/f"vpr_data_{query_folder.name}.jpg")
        # np.save(output_dir/f"D_all_{query_folder.name}.npy", D_all)        

    # visualize_vpr_data_queries(all_db_poses[0], all_query_poses, all_valid_pairs, query_names, output_dir)
    save_img_valid_pairs(all_test_ds, all_valid_pairs, query_names, output_dir)

if __name__ == '__main__':
    args = parse_arguments()
    evaluate_vpr_system(args)
