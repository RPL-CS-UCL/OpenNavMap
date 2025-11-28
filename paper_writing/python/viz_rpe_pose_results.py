#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from pathlib import Path
from zipfile import ZipFile
from io import TextIOWrapper
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from python.utils.utils_setting_color_font import acquire_color_palette, acquire_marker, acquire_linestyle, setting_font
from python.utils.benchmark.utils import load_poses
from python.utils.utils_geom import convert_vec_to_matrix

setting_font(fontsize=14, titlesize=14, legend_fontsize=14)
PALLETE = acquire_color_palette()
LINESTYLE = acquire_linestyle()
MARKER = acquire_marker()

def draw_orientation_arrow(ax, transform, length, style):
    """Draws orientation arrow for a single camera."""
    start = transform[:3, 3]
    direction = transform[:3, :3] @ np.array([0, 0, length])
    
    head_width = style['head_width']
    head_length = style['head_length']
    zorder = style['zorder']
    fc = style['fc']
    ax.arrow(start[0], start[2], direction[0], direction[2], 
             head_width=head_width*0.8, head_length=head_length*1.2,
             width=head_width*0.15, fc=fc, ec=fc, zorder=zorder)

def main():
    parser = argparse.ArgumentParser(description='Visualize RPE results')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--methods', type=str, required=True, nargs='+', help='Method name')
    parser.add_argument('--top_k', type=int, required=True, help='Top k matches')
    parser.add_argument('--scenes', type=str, required=True, nargs='+', help='Scenes to visualize')
    args = parser.parse_args()

    abbr_method = {
        'hloc_disk_dilg': 'HLoc (DISK+LG)',
        'hloc_superpoint_splg': 'HLoc (SP+LG)',
        'vpr_cosplace_resnet18_256': 'VPR (CosPlace)',
        'vpr_netvlad_resnet18_4096': 'VPR (NetVLAD)',
        'reloc3r': 'Reloc3R',
        'duster_nocalib_pretrain': 'DUSt3R',
        'duster_calib_pretrain': 'Ours (DUSt3R)',
        'master_nocalib_pretrain': 'MASt3R',
        'master_calib_pretrain': 'Ours (MASt3R)'
    }

    if len(args.scenes) > 0:
        scenes = args.scenes
    else:
        scenes = sorted(os.listdir(os.path.join(args.dataset_dir, 'test')))

    for scene in scenes:
        scene_path = Path(os.path.join(args.dataset_dir, 'test', scene))
        out_path = Path(os.path.join(args.dataset_dir, 'results_rpe', f'viz_pose_results_{args.top_k}', scene))
        out_path.mkdir(parents=True, exist_ok=True)

        with (scene_path / 'poses.txt').open('r', encoding='utf-8') as gt_poses_file:
            gt_poses = load_poses(gt_poses_file, load_confidence=False, is_multi_frame=False)

        reference_list = []
        est_method_results = {method: [] for method in args.methods}
        gt_method_results = {method: [] for method in args.methods}
            
        for method in args.methods:
            if not os.path.exists(os.path.join(args.dataset_dir, 'results_rpe', method, f'submission_{args.top_k}.zip')):
                print(f'{method} does not exist in {scene}')
                continue

            submission_zip = ZipFile(os.path.join(args.dataset_dir, 'results_rpe', method, f'submission_{args.top_k}.zip'))            
            with submission_zip.open(f'pose_{scene}.txt') as estimated_poses_file:
                estimated_poses_file_wrapper = TextIOWrapper(estimated_poses_file, encoding='utf-8')
                estimated_poses = load_poses(estimated_poses_file_wrapper, load_confidence=True, is_multi_frame=True)
                for frame_name, (q_est, t_est, confidence) in estimated_poses.items():
                    est_method_results[method].append((q_est, t_est, confidence))

                    query_frame_name = frame_name.split(',')[-1]
                    q_gt, t_gt, _ = gt_poses[query_frame_name]
                    gt_method_results[method].append((q_gt, t_gt))

            tmp_reference_list = []
            with submission_zip.open(f'pose_{scene}.txt') as estimated_poses_file:
                estimated_poses_file_wrapper = TextIOWrapper(estimated_poses_file, encoding='utf-8')
                for line_number, line in enumerate(estimated_poses_file_wrapper.readlines()):
                    parts = tuple(line.strip().split(' '))
                    if '#' not in parts[0]:
                        num_ref = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
                        tmp_reference_list.append(parts[1:1+num_ref])
            if len(tmp_reference_list) > len(reference_list):
                reference_list = tmp_reference_list

        for query_id, ref_list in enumerate(reference_list):
            fig, ax = plt.subplots(figsize=(10, 6))

            all_positions = []
            for ref_id, ref_frame_name in enumerate(ref_list):
                q_gt, t_gt, _ = gt_poses[ref_frame_name]
                all_positions.append(t_gt)

            all_positions = np.array(all_positions)
            bounds = np.max(all_positions[:, :3], axis=0) - np.min(all_positions[:, :3], axis=0)
            max_bound = np.max(bounds) / 2 * 1.5
            arrow_length = max_bound / 10
            head_size = max_bound / 20
            arrow_style = {
                'head_width': head_size * 1.0,
                'head_length': head_size * 1.0,
                'fc': PALLETE[0],
                'zorder': 0,
            }
                
            try:
                handles = []
                labels = []

                ##### Plot reference poses with green arrows
                for ref_id, ref_frame_name in enumerate(ref_list):
                    q_gt, t_gt, _ = gt_poses[ref_frame_name]
                    transform = convert_vec_to_matrix(t_gt, q_gt, mode='wxyz')               
                    draw_orientation_arrow(ax, transform, arrow_length, arrow_style)
                    if ref_id == 0:
                        label = 'Reference Poses'
                    else:
                        label = None
                    ax.plot(transform[0, 3], transform[2, 3], color=PALLETE[0], marker='o', markersize=8,
                            label=label, zorder=0, markerfacecolor=PALLETE[0], linestyle='none', markeredgewidth=2)
                    
                ##### Plot GT query pose with red arrow
                q_gt, t_gt = gt_method_results['master_calib_pretrain'][query_id]
                transform_gt = convert_vec_to_matrix(t_gt, q_gt, mode='wxyz')
                arrow_style['fc'] = PALLETE[1]
                draw_orientation_arrow(ax, transform_gt, arrow_length, arrow_style)
                handle, = ax.plot(transform_gt[0, 3], transform_gt[2, 3], color=PALLETE[1], marker='o', markersize=23, 
                                  label='GT Pose', zorder=10, markerfacecolor='none', linestyle='none', markeredgewidth=3.0, alpha=1.0)
                handles.append(handle)
                labels.append('GT Pose')

                ##### Plot estimated poses with blue arrows and different edge styles
                for method_idx, method in enumerate(args.methods):
                    if method not in est_method_results or query_id >= len(est_method_results[method]):
                        continue

                    q_est, t_est, confidence = est_method_results[method][query_id]
                    transform_est = convert_vec_to_matrix(t_est, q_est, mode='wxyz')
                    arrow_style['fc'] = PALLETE[method_idx + 2]
                    arrow_style['zorder'] = 10
                    draw_orientation_arrow(ax, transform_est, arrow_length, arrow_style)
                    if method == 'master_calib_pretrain':
                        marker = 'X'
                    else:
                        marker = MARKER[method_idx + 1]
                    handle, = ax.plot(transform_est[0, 3], transform_est[2, 3], color=PALLETE[method_idx + 2],
                                      marker=marker, markersize=23,
                                      markerfacecolor='none', label=f'Est. Pose-{abbr_method[method]}', zorder=10, linestyle='none',
                                      markeredgewidth=3.5, alpha=1.0)
                    handles.append(handle)
                    labels.append(f'Est. Pose-{abbr_method[method]}')
                
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.xaxis.set_tick_params(labelsize=12)
                ax.yaxis.set_tick_params(labelsize=12)
                # ax.set_title(f'Scene: {scene} - {query_id}')
                # ax.legend(loc='upper left', bbox_to_anchor=(0.98, 1.025), fontsize=17, markerscale=12/25)
                # leg = ax.legend(fontsize=17, markerscale=12/25)
                ax.set(aspect='equal')
                ax.legend()
                plt.savefig(out_path / f'{scene}_{query_id}.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(out_path / f'{scene}_{query_id}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved visualization to {out_path / f'{scene}_{query_id}.pdf'}")

            except Exception as e:
                print(f'Error: {e} in {scene} - {query_id}')
                continue

if __name__ == '__main__':
    main()
