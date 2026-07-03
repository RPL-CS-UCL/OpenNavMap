#!/usr/bin/env python3

import _bootstrap_imports  # noqa: F401
import os

import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation
from utils.utils_setting_color_font import acquire_color_palette, acquire_marker, acquire_linestyle, setting_font

setting_font(fontsize=18, titlesize=18, legend_fontsize=18)
colors = acquire_color_palette()
markers = acquire_marker()
linestyles = acquire_linestyle()

def count_images_in_seq(merge_folder_path):
    seq_folder = os.path.join(merge_folder_path, "seq")
    if not os.path.exists(seq_folder):
        return 0
    return len(glob.glob(os.path.join(seq_folder, "*.color.jpg")))

def load_cull_node_info(cull_info_file):
    """
    Load culled node information from cull_node_info.txt file.
    Each row format: node_id,type,compared_to,prob,method,prob_str
    Returns a dictionary with counts for each culling method.
    """
    cull_stats = {
        'culled_by_iqa': 0,
        'culled_by_forward': 0,
        'culled_by_backward': 0,
    }
    
    if not os.path.exists(cull_info_file):
        return cull_stats
    
    with open(cull_info_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                method = parts[4].strip()
                if method in cull_stats:
                    cull_stats[method] += 1
    
    return cull_stats

def aggregate_culling_data(base_dir):
    """
    Aggregate culling statistics from all merge folders.
    Returns lists of submap_indices and culling statistics for each merge.
    """
    merge_folders = [item for item in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, item)) and item.startswith("merge_")]
    merge_folders_sorted = sorted(merge_folders, key=lambda x: len(parse_merge_folder_name(x)))
    
    submap_indices = []
    cull_stats_list = []
    merge_labels = []
    
    for folder in merge_folders_sorted:
        folder_path = os.path.join(base_dir, folder)
        indices = parse_merge_folder_name(folder)
        if not indices:
            continue
        
        cull_info_file = os.path.join(folder_path, "preds", "cull_node_info.txt")
        cull_stats = load_cull_node_info(cull_info_file)
        
        last_submap_idx = indices[-1]
        submap_indices.append(last_submap_idx)
        cull_stats_list.append(cull_stats)
        merge_labels.append(folder)
        
        total_culled = sum(cull_stats.values())
        print(f"{folder}: Last submap = {last_submap_idx}, Total culled = {total_culled}")
        print(f"  - IQA: {cull_stats['culled_by_iqa']}, Forward: {cull_stats['culled_by_forward']}, "
              f"Backward: {cull_stats['culled_by_backward']}")
    
    return submap_indices, cull_stats_list, merge_labels

def load_submap_info(base_dir, submap_data_folder):
    submap_info = {}
    submap_idx = 0
    while True:
        submap_path = os.path.join(base_dir, submap_data_folder, str(submap_idx))
        timestamps_file = os.path.join(submap_path, "timestamps.txt")
        if not os.path.exists(timestamps_file):
            break
        timestamps = []
        with open(timestamps_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        timestamps.append(float(parts[1]))
        if timestamps:
            first_ts = timestamps[0]
            first_date = datetime.fromtimestamp(first_ts)
            num_images = len(timestamps)
            submap_info[submap_idx] = {
                'num_images': num_images,
                'first_timestamp': first_ts,
                'first_date': first_date,
                'date_str': first_date.strftime('%m/%d')
            }
            print(f"  Submap {submap_idx}: {num_images} images, date: {first_date.strftime('%Y-%m-%d')}")
        submap_idx += 1
    return submap_info

def parse_merge_folder_name(folder_name):
    if "merge_finalmap" in folder_name:
        return []
    if not folder_name.startswith("merge_"):
        return []
    indices_str = folder_name.replace("merge_", "")
    if not indices_str:
        return []
    return [int(x) for x in indices_str.split("_")]

def aggregate_merge_data(base_dir):
    merge_folders = [item for item in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, item)) and item.startswith("merge_")]
    merge_folders_sorted = sorted(merge_folders, key=lambda x: len(parse_merge_folder_name(x)))
    print(f"Found {len(merge_folders_sorted)} merge folders")
    submap_indices = []
    num_nodes = []
    merge_labels = []
    for folder in merge_folders_sorted:
        folder_path = os.path.join(base_dir, folder)
        indices = parse_merge_folder_name(folder)
        if not indices:
            continue
        num_images = count_images_in_seq(folder_path)
        last_submap_idx = indices[-1]
        submap_indices.append(last_submap_idx)
        num_nodes.append(num_images)
        merge_labels.append(folder)
        print(f"{folder}: Last submap = {last_submap_idx}, Num nodes = {num_images}")
    return submap_indices, num_nodes, merge_labels

def plot_growth(args, submap_indices_list, num_nodes_list, merge_labels_list, method_labels, submap_info):
    fig, ax = plt.subplots(figsize=(10, 5.0))
    cumulative_frames = [0]
    for submap_idx in sorted(submap_info.keys()):
        cumulative_frames.append(cumulative_frames[-1] + submap_info[submap_idx]['num_images'])
    for method_idx, (submap_indices, num_nodes, label) in enumerate(zip(submap_indices_list, num_nodes_list, method_labels)):
        x_positions = [0]
        y_values = [0]
        for idx, nodes in zip(submap_indices, num_nodes):
            if idx in submap_info:
                x_positions.append(cumulative_frames[idx + 1])
            else:
                x_positions.append(cumulative_frames[-1])
            y_values.append(nodes)       
        ax.plot(
            x_positions, y_values, 
            marker=markers[method_idx], 
            markersize=7, 
            linewidth=2.0, 
            color=colors[method_idx], 
            linestyle=linestyles[0],
            label=label
        )
    
    for i, submap_idx in enumerate(sorted(submap_info.keys())):
        x_pos = cumulative_frames[i + 1]
        ax.axvline(x=x_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
        if submap_idx in submap_info:
            x_text = (cumulative_frames[i] + cumulative_frames[i + 1]) / 2.0
            # if submap_idx == 0:
            #     ax.text(
            #         (cumulative_frames[i] - cumulative_frames[i + 1] * 1.1) / 2.0,
            #         ax.get_ylim()[1] * 1.005, f"ID", 
            #         ha='center', va='bottom', fontsize=18, rotation=0, fontweight='bold'
            #     )
            ax.text(
                x_text, ax.get_ylim()[1] * 0.99, f"{submap_idx}", 
                ha='center', va='bottom', fontsize=18, rotation=0, fontweight='bold'
            )

    ax.grid(True, alpha=0.7, linestyle='--')
    ax.set_xlabel('Incoming Nodes', fontsize=18)
    ax.set_ylabel('Nodes After Culling', fontsize=18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    max_nodes = max(max(nodes) for nodes in num_nodes_list)
    ax.set_xlim(left=-10, right=max_nodes + 30)
    ax.legend(fontsize=16)

    plt.tight_layout()
    output_dir = os.path.join(args.base_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "map_merging_lifelong_growth.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    output_path_pdf = os.path.join(output_dir, "map_merging_lifelong_growth.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Figure saved to: {output_path_pdf}")

def plot_culling_statistics(args, submap_indices_list, cull_stats_list_all, method_labels):
    colors = acquire_color_palette()  
    cull_method_labels = {
        'culled_by_iqa': 'PreFilter (IQA)',
        'culled_by_forward': 'Forward (IQA+IG)',
        'culled_by_backward': 'Backward (IG+TD)',
    }
    
    num_methods = len(method_labels)
    if num_methods == 0:
        return

    fig, axes = plt.subplots(1, num_methods, figsize=(5.2 * num_methods, 3.8), squeeze=False)
    plt.subplots_adjust(wspace=0.0)
    axes = axes.flatten()
    
    for method_idx, (submap_indices, cull_stats_list, label) in enumerate(zip(submap_indices_list, cull_stats_list_all, method_labels)):
        ax = axes[method_idx]
        x_positions = submap_indices
        iqa_counts = [stats['culled_by_iqa'] for stats in cull_stats_list]
        forward_counts = [stats['culled_by_forward'] for stats in cull_stats_list]
        backward_counts = [stats['culled_by_backward'] for stats in cull_stats_list]
        
        bar_width = 0.8
        p1 = ax.bar(
            x_positions, iqa_counts, bar_width, 
            label=cull_method_labels['culled_by_iqa'],
            edgecolor='black', linewidth=1.1,
            color=colors[0], alpha=0.7,
            hatch=''
        )
        p2 = ax.bar(
            x_positions, forward_counts, bar_width,
            bottom=iqa_counts,
            label=cull_method_labels['culled_by_forward'],
            edgecolor='black', linewidth=1.1,
            color=colors[1], alpha=0.7,
            hatch='///'
        )
        p3 = ax.bar(
            x_positions, backward_counts, bar_width,
            bottom=np.array(iqa_counts) + np.array(forward_counts),
            label=cull_method_labels['culled_by_backward'],
            edgecolor='black', linewidth=1.1,
            color=colors[2], alpha=0.7,
            hatch='xx'
        )
        
        ax.set_title(f'{label}', fontsize=16)
        ax.set_xlabel('Submap ID', fontsize=16)
        if method_idx == 0: ax.set_ylabel('Number of Culled Nodes', fontsize=16)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(bottom=y_min, top=y_max * 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(fontsize=13.5)
        ax.grid(True, alpha=0.7, linestyle='--', axis='y')
        
        if len(x_positions) <= 20:
            ax.set_xticks(x_positions)
        else:
            step = max(1, len(x_positions) // 20)
            ax.set_xticks(x_positions[::step])
    
    plt.tight_layout()
    output_dir = os.path.join(args.base_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "culling_statistics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCulling statistics figure saved to: {output_path}")
    output_path_pdf = os.path.join(output_dir, "culling_statistics.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Culling statistics figure saved to: {output_path_pdf}")

def print_culling_summary(submap_indices, cull_stats_list, method_label):
    """Print summary statistics for culled nodes."""
    print("\n" + "="*60)
    print(f"CULLING STATISTICS for '{method_label}'")
    print("="*60)
    
    total_iqa = sum(stats['culled_by_iqa'] for stats in cull_stats_list)
    total_forward = sum(stats['culled_by_forward'] for stats in cull_stats_list)
    total_backward = sum(stats['culled_by_backward'] for stats in cull_stats_list)
    total_culled = total_iqa + total_forward + total_backward
    print(f"Total culled nodes: {total_culled:,}")
    if total_culled > 0:
        print(f"  - Culling   (IQA: {total_iqa:,} ({100.0 * total_iqa / total_culled:.1f}%)")
        print(f"  - Culling-F (IQA+IG): {total_forward:,} ({100.0 * total_forward / total_culled:.1f}%)")
        print(f"  - Culling-B (IG+TD): {total_backward:,} ({100.0 * total_backward / total_culled:.1f}%)")

    avg_culled_per_merge = total_culled / len(cull_stats_list) if len(cull_stats_list) > 0 else 0
    print(f"Average culled per merge: {avg_culled_per_merge:.1f}")
    print("="*60)

def print_node_summary(submap_indices, num_nodes, method_label):
    print("\n" + "="*60)
    print(f"SUMMARY STATISTICS for '{method_label}'")
    print("="*60)
    print(f"Total number of merges: {len(submap_indices)}")
    print(f"Submap range: {min(submap_indices)} to {max(submap_indices)}")
    print(f"Initial nodes (submap {min(submap_indices)}): {num_nodes[0]:,}")
    print(f"Final nodes (submap {max(submap_indices)}): {num_nodes[-1]:,}")
    print(f"Total growth: {num_nodes[-1] - num_nodes[0]:,} nodes ({((num_nodes[-1] / num_nodes[0] - 1) * 100):.1f}% increase)")
    if len(num_nodes) > 1:
        print(f"Average nodes per submap: {np.mean(np.diff(num_nodes)):.1f}")
    else:
        print(f"Average nodes per submap: N/A")
    print("="*60)

def load_trajectory_data(merge_folder_path, time_gap_threshold=600.0):
    """
    Load trajectory data from a merge folder.
    Returns: dict with submap segments containing poses, timestamps, and image availability
    """
    timestamps_file = os.path.join(merge_folder_path, "timestamps.txt")
    poses_file = os.path.join(merge_folder_path, "poses.txt")
    seq_folder = os.path.join(merge_folder_path, "seq")
    
    if not os.path.exists(timestamps_file) or not os.path.exists(poses_file):
        print(f"Warning: Missing trajectory files in {merge_folder_path}")
        return None
    
    # Load timestamps
    timestamps_data = []
    image_names = []
    with open(timestamps_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name = parts[0]
                timestamp = float(parts[1])
                timestamps_data.append(timestamp)
                image_names.append(img_name)
    
    # Load poses (format: image_name qx qy qz qw tx ty tz)
    poses_data = []
    with open(poses_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                quat = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                trans = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                T_w2c = np.eye(4)
                T_w2c[:3, :3] = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
                T_w2c[:3, 3] = trans
                T_c2w = np.linalg.inv(T_w2c)
                poses_data.append(T_c2w)
    
    if len(timestamps_data) != len(poses_data):
        print(f"Warning: Mismatch between timestamps ({len(timestamps_data)}) and poses ({len(poses_data)})")
        return None
    
    # Check which images exist in seq folder
    image_exists = []
    for img_name in image_names:
        img_path = os.path.join(merge_folder_path, img_name)
        image_exists.append(os.path.exists(img_path))
    
    # Split into submap segments based on time gaps
    segments = []
    current_segment = {
        'timestamps': [timestamps_data[0]],
        'poses': [poses_data[0]],
        'image_exists': [image_exists[0]],
        'image_names': [image_names[0]]
    }
    
    for i in range(1, len(timestamps_data)):
        time_diff = timestamps_data[i] - timestamps_data[i-1]
        if time_diff > time_gap_threshold:
            segments.append(current_segment)
            current_segment = {
                'timestamps': [timestamps_data[i]],
                'poses': [poses_data[i]],
                'image_exists': [image_exists[i]],
                'image_names': [image_names[i]]
            }
        else:
            current_segment['timestamps'].append(timestamps_data[i])
            current_segment['poses'].append(poses_data[i])
            current_segment['image_exists'].append(image_exists[i])
            current_segment['image_names'].append(image_names[i])
    
    # Add last segment
    segments.append(current_segment)
    
    print(f"Found {len(segments)} submap segments with time gap threshold {time_gap_threshold}s")
    for idx, seg in enumerate(segments):
        num_with_images = sum(seg['image_exists'])
        num_total = len(seg['image_exists'])
        print(f"  Segment {idx}: {num_total} poses, {num_with_images} with images, "
              f"{num_total - num_with_images} without images")
    
    return segments

def plot_trajectory(args, merge_folder_name, result_dirs, method_labels, time_gap_threshold=600.0):
    num_methods = len(result_dirs)
    if num_methods == 0:
        return
    
    ncols = min(num_methods, 4)
    nrows = (num_methods + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 5.0 * nrows), squeeze=False, sharey=True)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = axes.flatten()
    
    for method_idx, (result_dir, label) in enumerate(zip(result_dirs, method_labels)):
        ax = axes[method_idx]
        merge_folder_path = os.path.join(args.base_dir, result_dir, merge_folder_name)
        
        print(f"\nLoading trajectory for: {label}")
        print(f"  Path: {merge_folder_path}")
        
        segments = load_trajectory_data(merge_folder_path, time_gap_threshold)
        if segments is None:
            print(f"  Warning: Failed to load trajectory data")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
            ax.set_title(label, fontsize=14)
            continue
        
        for seg_idx, segment in enumerate(segments):
            poses = np.array(segment['poses'])
            poses_xy = poses[:, :2, 3]
            image_exists = np.array(segment['image_exists'])
            poses_xy_without_img = poses_xy[~image_exists]
            if seg_idx == 0:
                plot_label = 'Merged Map'
            else:
                plot_label = None
            ax.plot(
                poses_xy[:, 0], poses_xy[:, 1], '.', 
                color=colors[0], alpha=1.0,
                markersize=10, linestyle=None,
                label=plot_label
            )            

            if seg_idx == 0:
                plot_label = 'Culled Nodes'
            else:
                plot_label = None
            ax.plot(
                poses_xy_without_img[:, 0], poses_xy_without_img[:, 1], '^',
                color=colors[1], linewidth=1.5, alpha=1.0, markersize=9, zorder=100,
                markeredgecolor='black', markeredgewidth=0.9,
                label=plot_label
            )
        
        ax.set_title(label, fontsize=18)
        ax.set_xlabel('X [m]', fontsize=16)
        if method_idx == 0:
            ax.set_ylabel('Y [m]', fontsize=16)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.7, linestyle='--')
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=14)
    
    for idx in range(num_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()   
    output_dir = os.path.join(args.base_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"merged_trajectory_s00003_exp_culling.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMerged trajectory figure saved to: {output_path}")
    output_path_pdf = os.path.join(output_dir, f"merged_trajectory_s00003_exp_culling.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Merged trajectory figure saved to: {output_path_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Visualize lifelong map merging growth (number of nodes vs submap index) for one or more methods.")
    parser.add_argument('--base_dir', type=str, required=True, 
                        help="Base directory for the result directories")
    parser.add_argument('--result_dirs', type=str, nargs='+', required=True, 
                        help="List of result directories for different experiment methods, e.g., .../exp1 .../exp2")
    parser.add_argument('--labels', type=str, nargs='+', required=True, 
                        help="Legend label for each method (must match number of result_dirs)")
    parser.add_argument('--submap_data_folder', type=str, required=True,
                        help="Folder containing individual submap data (e.g., s00003_exp_culling_aria_data)")
    parser.add_argument('--analyze_culling', action='store_true',
                        help="Analyze and plot culling statistics from cull_node_info.txt files")
    parser.add_argument('--viz_trajectory', action='store_true',
                        help="Visualize trajectories from merge folders")
    parser.add_argument('--trajectory_merge_folder', type=str, default='merge_0_1_2_3_4',
                        help="Name of the merge folder to visualize trajectory (default: merge_0_1_2_3_4)")
    parser.add_argument('--time_gap_threshold', type=float, default=600.0,
                        help="Time gap threshold in seconds to split submaps (default: 600.0)")
    args = parser.parse_args()

    if len(args.result_dirs) != len(args.labels):
        raise ValueError("The number of --result_dirs must match the number of --labels")

    print("============================\nLoading submap information...")
    submap_info = load_submap_info(args.base_dir, args.submap_data_folder)
    print(f"Loaded information for {len(submap_info)} submaps\n")

    submap_indices_list = []
    num_nodes_list = []
    merge_labels_list = ['']
    for result_dir, label in zip(args.result_dirs, args.labels):
        print(f"============================\nLoading method: {label}")
        submap_indices, num_nodes, merge_labels = aggregate_merge_data(os.path.join(args.base_dir, result_dir))
        submap_indices_list.append(submap_indices)
        num_nodes_list.append(num_nodes)
        merge_labels_list.append(merge_labels)
        print_node_summary(submap_indices, num_nodes, label)
    plot_growth(args, submap_indices_list, num_nodes_list, merge_labels_list, args.labels, submap_info)

    # Analyze culling statistics if requested
    if args.analyze_culling:
        print("\n" + "="*60)
        print("ANALYZING CULLING STATISTICS")
        print("="*60)
        
        submap_indices_cull_list = []
        cull_stats_list_all = []
        for result_dir, label in zip(args.result_dirs, args.labels):
            print(f"\n============================\nLoading culling data for: {label}")
            submap_indices_cull, cull_stats_list, merge_labels_cull = aggregate_culling_data(os.path.join(args.base_dir, result_dir))
            submap_indices_cull_list.append(submap_indices_cull)
            cull_stats_list_all.append(cull_stats_list)
            print_culling_summary(submap_indices_cull, cull_stats_list, label)
        
        rm_idx_list = []
        for i in range(len(args.labels)):
            if args.labels[i] == "Culling with Factors: IQA" or args.labels[i] == "W.O. Culling":
                rm_idx_list.append(i)
        
        filter_labels = args.labels.copy()
        for idx in sorted(rm_idx_list, reverse=True):
            filter_labels.pop(idx)
            submap_indices_cull_list.pop(idx)
            cull_stats_list_all.pop(idx)
        
        plot_culling_statistics(args, submap_indices_cull_list, cull_stats_list_all, filter_labels)

    # Visualize trajectories if requested
    if args.viz_trajectory:
        print("\n" + "="*60)
        print("VISUALIZING TRAJECTORIES")
        print("="*60)
        
        plot_trajectory(
            args, 
            args.trajectory_merge_folder, 
            args.result_dirs,
            args.labels,
            args.time_gap_threshold
        )

if __name__ == "__main__":
    main()
