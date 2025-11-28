#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import matplotlib.pyplot as plt
import numpy as np
import argparse
from python.utils.utils_setting_color_font import acquire_color_palette, acquire_marker, acquire_linestyle, setting_font

setting_font(fontsize=14, titlesize=14, legend_fontsize=14)
colors = acquire_color_palette()
markers = acquire_marker()
linestyles = acquire_linestyle()

def parse_merge_folder_name(folder_name):
    """
    Parse merge folder name to extract submap indices.
    Example: "merge_0_1_2" -> [0, 1, 2]
    """
    if "merge_finalmap" in folder_name:
        return []
    if not folder_name.startswith("merge_"):
        return []
    indices_str = folder_name.replace("merge_", "")
    if not indices_str:
        return []
    return [int(x) for x in indices_str.split("_")]

def load_edge_history(edge_history_file):
    """
    Load precision, recall, and edge count data from edge_history.txt file.
    Returns dict with 'precision', 'recall', and 'edge_counts' keys or None if parsing fails.
    """
    if not os.path.exists(edge_history_file):
        return None
    
    precision_values = None
    recall_values = None
    edges_added_vpr = None
    edges_removed_gv = None
    edges_removed_ccm = None
    
    try:
        with open(edge_history_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Precision:'):
                    precision_str = line.split(':', 1)[1].strip()
                    precision_values = [float(x) for x in precision_str.split(',')]
                elif line.startswith('Recall:'):
                    recall_str = line.split(':', 1)[1].strip()
                    recall_values = [float(x) for x in recall_str.split(',')]
                elif line.startswith('Number of edges added by VPR:'):
                    edges_added_vpr = int(line.split(':')[1].strip())
                elif line.startswith('Number of edges removed by GV:'):
                    edges_removed_gv = int(line.split(':')[1].split('(')[0].strip())
                elif line.startswith('Number of edges removed by CCM:'):
                    edges_removed_ccm = int(line.split(':')[1].split('(')[0].strip())
        
        if (precision_values is not None and recall_values is not None and
            edges_added_vpr is not None and edges_removed_gv is not None and
            edges_removed_ccm is not None):
            if len(precision_values) == 3 and len(recall_values) == 3:
                after_vpr = edges_added_vpr
                after_gv = after_vpr - edges_removed_gv
                after_ccm = after_gv - edges_removed_ccm
                return {
                    'precision': precision_values,
                    'recall': recall_values,
                    'edge_counts': [after_vpr, after_gv, after_ccm]  # [After VPR, After GV, After CCM]
                }
    except Exception as e:
        print(f"Warning: Error parsing {edge_history_file}: {e}")
        return None
    
    return None

def aggregate_precision_recall_data(base_dir):
    """
    Aggregate precision, recall, and edge count data from all merge folders.
    Returns lists: submap_indices, precision_data, recall_data, edge_counts_data, merge_labels.
    """
    merge_folders = [item for item in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, item)) and item.startswith("merge_")]
    merge_folders_sorted = sorted(merge_folders, key=lambda x: len(parse_merge_folder_name(x)))
    
    submap_indices = []
    precision_data = []
    recall_data = []
    edge_counts_data = []
    merge_labels = []
    
    print(f"Found {len(merge_folders_sorted)} merge folders")
    
    for folder in merge_folders_sorted:
        folder_path = os.path.join(base_dir, folder)
        indices = parse_merge_folder_name(folder)
        if not indices:
            continue
        
        edge_history_file = os.path.join(folder_path, "preds", "edge_history.txt")
        edge_data = load_edge_history(edge_history_file)
        
        if edge_data is None:
            print(f"Warning: Could not load edge history for {folder}")
            continue
        
        last_submap_idx = indices[-1]
        submap_indices.append(last_submap_idx - 1)
        precision_data.append(edge_data['precision'])
        recall_data.append(edge_data['recall'])
        edge_counts_data.append(edge_data['edge_counts'])
        merge_labels.append(folder)
        
        print(f"{folder}: Last submap = {last_submap_idx}, "
              f"Edges (VPR/GV/CCM) = {edge_data['edge_counts']}, "
              f"Precision (VPR/GV/CCM) = {edge_data['precision']}, "
              f"Recall (VPR/GV/CCM) = {edge_data['recall']}")
    
    return submap_indices, precision_data, recall_data, edge_counts_data, merge_labels

def plot_precision_recall(args, submap_indices, precision_data, recall_data, edge_counts_data, merge_labels):
    """
    Plot edge counts, precision and recall curves (vertical layout with 3 subplots).
    """

    remove_indices = []
    for index, (precision, recall) in enumerate(zip(precision_data, recall_data)):
        if recall[0] == 0 or precision[0] == 0:
            remove_indices.append(index)

    precision_data = [precision for i, precision in enumerate(precision_data) if i not in remove_indices]
    recall_data = [recall for i, recall in enumerate(recall_data) if i not in remove_indices]
    edge_counts_data = [edge_counts for i, edge_counts in enumerate(edge_counts_data) if i not in remove_indices]
    # submap_indices = [submap_indices[i] for i in range(len(submap_indices)) if i not in remove_indices]
    submap_indices = [i for i in range(len(precision_data))]
    
    precision_array = np.array(precision_data) * 100
    recall_array = np.array(recall_data) * 100
    edge_counts_array = np.array(edge_counts_data)
    
    method_labels = ['SM', 'GV', 'CCM']
    stage_labels = ['SM', 'GV', 'CCM']
    
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.0))
    fig.subplots_adjust(hspace=0.00)
    
    ax_edges = axes[0]
    for i, label in enumerate(stage_labels):
        ax_edges.plot(
            submap_indices, edge_counts_array[:, i],
            linewidth=1.0, 
            color=colors[i],
            linestyle=linestyles[i],
            label=label,
            zorder=i,
        )
    for x in submap_indices[::5]:
        ax_edges.axvline(x=x, color='gray', linestyle=':', linewidth=0.3, zorder=-1)

    ax_edges.set_ylabel('Number of Node Pairs', fontsize=12)
    # ax_edges.grid(True, alpha=0.7, linestyle='--')
    ax_edges.legend(fontsize=12, loc='upper center', ncol=len(method_labels), bbox_to_anchor=(0.5, 1.05))
    ax_edges.tick_params(axis='both', labelsize=12)
    all_x_ticks = np.arange(min(submap_indices), max(submap_indices)+1)
    ax_edges.set_xticks(all_x_ticks[::5])
    ax_edges.set_xlim(-0.5, len(submap_indices)-0.5)

    # Middle subplot: Precision (now second, previously Recall)
    ax_precision = axes[1]
    for i, label in enumerate(method_labels):
        ax_precision.plot(
            submap_indices, precision_array[:, i],
            linewidth=1.0, 
            color=colors[i],
            linestyle=linestyles[i],
            label=label,
            zorder=i,
        )
    for x in submap_indices[::5]:
        ax_precision.axvline(x=x, color='gray', linestyle=':', linewidth=0.3, zorder=-1)

    ax_precision.set_ylabel('Precision@[7.5m, 75°](\%)', fontsize=12)
    ax_precision.set_ylim([0, 105])
    # ax_precision.grid(True, alpha=0.7, linestyle='--')
    # ax_precision.legend(fontsize=12, loc='upper right', ncol=len(method_labels))
    ax_precision.tick_params(axis='both', labelsize=12)
    ax_precision.set_xticks(all_x_ticks[::5])
    ax_precision.set_xlim(-0.5, len(submap_indices)-0.5)

    # Bottom subplot: Recall (now third)
    ax_recall = axes[2]
    for i, label in enumerate(method_labels):
        ax_recall.plot(
            submap_indices, recall_array[:, i],
            linewidth=1.0, 
            color=colors[i],
            linestyle=linestyles[i],
            label=label,
            zorder=i,
        )
    for x in submap_indices[::5]:
        ax_recall.axvline(x=x, color='gray', linestyle=':', linewidth=0.3, zorder=-1)

    ax_recall.set_xlabel('Incoming Submap', fontsize=12)
    ax_recall.set_ylabel('Recall@[7.5m, 75°](\%)', fontsize=12)
    ax_recall.set_ylim([0, 105])
    # ax_recall.grid(True, alpha=0.7, linestyle='--')
    # ax_recall.legend(fontsize=12, loc='lower right', ncol=len(method_labels))
    ax_recall.tick_params(axis='both', labelsize=12)
    ax_recall.set_xticks(all_x_ticks[::5])
    ax_recall.set_xlim(-0.5, len(submap_indices)-0.5)
    plt.tight_layout()
    
    output_dir = os.path.join(args.base_dir, "../figures")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_png = os.path.join(output_dir, "map_merging_precision_recall.png")
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path_png}")
    
    output_path_pdf = os.path.join(output_dir, "map_merging_precision_recall.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Figure saved to: {output_path_pdf}")

def print_summary_statistics(submap_indices, precision_data, recall_data):
    print("\n" + "="*60)
    print("PRECISION AND RECALL STATISTICS")
    print("="*60)
    
    precision_array = np.array(precision_data)
    recall_array = np.array(recall_data)
    
    method_labels = ['VPR', 'GV', 'CCM']
    
    for i, label in enumerate(method_labels):
        print(f"\n{label}:")
        print(f"  Precision: mean={precision_array[:, i].mean():.3f}, "
              f"std={precision_array[:, i].std():.3f}, "
              f"min={precision_array[:, i].min():.3f}, "
              f"max={precision_array[:, i].max():.3f}")
        print(f"  Recall:    mean={recall_array[:, i].mean():.3f}, "
              f"std={recall_array[:, i].std():.3f}, "
              f"min={recall_array[:, i].min():.3f}, "
              f"max={recall_array[:, i].max():.3f}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Visualize precision and recall metrics for map merging from edge_history.txt files."
    )
    parser.add_argument('--base_dir', type=str, required=True,
                        help="Base directory containing merge folders (merge_0, merge_0_1, etc.)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        raise ValueError(f"Base directory does not exist: {args.base_dir}")
    
    print("="*60)
    print("LOADING PRECISION AND RECALL DATA")
    print("="*60)
    
    submap_indices, precision_data, recall_data, edge_counts_data, merge_labels = aggregate_precision_recall_data(args.base_dir)
    
    if len(submap_indices) == 0:
        print("\nNo valid merge folders found with edge_history.txt files!")
        return
    
    print(f"\nLoaded data from {len(submap_indices)} merge folders")
    
    print_summary_statistics(submap_indices, precision_data, recall_data)
    
    plot_precision_recall(args, submap_indices, precision_data, recall_data, edge_counts_data, merge_labels)

if __name__ == "__main__":
    main()

