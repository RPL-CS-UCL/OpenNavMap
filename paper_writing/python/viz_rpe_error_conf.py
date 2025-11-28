#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../python'))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils.utils_setting_color_font import acquire_color_palette, acquire_marker, setting_font

setting_font(fontsize=16, titlesize=16, legend_fontsize=13)
PALETTE = acquire_color_palette()
MARKERS = acquire_marker()

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize error vs confidence relationship')
    parser.add_argument('--model_names', type=str, nargs=2, required=True,
                      help='Two model names to load the corresponding data files')
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='Directory containing the data files')
    parser.add_argument('--error_thresholds', type=float, nargs='+', 
                      default=[0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
                      help='Error thresholds for grouping (default: 0.0 to 2.0 in 0.2 steps)')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory for the plot (default: same as data_dir)')
    return parser.parse_args()

def load_data(file_path):
    """
    Load data from file with format: trans_err, rot_err, conf (each row)
    """
    # Load data using numpy loadtxt
    data_array = np.loadtxt(file_path)
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(data_array, columns=['trans_err', 'rot_err', 'conf'])
    
    return df

def group_by_error_thresholds(data, thresholds):
    """
    Group data by translation error thresholds and compute mean confidence for each group
    """
    # Sort data by translation error
    data_sorted = data.sort_values('trans_err').reset_index(drop=True)
    
    # Create groups based on error thresholds
    groups = []
    mean_confidences = []
    group_labels = []
    
    for i in range(len(thresholds) - 1):
        lower_thresh = thresholds[i]
        upper_thresh = thresholds[i + 1]
        
        # Filter data within this error range
        mask = (data_sorted['trans_err'] >= lower_thresh) & (data_sorted['trans_err'] < upper_thresh)
        group_data = data_sorted[mask]
        
        if len(group_data) > 0:
            mean_conf = group_data['conf'].mean()
            groups.append(group_data)
            mean_confidences.append(mean_conf)
            group_labels.append(f'{lower_thresh:.1f}-{upper_thresh:.1f}')
        else:
            # If no data in this range, set confidence to 0
            mean_confidences.append(0.0)
            group_labels.append(f'{lower_thresh:.1f}-{upper_thresh:.1f}')
    
    return mean_confidences, group_labels, data_sorted

def main():
    # Parse arguments
    args = parse_args()

    # Load data for both models
    all_data = {}
    all_mean_confidences = {}
    all_sorted_data = {}
    
    for i, model_name in enumerate(args.model_names):
        # Construct file path
        file_path = os.path.join(args.data_dir, f'{model_name}/error_conf_8.txt')
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found!")
            return
        
        # Load data
        print(f"Loading data from: {file_path}")
        data = load_data(file_path)
        print(f"Loaded {len(data)} data points for {model_name}")
        print(f"Translation error range: {data['trans_err'].min():.3f} - {data['trans_err'].max():.3f}")
        print(f"Confidence range: {data['conf'].min():.3f} - {data['conf'].max():.3f}")
        
        # Group by error thresholds
        mean_confidences, group_labels, sorted_data = group_by_error_thresholds(data, args.error_thresholds)
        
        all_data[model_name] = data
        all_mean_confidences[model_name] = mean_confidences
        all_sorted_data[model_name] = sorted_data
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean confidence vs error groups for both models
    x_pos = np.arange(len(group_labels))
    
    for i, model_name in enumerate(args.model_names):
        mean_confidences = all_mean_confidences[model_name]
        ax1.plot(x_pos, mean_confidences, marker=MARKERS[i], markersize=8, 
                 color=PALETTE[i], linewidth=2, label=f'{model_name} (Mean Confidence)')
        
        # Add value labels on points
        for j, conf in enumerate(mean_confidences):
            if conf > 0:
                ax1.text(j, conf + 0.01, f'{conf:.3f}', 
                        ha='center', va='bottom', fontsize=8, color=PALETTE[i])
    
    ax1.set_xlabel('Translation Error Range (m)')
    ax1.set_ylabel('Mean Confidence')
    ax1.set_title(f'Mean Confidence vs Translation Error Groups\n({args.model_names[0]} vs {args.model_names[1]})')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(group_labels, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot 2: Scatter plot of all data points for both models
    for i, model_name in enumerate(args.model_names):
        sorted_data = all_sorted_data[model_name]
        # Filter data points with error < 1.0
        mask = sorted_data['trans_err'] < 1.0
        filtered_data = sorted_data[mask]
        
        scatter = ax2.scatter(filtered_data['trans_err'], filtered_data['conf'], 
                             alpha=0.6, s=20, c=filtered_data['trans_err'], 
                             cmap='viridis', edgecolors=PALETTE[i], linewidth=0.5,
                             marker=MARKERS[i], label=f'{model_name} (scatter)')
    
    ax2.set_xlabel('Translation Error (m)')
    ax2.set_ylabel('Confidence')
    ax2.set_title(f'Translation Error vs Confidence\n({args.model_names[0]} vs {args.model_names[1]})')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Add colorbar for scatter plot
    scatter = ax2.scatter([], [], c=[], cmap='viridis')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Translation Error (m)')
    
    plt.tight_layout()
    
    # Save figure
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{args.model_names[0]}_vs_{args.model_names[1]}_error_conf_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")
    
    # Print summary statistics for both models
    for model_name in args.model_names:
        data = all_data[model_name]
        print(f"\nSummary Statistics for {model_name}:")
        print(f"Total data points: {len(data)}")
        print(f"Mean translation error: {data['trans_err'].mean():.3f} ± {data['trans_err'].std():.3f}")
        print(f"Mean confidence: {data['conf'].mean():.3f} ± {data['conf'].std():.3f}")
        
        print(f"\nGroup Statistics for {model_name}:")
        mean_confidences = all_mean_confidences[model_name]
        for i, (label, conf) in enumerate(zip(group_labels, mean_confidences)):
            if conf > 0:
                print(f"  {label}: mean_conf = {conf:.3f}")

if __name__ == '__main__':
    main()
