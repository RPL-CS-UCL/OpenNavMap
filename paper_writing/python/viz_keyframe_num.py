#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../python'))

import csv
import argparse
from matplotlib import pyplot as plt
import numpy as np
from utils.utils_setting_color_font import acquire_color_palette, acquire_marker, acquire_linestyle, setting_font

setting_font()  # Assume this configures fonts as previously discussed
PALLETE = acquire_color_palette()  # Assume returns color list
MARKERS = acquire_marker()  # Assume returns marker list
LINE_STYLE = acquire_linestyle()  # Assume returns line style list

def load_keyframe_counts(folder_path):
    """Load keyframe counts from merge_xxx folders in sorted order."""
    merge_folders = []
    
    # Find all merge_xxx folders and sort them
    for item in os.listdir(folder_path):
        if item.startswith('merge_') and os.path.isdir(os.path.join(folder_path, item)):
            if '55' not in item and '56' not in item:
                merge_folders.append(item)
    
    merge_folders.sort()

    keyframe_counts, segment_numbers = [], []
    for i, merge_folder in enumerate(merge_folders):
        seq_path = os.path.join(folder_path, merge_folder, 'seq')
        if os.path.exists(seq_path):
            color_files = [f for f in os.listdir(seq_path) if f.endswith('.color.jpg')]
            keyframe_count = len(color_files)
            keyframe_counts.append(keyframe_count)
            segment_numbers.append(i + 1)  # 1-based segment numbering
    
    return segment_numbers, keyframe_counts

def plot_keyframe_numbers(folder_paths, output_dir, labels):
    """Plot keyframe numbers for multiple folders."""
    plt.figure(figsize=(10, 6))
    
    for idx, folder_path in enumerate(folder_paths):
        segment_numbers, keyframe_counts = load_keyframe_counts(folder_path)
        folder_name = os.path.basename(folder_path)
        plt.plot(segment_numbers, keyframe_counts, 
                color=PALLETE[idx], 
                linewidth=2, 
                label=labels[idx])
    
    plt.xlabel('Segment Number', fontsize=14)
    plt.ylabel('Total Number of Keyframes', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'keyframe_numbers.pdf'), 
                bbox_inches='tight', dpi=300)
    plt.show()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize keyframe numbers across segments")
    parser.add_argument('--folders', nargs='+', required=True,
                        help='List of root directories containing merge_xxx folders')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='List of labels for each folder')
    parser.add_argument(
        '--output', default='./output',
                        help='Output directory for plots (default: ./output)')
    
    args = parser.parse_args()
    
    # Validate folders exist
    valid_folders = []
    for folder in args.folders:
        if os.path.exists(folder):
            valid_folders.append(folder)
        else:
            print(f"Warning: Folder {folder} does not exist, skipping...")
    
    if not valid_folders:
        print("Error: No valid folders provided!")
        return
    
    # Plot keyframe numbers
    plot_keyframe_numbers(valid_folders, args.output, args.labels)
    
    # Print summary statistics
    for folder in valid_folders:
        segment_numbers, keyframe_counts = load_keyframe_counts(folder)
        if segment_numbers:
            folder_name = os.path.basename(folder)
            total_keyframes = keyframe_counts[-1]
            print(f"\n{folder_name}:")
            print(f"  Total segments: {len(segment_numbers)}")
            print(f"  Total keyframes: {total_keyframes}")
            print(f"  Average keyframes per segment: {total_keyframes/len(segment_numbers):.1f}")

if __name__ == "__main__":
    main()


