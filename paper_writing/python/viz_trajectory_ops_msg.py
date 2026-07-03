#!/usr/bin/env python

import _bootstrap_imports  # noqa: F401
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils.utils_setting_color_font import acquire_color_palette, acquire_marker, setting_font
from viz_trajectory_ops import plot_connected_arrows

# Visualization setup
setting_font(fontsize=12, titlesize=20, legend_fontsize=12)
PALLETE = acquire_color_palette()
MARKERS = acquire_marker()

TARGET_POS_OPS_MSG = np.array([12.044489277, 16.127524238])

def filter_positions(positions):
    """Filter positions to remove points that are too close to each other."""
    target_pos = TARGET_POS_OPS_MSG

    for i in range(len(positions)):
        dis = np.linalg.norm(positions[i, 1:3] - target_pos)
        if dis < 1.0:
            print(abs(positions[i, 0] - positions[0, 0]))
            positions = positions[i:]
            break

    for i in range(len(positions)):
        dis = np.linalg.norm(positions[i, 1:3] - target_pos)
        if abs(positions[i, 0] - positions[0, 0]) > 200.0 and dis < 1.0:
            positions = positions[:i]
            break

    filtered_positions = []
    for i in range(len(positions)):
        if abs(positions[i, 0] - positions[0, 0]) > 10.0:
            filtered_positions.append(positions[i])
            
    return np.array(filtered_positions)
    

def plot_trajectory(ax, positions, velocities, idx, label):
    """Plot a trajectory on a 2D axis."""
    # Plot trajectory
    if idx == 0:
        ax.plot(positions[0, 0], positions[0, 1], color=PALLETE[5], markersize=15, marker='*', label='Start and End Point', zorder=100, linestyle='None')
        # ax.plot(positions[-1, 0], positions[-1, 1], color=PALLETE[5], markersize=12, marker='^', label='End Point', zorder=100, linestyle='None')    
    
    if idx % 2 == 0:
        ax.plot(positions[:, 0], positions[:, 1], color=PALLETE[idx], linestyle='-', alpha=0.9, label=label, linewidth=1.7, zorder=0)
    else:
        ax.plot(positions[:, 0], positions[:, 1], color=PALLETE[idx], linestyle='--', alpha=0.9, label=label, linewidth=1.7, zorder=10)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multiple trajectories visualization from TUM format files")
    parser.add_argument('--map_pose_file', help='Path to the map pose file')
    parser.add_argument('--map_edge_file', help='Path to the map edge file')
    parser.add_argument('--tum_files', nargs='+', help='Paths to the TUM format trajectory files')
    parser.add_argument('--vel_files', nargs='+', help='Paths to the TUM format velocity files')
    parser.add_argument('-o', '--output', help='Output PDF file path (default: trajectories.pdf)')
    args = parser.parse_args()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    ##### Plot map nodes
    poses = np.loadtxt(args.map_pose_file)
    positions = poses[:, 1:4]
    rotations = poses[:, 4: ] # xyzw
    edges = np.loadtxt(args.map_edge_file)
    # plot_connected_arrows(ax, positions, rotations, edges)
    
    if args.vel_files is not None:
        for idx, (tum_file, vel_file) in enumerate(zip(args.tum_files, args.vel_files)):
            # Read TUM file
            data = np.loadtxt(tum_file)
            positions_stamp = data[:, :4]  # timestamp, tx, ty, tz
            data = np.loadtxt(vel_file)
            velocities_stamp = data[:, :4]  # timestamp, vx, vy, vz
                    
            # For each position timestamp, find closest velocity timestamp
            velocities = np.zeros((len(positions_stamp), 3))
            for i, pos_time in enumerate(positions_stamp[:, 0]):
                time_diffs = np.abs(velocities_stamp[:, 0] - pos_time)
                closest_idx = np.argmin(time_diffs)
                velocities[i] = velocities_stamp[closest_idx, 1:4]

            # Plot trajectory
            filter_positions_stamp = filter_positions(positions_stamp)
            plot_trajectory(ax, filter_positions_stamp[:, 1:], velocities, idx, f"Robot Trajectory {idx}")
    else:
        for idx, tum_file in enumerate(args.tum_files):
            # Read TUM file
            data = np.loadtxt(tum_file)
            positions_stamp = data[:, :4]  # timestamp, tx, ty, tz
            filter_positions_stamp = filter_positions(positions_stamp)
            plot_trajectory(ax, filter_positions_stamp[:, 1:], None, idx, f"Robot Trajectory {idx}")

    # Set axis limits based on all positions
    all_positions = np.zeros((0, 2))
    for tum_file in args.tum_files:
        data = np.loadtxt(tum_file)
        all_positions = np.vstack((all_positions, data[:, 1:3]))  # Add trajectory positions
    
    x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
    y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
    
    # Add some padding
    x_padding = (x_max - x_min) * 0.02
    y_padding = (y_max - y_min) * 0.02
    
    # Calculate the aspect ratio to make x and y equal
    x_range = x_max - x_min + 2 * x_padding
    y_range = y_max - y_min + 2 * y_padding    
    aspect_ratio = x_range / y_range
    
    # Set the figure size to maintain equal aspect ratio
    fig.set_size_inches(6 * aspect_ratio, 5)
    
    # Set the limits while maintaining equal aspect ratio
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_aspect('equal')

    # ax.set_title("Visualization of Trajectories")
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Save and close
    plt.tight_layout()
    output_path = args.output if args.output else "trajectories.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved visualization to: {output_path}")

if __name__ == "__main__":
    main() 
