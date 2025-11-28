#!/usr/bin/env python

import os
import sys
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from python.utils.utils_setting_color_font import *

setting_font(fontsize=14, titlesize=14, legend_fontsize=14)
PALLETE = acquire_color_palette()

def load_poses_tum_format(poses_file):
	"""
	Load poses from TUM format file.
	TUM format: timestamp tx ty tz qx qy qz qw
	Args:
		poses_file: Path to poses_closed_loop.txt file in TUM format
	Returns:
		numpy array of positions (N, 3)
	"""
	positions = []
	
	if not os.path.exists(poses_file):
		print(f"Warning: File not found: {poses_file}")
		return None
	
	with open(poses_file, 'r') as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith('#'):
				continue
			parts = line.split()
			if len(parts) >= 8:
				# TUM format: timestamp tx ty tz qx qy qz qw
				tx = float(parts[1])
				ty = float(parts[2])
				tz = float(parts[3])
				positions.append([tx, ty, tz])
	
	if len(positions) == 0:
		print(f"Warning: No poses loaded from {poses_file}")
		return None
	
	return np.array(positions)

def plot_all_trajectories(trajectories_dict, output_folder):
	"""
	Plot all trajectories in a single figure with different colors.
	Args:
		trajectories_dict: Dictionary mapping sequence_id -> positions array
		output_folder: Output folder for saving the plot
	"""
	fig, ax = plt.subplots(figsize=(10, 8))
	
	sorted_seq_ids = sorted(trajectories_dict.keys())
	
	for idx, seq_id in enumerate(sorted_seq_ids):
		positions = trajectories_dict[seq_id][::10]
		color_idx = idx % len(PALLETE)
		
		# Plot trajectory
		ax.plot(
			positions[:, 0], positions[:, 1],
			c=PALLETE[color_idx],
			linewidth=2.0,
			label=f"Seq {seq_id}",
			alpha=0.8
		)
		
		# Set a random offset for the text location to reduce overlap, and add an arrow connecting the label to the trajectory mid-point
		midpoint = positions[int(len(positions)/2)]
		rng = np.random.default_rng(hash(seq_id) % (2**32))  # deterministic per seq_id
		angle = rng.uniform(0, 2 * np.pi)
		distance = rng.uniform(5.0, 8.0)  # random offset distance
		dx = np.cos(angle) * distance
		dy = np.sin(angle) * distance
		text_x = midpoint[0] + dx
		text_y = midpoint[1] + dy
		ax.annotate(
			f"{seq_id}",
			xy=(midpoint[0], midpoint[1]),
			xytext=(text_x, text_y),
			textcoords='data',
			arrowprops=dict(
				arrowstyle="->",
				color='k',
				lw=1.5,
				shrinkA=2, shrinkB=2,
				alpha=0.8
			),
			fontsize=10,
			color=PALLETE[color_idx],
			fontweight='bold',
			ha='center',
			va='center',
			bbox=dict(boxstyle='circle,pad=0.30', facecolor='white', edgecolor=PALLETE[color_idx], linewidth=1)
		)

	# ax.set_xlabel('X [m]', fontsize=16)
	# ax.set_ylabel('Y [m]', fontsize=16)
	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	# ax.legend(fontsize=12, loc='best', ncol=2)
	ax.grid(True, linestyle='--', alpha=0.5)
	ax.set_aspect('equal')
	
	plt.tight_layout()
	output_png = os.path.join(output_folder, 'all_trajectories_1.png')
	output_pdf = os.path.join(output_folder, 'all_trajectories_1.pdf')
	plt.savefig(output_png, dpi=300, bbox_inches='tight')
	plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
	plt.close()
	
	print(f"Saved trajectory visualization to:")
	print(f"  - {output_png}")
	print(f"  - {output_pdf}")

def main():
	"""Main entry point."""
	parser = argparse.ArgumentParser(
		description="Visualize trajectories from out_general_* folders"
	)
	parser.add_argument(
		'--dataset_dir', 
		required=True,
		help='Root directory containing out_general_* folders'
	)
	parser.add_argument(
		'--prefix',
		default='out_general_ucl_campus_',
		help='Prefix of the folder name'
	)
	parser.add_argument(
		'--output_folder', 
		required=True,
		help='Output folder for saving the plots'
	)
	args = parser.parse_args()
	
	# Find all matching folders
	pattern = os.path.join(args.dataset_dir, f'{args.prefix}*')
	matching_folders = sorted(glob.glob(pattern))
	
	if len(matching_folders) == 0:
		print(f"Error: No folders matching pattern '{pattern}' found!")
		return
	
	print(f"Found {len(matching_folders)} folders matching '{args.prefix}*':")
	for folder in matching_folders:
		print(f"  - {os.path.basename(folder)}")
	
	# Load trajectories from each folder
	trajectories_dict = {}
	for seq_id, folder in enumerate(matching_folders):
		poses_file = os.path.join(folder, 'poses_closed_loop.txt')
		positions = load_poses_tum_format(poses_file)
		
		if positions is not None:
			trajectories_dict[seq_id] = positions
			print(f"Loaded sequence {seq_id}: {len(positions)} poses")
	
	if len(trajectories_dict) == 0:
		print("Error: No valid trajectories loaded!")
		return
	
	# Create output folder
	os.makedirs(args.output_folder, exist_ok=True)
	
	# Plot all trajectories
	plot_all_trajectories(trajectories_dict, args.output_folder)
	
	print(f"\nVisualization complete! Total sequences plotted: {len(trajectories_dict)}")

if __name__ == "__main__":
	main()
