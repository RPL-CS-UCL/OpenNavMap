#!/usr/bin/env python

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from python.utils.utils_setting_color_font import *
from python.utils.utils_geom import read_timestamps

setting_font(fontsize=14, titlesize=14, legend_fontsize=14)
PALLETE = acquire_color_palette()
MARKERS = acquire_marker()
LINES = acquire_linestyle()
BAR_STYLE = acquire_bar_style()

def load_poses_from_file(poses_file):
	"""
	Load poses from file and convert to camera-to-world transformation matrices.
	Args:
		poses_file: Path to poses.txt file
	Returns:
		List of 4x4 transformation matrices
	"""
	poses_data = []
	
	if not os.path.exists(poses_file):
		return poses_data
	
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
	
	return poses_data

def load_merged_trajectory_data(merge_folder_path, translation_threshold=5.0):
	poses_file = os.path.join(merge_folder_path, "poses.txt")
	if not os.path.exists(poses_file):
		print(f"Warning: Missing poses.txt in {merge_folder_path}")
		return None
	
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
	
	if len(poses_data) == 0:
		print(f"Warning: No poses loaded from {poses_file}")
		return None
	
	sessions = []
	current_session = [poses_data[0]]
	
	for i in range(1, len(poses_data)):
		prev_pose = poses_data[i-1]
		curr_pose = poses_data[i]
		trans_diff = np.linalg.norm(curr_pose[:3, 3] - prev_pose[:3, 3])
		
		if trans_diff > translation_threshold:
			if len(current_session) > 5:
				sessions.append(np.array(current_session))
			current_session = [curr_pose]
		else:
			current_session.append(curr_pose)
	
	sessions.append(np.array(current_session))
	
	print(f"  Split trajectory into {len(sessions)} sessions (threshold: {translation_threshold}m)")
	for idx, session in enumerate(sessions):
		print(f"    Session {idx}: {len(session)} poses")
	
	return sessions

def plot_iqa_data(iqa_values_list, labels, output_folder):
	fig, ax = plt.subplots(figsize=(6, 6))
	
	all_values = np.concatenate(iqa_values_list)
	min_val = np.min(all_values)
	max_val = np.max(all_values)
	num_bins = 20
	bin_edges = np.linspace(min_val, max_val, num_bins + 1)
	
	hist_data = []
	for iqa_values in iqa_values_list:
		counts, _ = np.histogram(iqa_values, bins=bin_edges)
		hist_data.append(counts)
	
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
	bin_width = bin_edges[1] - bin_edges[0]
	
	bottom = np.zeros(num_bins)
	for i, (counts, label) in enumerate(zip(hist_data, labels)):
		ax.bar(
			bin_centers, counts, width=bin_width * 0.8,
			bottom=bottom, color=PALLETE[i], 
			edgecolor='black', linewidth=1.0,
			label=label, alpha=0.8,
			hatch=BAR_STYLE[i]
		)
		bottom += counts
	
	ax.tick_params(axis='x', labelsize=16)
	ax.tick_params(axis='y', labelsize=16)
	ax.set_xlabel('IQA Score', fontsize=20)
	ax.set_ylabel('Number of Images', fontsize=20)
	ax.legend(fontsize=20)
	ax.grid(True, linestyle='--', alpha=0.7, axis='y')
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_folder, 'iqa_data.png'), dpi=300)
	plt.savefig(os.path.join(output_folder, 'iqa_data.pdf'), dpi=300)
	plt.close()

def plot_poses_data(all_poses_data, label, output_folder, plot_start_end_points=False, plot_legend=False):
	fig, ax = plt.subplots(figsize=(5, 6))

	for i, pose_data in enumerate(all_poses_data):
		poses_array = np.array(pose_data)
		positions = poses_array[:, :3, 3]
		
		ax.plot(
			positions[:, 0], positions[:, 1], 
			c=PALLETE[i], linewidth=1.8,
			label=f"S{i}"
		)

	if plot_start_end_points:
		for i, pose_data in enumerate(all_poses_data):
			poses_array = np.array(pose_data)
			positions = poses_array[:, :3, 3]
			
			if i == 0:
				start_label = 'Start Point'
				end_label = 'End Point'
			else:
				start_label = None
				end_label = None

			ax.plot(
				positions[0, 0], positions[0, 1], '*', 
				c=PALLETE[5], markersize=10,
				label=start_label
			)
			ax.plot(
				positions[-1, 0], positions[-1, 1], '^', 
				c=PALLETE[5], markersize=8,
				label=end_label
			)
			
	if plot_legend:
		ax.legend(fontsize=10)
		# ax.set_xlabel('X [m]', fontsize=14)
		# ax.set_ylabel('Y [m]', fontsize=14)
	else:
		ax.invert_xaxis()
		ax.invert_yaxis()

	ax.tick_params(axis='x', labelsize=10)
	ax.tick_params(axis='y', labelsize=10)	
	ax.grid(True, linestyle='--', alpha=0.7)
	ax.set_aspect('equal')
	plt.tight_layout()
	plt.savefig(os.path.join(output_folder, f'raw_trajectory_{label}.png'), dpi=300, bbox_inches='tight')
	plt.savefig(os.path.join(output_folder, f'raw_trajectory_{label}.pdf'), dpi=300, bbox_inches='tight')
	plt.close()

def plot_merged_trajectory(sessions, label, output_folder, plot_start_end_points=False, plot_legend=False):
	fig, ax = plt.subplots(figsize=(6, 6))
	
	labeled_sessions = set()
	for sess_idx, session_poses in enumerate(sessions):
		positions = session_poses[:, :3, 3]
		if sess_idx not in labeled_sessions:
			session_label = f"S{sess_idx}"
			labeled_sessions.add(sess_idx)
		else:
			session_label = None
		
		ax.plot(
			positions[:, 0], positions[:, 1],
			c=PALLETE[sess_idx], 
			linewidth=1.8, 
			label=session_label
		)

	if plot_start_end_points:
		for sess_idx, session_poses in enumerate(sessions):
			positions = session_poses[:, :3, 3]
			if sess_idx == 0:
				start_label = 'Start Point'
				end_label = 'End Point'
			else:
				start_label = None
				end_label = None

			ax.plot(
				positions[0, 0], positions[0, 1], '*',
				c=PALLETE[5], markersize=10,
				label=start_label
			)
			ax.plot(
				positions[-1, 0], positions[-1, 1], '^', 
				c=PALLETE[5], markersize=8,
				label=end_label
			)
	
	if plot_legend:
		ax.legend(fontsize=10)
		# ax.set_xlabel('X [m]')
		# ax.set_ylabel('Y [m]')
	else:
		ax.set_xticks([])
		ax.set_yticks([])

	# ax.invert_xaxis()
	# ax.invert_yaxis()
	ax.tick_params(axis='x', labelsize=10)
	ax.tick_params(axis='y', labelsize=10)		
	ax.grid(True, linestyle='--', alpha=0.7)
	ax.set_aspect('equal')
	plt.tight_layout()
	plt.savefig(os.path.join(output_folder, f'merged_trajectory_{label}.png'), dpi=300, bbox_inches='tight')
	plt.savefig(os.path.join(output_folder, f'merged_trajectory_{label}.pdf'), dpi=300, bbox_inches='tight')
	plt.close()
	print(f"  Saved: merged_trajectory_{label}.png")

def main():
	"""Main entry point."""
	parser = argparse.ArgumentParser(description="Pose graph visualization for multiple merges")
	parser.add_argument('--data_folders', required=True, nargs='+',
					 help='Root directory containing s00000_aria_data folders')
	parser.add_argument('--labels', required=True, nargs='+',
					 help='Labels for the data folders')
	parser.add_argument('--output_folder', required=True,
					 help='Output folder for the plots')
	parser.add_argument('--result_folders', nargs='+', default=None,
					 help='Result directories containing merge_finalmap (e.g., s00000_xx_results_in_kf_spgo_cc_seqmatch_master)')
	parser.add_argument('--translation_threshold', type=float, default=5.0,
					 help='Translation threshold in meters for splitting sessions (default: 5.0)')
	parser.add_argument('--plot_start_end_points', action='store_true', default=False,
					 help='Plot start and end points of the merged trajectory')
	parser.add_argument('--plot_legend', action='store_true', default=False,
					 help='Plot legend for the plots')
	args = parser.parse_args()
	
	iqa_values_list = []
	for i, data_folder in enumerate(args.data_folders):
		all_iqa_values = []
		for seq_folder in sorted(os.listdir(data_folder)):
			iqa_data = read_timestamps(os.path.join(data_folder, seq_folder, 'iqa_data.txt'))
			for img_name, iqa_value in iqa_data.items():
				all_iqa_values.append(iqa_value[0])
		
		iqa_values_list.append(all_iqa_values)

	os.makedirs(args.output_folder, exist_ok=True)
	plot_iqa_data(iqa_values_list, args.labels, args.output_folder)

	for i, data_folder in enumerate(args.data_folders):
		all_poses_data = []
		for seq_folder in sorted(os.listdir(data_folder)):
			poses_file = os.path.join(data_folder, seq_folder, 'poses.txt')
			poses_data = load_poses_from_file(poses_file)
			if len(poses_data) > 0:
				all_poses_data.append(poses_data)

		plot_poses_data(all_poses_data, args.labels[i], args.output_folder, args.plot_start_end_points, args.plot_legend)

	if args.result_folders is not None:
		for i, result_folder in enumerate(args.result_folders):
			merge_finalmap_path = os.path.join(result_folder, 'merge_finalmap')
			if not os.path.exists(merge_finalmap_path):
				print(f"Warning: {merge_finalmap_path} does not exist, skipping...")
				continue
		
			sessions = load_merged_trajectory_data(merge_finalmap_path, args.translation_threshold)		
			if sessions is not None:
				plot_merged_trajectory(sessions, args.labels[i], args.output_folder, args.plot_start_end_points, args.plot_legend)

if __name__ == "__main__":
	main()
