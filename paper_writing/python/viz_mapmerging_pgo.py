#!/usr/bin/env python

# Usage: rosrun litevloc viz_mapmerging_pgo.py --folder data_litevloc/map_multisession_eval/ucl_campus_aria/s00000_results_r0_kf_spgo_cc_seqmatch/

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import argparse
import matplotlib.pyplot as plt
import gtsam
import numpy as np
from pathlib import Path

from python.utils.gtsam_pose_graph import PoseGraph
from python.utils.utils_setting_color_font import acquire_color_palette, setting_font

# Visualization setup
setting_font(fontsize=18, titlesize=18, legend_fontsize=18)
PALLETE = acquire_color_palette()

def plot_single_graph(ax, key_split, graph, values, title):
	"""Plot a single pose graph on a 3D axis."""
	result_poses = gtsam.utilities.allPose3s(values)
	x = [result_poses.atPose3(i).translation()[0] for i in range(result_poses.size())]
	y = [result_poses.atPose3(i).translation()[1] for i in range(result_poses.size())]
	z = [result_poses.atPose3(i).translation()[2] for i in range(result_poses.size())]
	
	# Plot nodes
	comp_graph_keys = PoseGraph.find_connected_components(graph)
	for comp_id, comp_keys in enumerate(comp_graph_keys):
		x_subgraph = [x[i] for i in comp_keys]
		y_subgraph = [y[i] for i in comp_keys]
		ax.plot(
			x_subgraph[::1], y_subgraph[::1], 'o', color='k', markersize=1.0, 
			linewidth=1.0
		)
	
	if 'Before' in title:
		x_newgraph = x[key_split:]
		y_newgraph = y[key_split:]
		ax.plot(
			x_newgraph[::1], y_newgraph[::1], 'o', color=PALLETE[1], markersize=4.5, zorder=2, 
			linewidth=2.5
		)

		for key in range(graph.size()):
			factor = graph.at(key)
			if isinstance(factor, gtsam.BetweenFactorPose3):
				key1, key2 = factor.keys()
				t1 = result_poses.atPose3(key1).translation()
				t2 = result_poses.atPose3(key2).translation()
				if key1 + 1 < key2 and key2 >= key_split:
					ax.plot(
						[t1[0], t2[0]], [t1[1], t2[1]], '-', color=PALLETE[0], alpha=0.7, zorder=100,
						linewidth=2.0
					)
		
	# ax.set_title(title, loc='center', fontsize=18, y=0.9)
	ax.tick_params(axis='x', labelsize=18)
	ax.tick_params(axis='y', labelsize=18)
	# ax.set_xlabel('X [m]')
	# ax.set_ylabel('Y [m]')
	ax.grid(True, linestyle='--', alpha=0.7)
	ax.axis('equal')

def main():
	"""Main entry point."""
	parser = argparse.ArgumentParser(description="Pose graph visualization for multiple merges")
	parser.add_argument('-f', '--folders', required=True, nargs='+',
						help='Root directory containing merge_xxx folders')
	args = parser.parse_args()
	
	import cv2

	for folder in args.folders:
		# Find and process all merge folders
		merge_folders = sorted(
			[os.path.join(folder, f) for f in os.listdir(folder) 
			if f.startswith('merge_') and os.path.isdir(os.path.join(folder, f))]
		)
		path_merge = Path(os.path.join(folder, 'viz_merge'))
		path_merge.mkdir(parents=True, exist_ok=True)
		
		last_refined_values = gtsam.NonlinearFactorGraph()
		png_files = []
		for idx, merge_folder in enumerate(merge_folders):
			print(f"Processing {os.path.basename(merge_folder)}...")

			preds_dir = os.path.join(merge_folder, 'preds')
			initial_g2o = os.path.join(preds_dir, 'initial_pose_graph.g2o')
			refined_g2o = os.path.join(preds_dir, 'refine_pose_graph.g2o')
			if not os.path.exists(refined_g2o):
				refined_g2o = initial_g2o
			
			# Load data
			initial_graph, initial_values = gtsam.readG2o(initial_g2o, is3D=True)
			refined_graph, refined_values = gtsam.readG2o(refined_g2o, is3D=True)
			key_split = last_refined_values.size()
			# print(key_split, refined_values.size())
			
			# Create figure
			fig = plt.figure(figsize=(6, 7))
			ax1 = fig.add_subplot(2, 1, 1)
			ax2 = fig.add_subplot(2, 1, 2)
			fig.subplots_adjust(hspace=0.00)
			plot_single_graph(ax1, key_split, initial_graph, initial_values, "Before Merging")
			plot_single_graph(ax2, key_split, refined_graph, refined_values, "After Merging")
			
			# Save and close
			plt.tight_layout()
			output_path = str(path_merge / f"pose_graph_mapmerge_{merge_folder.split('merge_')[-1]}.png")
			plt.savefig(output_path, bbox_inches='tight', dpi=300)
			png_files.append(output_path)
			output_path_pdf = output_path.replace('.png', '.pdf')
			plt.savefig(output_path_pdf, dpi=300)
			plt.close()
			# print(f"Saved visualization to: {output_path_pdf}")
			# input()

			last_refined_values = refined_values

		# Create a video from all PNGs (fps=5)
		if png_files:
			video_path = str(path_merge / "pose_graph_mapmerge.mp4")
			print(f"Creating video {video_path} ...")
			first_frame = cv2.imread(png_files[0])
			height, width, layers = first_frame.shape
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			video = cv2.VideoWriter(video_path, fourcc, 3, (width, height))
			for png_file in png_files:
				img = cv2.imread(png_file)
				if img is not None:
					video.write(img)
			video.release()
			print(f"Saved video: {video_path}")

if __name__ == "__main__":
	main()
