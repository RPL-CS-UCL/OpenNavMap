#! /usr/bin/env python

import _bootstrap_imports  # noqa: F401

import torch
import pathlib
import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from typing import List, Tuple
from PIL import Image
import cv2

from map_manager import MapManager
from utils_map_merging import initialize_pose_estimator
from utils.utils_geom import convert_vec_to_matrix
from utils.utils_image import to_numpy
from utils.utils_setting_color_font import setting_font, acquire_color_palette

setting_font(fontsize=16, titlesize=16, legend_fontsize=16)
PALLETE = acquire_color_palette()

def parse_arguments():
	"""Parse command line arguments"""
	parser = argparse.ArgumentParser(
		description='Visualize local scene recovery from a map using spatial bounding box',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	
	parser.add_argument("--map_path", type=str, required=True, help="Path to the map folder")
	parser.add_argument("--x_min", type=float, required=True, help="Minimum X coordinate")
	parser.add_argument("--x_max", type=float, required=True, help="Maximum X coordinate")
	parser.add_argument("--y_min", type=float, required=True, help="Minimum Y coordinate")
	parser.add_argument("--y_max", type=float, required=True, help="Maximum Y coordinate")
	parser.add_argument("--pose_estimation_method", type=str, default="master", help="Pose estimation method (master, duster, etc.)")
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
	parser.add_argument("--image_size", type=int, nargs="+", default=[512, 288], help="Image resize dimensions (width, height)")
	parser.add_argument("--output_dir", type=str, default="./viz_local_scene_recovery", help="Output directory for visualizations")
	parser.add_argument("--thumbnail_size", type=int, default=200, help="Size for image thumbnails in the plot")
	parser.add_argument("--opts", type=str, default="", help="Options for the plot (none, others)")
	args = parser.parse_args()

	return args

def load_map(map_path: pathlib.Path, image_size: Tuple[int, int]):
	"""Load the map from a folder"""
	graph_configs = {
		'odom': {},
		'trav': {},
		'covis': {
			'resize': tuple(image_size),
			'depth_scale': 0.0,
			'load_rgb': True,
			'load_depth': False,
			'normalized': False,
		},
	}
	
	map_manager = MapManager(map_path, map_id=0)
	map_manager.load_graphs(graph_configs)
	
	logging.info(f"Loaded map from {map_path}")
	logging.info(f"Map info:\n{map_manager}")
	
	return map_manager

def find_nodes_in_bbox(
	map_manager: MapManager, 
	x_range: Tuple[float, float],
	y_range: Tuple[float, float]
):
	"""Find all nodes within the specified bounding box"""
	nodes = list(map_manager.covis.nodes.values())
	
	selected_nodes = []
	for node in nodes:
		x, y = node.trans[0], node.trans[1]
		if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
			selected_nodes.append(node)
	
	logging.info(f"Found {len(selected_nodes)} nodes in bounding box x:[{x_range[0]}, {x_range[1]}], y:[{y_range[0]}, {y_range[1]}]")
	for i, node in enumerate(selected_nodes):
		date_str = datetime.fromtimestamp(node.time).strftime('%Y-%m-%d')
		logging.info(f"  Node {i}: ID={node.id}, pos=({node.trans[0]:.2f}, {node.trans[1]:.2f}, {node.trans[2]:.2f}), time={date_str}")
	
	return selected_nodes

def perform_pose_estimation(
	estimator,
	map_root: pathlib.Path,
	db_nodes: List,
	query_node,
	est_opts: dict
):
	"""Perform pose estimation using the provided nodes"""
	db_poses = [
		torch.from_numpy(convert_vec_to_matrix(node.trans, node.quat, 'xyzw')) 
		for node in db_nodes
	]
	db_intrs = [{
		'K': torch.from_numpy(node.raw_K),
		'im_size': torch.from_numpy(node.raw_img_size)
	} for node in db_nodes]
	
	query_intr = {
		'K': torch.from_numpy(query_node.raw_K),
		'im_size': torch.from_numpy(query_node.raw_img_size)
	}
	
	try:
		result = estimator(
			map_root,
			[node.rgb_image for node in db_nodes],
			query_node.rgb_image,
			db_poses, db_intrs,
			query_intr,
			est_opts
		)
		
		im_pose = result["im_pose"]
		if im_pose is None:
			raise ValueError("Estimated pose is None")
		if np.isnan(im_pose).any():
			raise ValueError("Estimated pose contains NaN")
		
		logging.info(f"Estimated camera pose (world frame): {im_pose[:3, 3]}")
		return result, estimator
		
	except Exception as e:
		logging.error(f"Pose estimation failed: {str(e)}")
		raise

def timestamp_to_datetime(timestamp: float):
	"""Convert Unix timestamp to datetime string"""
	dt = datetime.fromtimestamp(timestamp)
	return dt.strftime('%Y-%m-%d %H:%M:%S')

def draw_orientation_arrow(ax, transform, length, style):
	"""Draws orientation arrow for a single camera"""
	start = transform[:3, 3]
	direction = transform[:3, :3] @ np.array([0, 0, length])
	
	head_width = style['head_width']
	head_length = style['head_length']
	zorder = style['zorder']
	fc = style['fc']
	ax.arrow(start[0], start[1], direction[0], direction[1], 
			 head_width=head_width*0.8, head_length=head_length*1.2,
			 width=head_width*0.15, fc=fc, ec=fc, zorder=zorder, alpha=1.0)

def plot_all_nodes_with_timestamps(
	map_manager: MapManager,
	output_dir: pathlib.Path,
	x_range: Tuple[float, float],
	y_range: Tuple[float, float],
	selected_nodes: List = None,
	opts: str = "none"
):
	"""Plot all map nodes with their timestamps as text annotations and orientation arrows,
	reordering nodes by timestamp and using different colors for each."""
	fig, ax = plt.subplots(figsize=(8, 7.5))

	all_nodes = list(map_manager.covis.nodes.values())

	from collections import defaultdict
	node_groups = defaultdict(list)
	dates = []
	for node in all_nodes:
		dt = datetime.fromtimestamp(node.time)
		date = (dt.year, dt.month, dt.day)
		node_groups[date].append(node)
		if date not in dates:
			dates.append(date)

	color_list = PALLETE if len(PALLETE) >= len(dates) else PALLETE * (len(dates) // len(PALLETE) + 1)

	all_positions = np.array([node.trans for node in all_nodes])
	bounds = np.max(all_positions[:, :2], axis=0) - np.min(all_positions[:, :2], axis=0)
	max_bound = np.max(bounds) / 4	
	arrow_length = max(0.5, max_bound / 10)
	head_size = max(0.2, max_bound / 20)
	for idx, date in enumerate(dates):
		nodes = node_groups[date]
		color = color_list[idx]
		arrow_style = {
			'head_width': head_size * 1.0,
			'head_length': head_size * 1.0,
			'fc': color,
			'zorder': len(dates) - idx,
		}
		for node in nodes:
			x, y = node.trans[0], node.trans[1]
			transform = convert_vec_to_matrix(node.trans, node.quat, 'xyzw')
			draw_orientation_arrow(ax, transform, arrow_length, arrow_style)

	for idx, date in enumerate(dates):
		label_str = "%04d-%02d-%02d" % date
		ax.plot([], [], color=color_list[idx], marker='>', linestyle='None', label=label_str)
	
	# import matplotlib.patches as patches
	# rect_width = x_range[1] - x_range[0]
	# rect_height = y_range[1] - y_range[0]
	# rect = patches.Rectangle(
	# 	(x_range[0], y_range[0]), rect_width, rect_height,
	# 	linewidth=2, edgecolor=PALLETE[1], facecolor='none', zorder=15
	# )
	# ax.add_patch(rect)

	ax.legend(fontsize=22)
	ax.set_xlim((-20, 25))
	ax.set_ylim((-25, 1))
	ax.tick_params(axis='x', labelsize=20)
	ax.tick_params(axis='y', labelsize=20)
	# ax.set_xlabel('X [m]', fontsize=14)
	# ax.set_ylabel('Y [m]', fontsize=14)
	# ax.set_title('Map Nodes with Timestamps', fontsize=16, fontweight='bold')
	ax.grid(True, alpha=0.7, linestyle='--')
	ax.set_aspect('equal')	# import matplotlib.patches as patches
	# rect_width = x_range[1] - x_range[0]
	# rect_height = y_range[1] - y_range[0]
	# rect = patches.Rectangle(
	# 	(x_range[0], y_range[0]), rect_width, rect_height,
	# 	linewidth=2, edgecolor=PALLETE[1], facecolor='none', zorder=15
	# )
	# ax.add_patch(rect)

	plt.tight_layout()
	plt.savefig(str(output_dir / f'map_nodes_with_timestamps_{opts}.png'), dpi=300, bbox_inches='tight')
	plt.savefig(str(output_dir / f'map_nodes_with_timestamps_{opts}.pdf'), dpi=300, bbox_inches='tight')
	plt.close()

def plot_selected_nodes_with_images(
	selected_nodes: List,
	output_dir: pathlib.Path,
	opts: str = "none"
):
	"""Plot only the selected node images in a grid layout"""
	num_images = len(selected_nodes)
	
	cols = min(3, num_images)
	rows = (num_images + cols - 1) // cols
	
	fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
	
	if num_images == 1:
		axes = np.array([axes])
	axes = axes.flatten() if num_images > 1 else axes
	
	for i, node in enumerate(selected_nodes):
		ax = axes[i] if num_images > 1 else axes[0]
		img = to_numpy(node.rgb_image.permute(1, 2, 0))
		h, w = img.shape[:2]
		img_resized = cv2.resize(img, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_LINEAR)
		ax.imshow(img_resized)
		
		time_str = timestamp_to_datetime(node.time)
		title = f'Node {node.id} ({i+1})\n{time_str}\n'
		title += f'Pos: ({node.trans[0]:.1f}, {node.trans[1]:.1f}, {node.trans[2]:.1f})'
		ax.set_title(title, fontsize=10)
		ax.axis('off')
	
	for i in range(num_images, len(axes)):
		axes[i].axis('off')
	
	plt.tight_layout()
	plt.savefig(str(output_dir / f'selected_node_images_{opts}.png'), dpi=300, bbox_inches='tight')
	plt.savefig(str(output_dir / f'selected_node_images_{opts}.pdf'), dpi=300, bbox_inches='tight')
	logging.info(f"Saved selected node images to {output_dir / f'selected_node_images_{opts}.png'}")
	plt.close()

def main():
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[logging.StreamHandler()]
	)
	
	args = parse_arguments()
	
	output_dir = pathlib.Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	map_path = pathlib.Path(args.map_path)
	map_manager = load_map(map_path, args.image_size)
	
	x_range = (args.x_min, args.x_max)
	y_range = (args.y_min, args.y_max)
	logging.info(f"Bounding box - X: [{args.x_min}, {args.x_max}], Y: [{args.y_min}, {args.y_max}]")
	
	selected_nodes = find_nodes_in_bbox(map_manager, x_range, y_range)
	
	if len(selected_nodes) == 0:
		logging.error("No nodes found in the specified bounding box!")
		return
	
	plot_all_nodes_with_timestamps(
		map_manager,
		output_dir,
		x_range=x_range,
		y_range=y_range,
		selected_nodes=selected_nodes,
		opts=args.opts
	)
	
	plot_selected_nodes_with_images(
		selected_nodes,
		output_dir,
		opts=args.opts
	)

	logging.info(f"Initializing pose estimator: {args.pose_estimation_method}")
	pose_estimator = initialize_pose_estimator(
		args.pose_estimation_method,
		args.device
	)
	pose_estimator.verbose = True
	
	est_opts = {
		'known_extrinsics': False,
		'known_intrinsics': False,
		'niter': 300,
		'two_stage_opt_niter': 0,
		'crop_image_to_database': False
	}
	
	query_node = selected_nodes[1]
	db_nodes = selected_nodes[1:]
	
	logging.info(f"\nPerforming pose estimation:")
	logging.info(f"  Database nodes: {[n.id for n in db_nodes]}")
	logging.info(f"  Query node: {query_node.id}")	
	result, estimator = perform_pose_estimation(
		pose_estimator,
		map_manager.map_root,
		db_nodes,
		query_node,
		est_opts
	)	
	estimator.show_reconstruction()

	logging.info(f"\n{'='*70}")
	logging.info(f"Visualization complete!")
	logging.info(f"Results saved to: {output_dir}")
	logging.info(f"{'='*70}")


if __name__ == '__main__':
	main()

