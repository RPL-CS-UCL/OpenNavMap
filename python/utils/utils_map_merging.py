import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../VPR-methods-evaluation'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../VPR-methods-evaluation/third_party/deep-image-retrieval'))

import argparse
# from datetime import datetime
import logging
import numpy as np
import pathlib

from estimator import get_estimator, available_models
from estimator.utils import to_numpy
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

from .utils_geom import convert_vec_to_matrix, compute_pose_error

RMSE_THRESHOLD = 3.0
VPR_MATCH_THRESHOLD = 0.90
REFINE_GV_SCORE_THRESHOLD = 100.0
MAX_LOSS = 10.0 

RELIABLE_CONF_THRESHOLD = 0.1
REFINE_CONF_THRESHOLD = 0.5 # threshold to select good refinement: out-of-range image, wrong coarse localization
assert RELIABLE_CONF_THRESHOLD < REFINE_CONF_THRESHOLD

def setup_logging(log_dir, stdout_level='info'):
	os.makedirs(log_dir, exist_ok=True)
	log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(
		level=getattr(logging, stdout_level.upper(), 'INFO'),
		format=log_format,
		handlers=[
			logging.FileHandler(os.path.join(log_dir, 'info.log')),
			logging.StreamHandler(sys.stdout)
		]
	)

def setup_log_environment(out_dir: pathlib.Path, args):
	"""Setup logging and directories."""
	out_dir.mkdir(parents=True, exist_ok=True)
	(out_dir / "seq").mkdir(parents=True, exist_ok=True)
	(out_dir / "preds").mkdir(parents=True, exist_ok=True)
	# start_time = datetime.now()
	# log_dir = os.path.join(out_dir, f"outputs_{args.pose_estimation_method}", start_time.strftime("%Y-%m-%d_%H-%M-%S"))
	# setup_logging(log_dir, stdout_level="info")
	# logging.info(" ".join(sys.argv))
	# logging.info(f"Arguments: {args}")
	# logging.info(f"Testing with {args.pose_estimation_method} with image size {args.image_size}")
	# logging.info(f"The outputs are being saved in {log_dir}")
	# os.makedirs(os.path.join(log_dir, "preds"))
	# os.system(f"rm {os.path.join(out_dir, f'outputs_{args.pose_estimation_method}', 'latest')}")
	# os.system(f"ln -s {log_dir} {os.path.join(out_dir, f'outputs_{args.pose_estimation_method}', 'latest')}")
	return out_dir

# def initialize_vpr_model(method, backbone, descriptors_dimension, device):
# 	"""Initialize and return the model."""
# 	model = vpr_models.get_model(method, backbone, descriptors_dimension)
# 	return model.eval().to(device)

def initialize_pose_estimator(model, device):
	"""Initialize and return the model."""
	return get_estimator(model, device=device)

"""
Visualization
"""
def save_vis_vpr(log_dir, db_submap, query_submap, query_submap_id, preds, suffix=''):
	db_images = [to_numpy(node.rgb_image.permute(1, 2, 0)) for _, node in db_submap.nodes.items()]
	query_images = [to_numpy(node.rgb_image.permute(1, 2, 0)) for _, node in query_submap.nodes.items()]
	fig, axes = plt.subplots(preds.shape[0], preds.shape[1]+1, figsize=(20, 2 * (preds.shape[1]+1)))
	for query_id in range(preds.shape[0]):
		axes[query_id, 0].imshow(query_images[query_id])
		axes[query_id, 0].set_title(f'Q{query_id}')
		for i in range(preds.shape[1]):
			axes[query_id, i + 1].imshow(db_images[preds[query_id, i]])
			axes[query_id, i + 1].set_title(f'DB{preds[query_id, i]}')
	if suffix == '':
		plt.savefig(os.path.join(log_dir, f"results_{query_submap_id}_vpr.png"))
	else:
		plt.savefig(os.path.join(log_dir, f"results_{suffix}_{query_submap_id}_vpr.png"))

def save_vis_pose_graph(log_dir, db_submap, query_submap, query_submap_id, edges_nodeA_to_nodeB, suffix=''):
	"""
	Save visualization of graph-based map with nodes and edges.
	Plot the trajectory onto the X-Z plane.
	"""
	fig, ax = plt.subplots(figsize=(10, 10))
	
	# Plot submap
	logging.debug('Plot db_submap')
	for node_id, node in db_submap.nodes.items():
		ax.plot(node.trans_gt[0], node.trans_gt[1], 'ko', markersize=5)
		# ax.text(node.trans_gt[0], node.trans_gt[1], f'DB{node_id}', fontsize=12, color='k')
		for edge in node.edges.values():
			next_node = edge[0]
			ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=1)

	for node_id, node in query_submap.nodes.items():			
		ax.plot(node.trans_gt[0], node.trans_gt[1], 'bo', markersize=5)
		ax.text(node.trans_gt[0], node.trans_gt[1], f'Q{node_id}', fontsize=12, color='k')		
		for edge in node.edges.values():
			next_node = edge[0]
			ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=1)
	
	# Plot connections
	num_cor_loop = 0
	str_title = f"Pose Graph"
	for edge in edges_nodeA_to_nodeB:
		nodeA, nodeB, T_rel, score = edge[:4]
		# Identify correct and wrong connections
		if 'coarse' in suffix:
			dis_tsl, dis_angle = nodeA.compute_gt_distance(nodeB)
			if dis_tsl < 7.5:
				num_cor_loop += 1
				ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]], [nodeA.trans_gt[1], nodeB.trans_gt[1]], 'g-', linewidth=2)
				ax.text(nodeB.trans_gt[0], nodeB.trans_gt[1]+0.4, f'P={score:.1f}', fontsize=12, color='k')
			else:
				ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]], [nodeA.trans_gt[1], nodeB.trans_gt[1]], 'r-', linewidth=2)
				ax.text(nodeB.trans_gt[0], nodeB.trans_gt[1]+0.4, f'P={score:.1f}', fontsize=12, color='k')
			str_title = f"Pose Graph: Find {num_cor_loop} Correct Loops/{len(edges_nodeA_to_nodeB)} (7.5m)"

		elif 'refine' in suffix:
			T_nodeA_gt = convert_vec_to_matrix(nodeA.trans_gt, nodeA.quat_gt, 'xyzw')
			T_nodeB_gt = convert_vec_to_matrix(nodeB.trans_gt, nodeB.quat_gt, 'xyzw')
			T_rel_gt = np.linalg.inv(T_nodeA_gt) @ T_nodeB_gt
			dis_tsl, dis_angle = compute_pose_error(T_rel, T_rel_gt, 'matrix')
			if dis_tsl < 3.0:
				num_cor_loop += 1
				ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]], [nodeA.trans_gt[1], nodeB.trans_gt[1]], 'g-', linewidth=2)
				ax.text(nodeB.trans_gt[0], nodeB.trans_gt[1]+0.4, f'P={score:.1f}', fontsize=12, color='k')
			else:
				ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]], [nodeA.trans_gt[1], nodeB.trans_gt[1]], 'r-', linewidth=2)
				ax.text(nodeB.trans_gt[0], nodeB.trans_gt[1]+0.4, f'P={score:.1f}', fontsize=12, color='k')
			str_title = f"Pose Graph: Find {num_cor_loop} Correct Loops/{len(edges_nodeA_to_nodeB)} (3.0m)"
	
	ax.grid(ls='--', color='0.7')
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	plt.axis('equal')
	plt.title(str_title) 
	if suffix == '':
		plt.savefig(os.path.join(log_dir, f"results_{query_submap_id}_posegraph.png"))
	else:
		plt.savefig(os.path.join(log_dir, f"results_{suffix}_{query_submap_id}_posegraph.png"))

def save_vis_edge_history(log_dir, db_submap, query_submap, query_submap_id, edge_history):
	"""
	Save visualization of graph-based map with nodes and edges.
	Plot the trajectory onto the X-Z plane.
	"""
	fig, axes = plt.subplots(1, 3, figsize=(24, 8))
	# --- Helper to plot base map ---
	def plot_base(ax):
		for _, node in db_submap.nodes.items():
			ax.plot(node.trans_gt[0], node.trans_gt[1], 'bo', markersize=5)
			for edge in node.edges.values():
				next_node = edge[0]
				ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=0.5)
		for _, node in query_submap.nodes.items():
			ax.plot(node.trans_gt[0], node.trans_gt[1], 'bo', markersize=5)
			for edge in node.edges.values():
				next_node = edge[0]
				ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=0.5)

	# --- Edge filters for each subplot ---
	edge_selector = [
		lambda value: 'added_by_vpr' in value or 'removed_by_gv' in value or 'removed_by_ccm' in value,
		lambda value: 'added_by_vpr' in value or 'removed_by_ccm' in value,  # ignore removed_by_gv
		lambda value: 'added_by_vpr' in value,  # ignore both removed_by_gv and removed_by_ccm
	]
	sub_titles = [
		"Edges: VPR",
		"Edges: VPR -> GV",
		"Edges: VPR -> GV -> CCM"
	]

	# --- Threshold for true positive ---
	trans_threshold = 7.5
	ori_threshold = 75.0
	
	# --- Generate binary labels ---
	y_true = [0] * len(edge_history)
	for query_idx in range(len(edge_history)):
		nodeB = query_submap.get_node(query_idx)
		for nodeA in db_submap.nodes.values():
			dis_tsl, dis_angle = nodeA.compute_gt_distance(nodeB)
			if dis_tsl < trans_threshold and dis_angle < ori_threshold:
				y_true[query_idx] = 1
				break

	for subplot_idx, ax in enumerate(axes):
		plot_base(ax)
		
		num_edges = 0
		y_pred = [0] * len(edge_history)
		for key, value in edge_history.items():
			if not edge_selector[subplot_idx](value): 
				continue
			db_idx, query_idx = key[0], key[1]
			nodeA = db_submap.get_node(db_idx)
			nodeB = query_submap.get_node(query_idx)
			dis_tsl, dis_angle = nodeA.compute_gt_distance(nodeB)
			ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]],
					[nodeA.trans_gt[1], nodeB.trans_gt[1]],
					'g-', linewidth=2)
			num_edges += 1

			if y_true[query_idx]:
				if dis_tsl < trans_threshold and dis_angle < ori_threshold:
					y_pred[query_idx] = 1 # true positive
				else:
					y_pred[query_idx] = 0 # false negative
			else:
				y_pred[query_idx] = 1 # false positive

		precision = precision_score(y_true, y_pred, zero_division=0)
		recall = recall_score(y_true, y_pred, zero_division=0)
		ax.grid(ls='--', color='0.7')
		ax.set_xlabel('X [m]')
		ax.set_ylabel('Y [m]')
		ax.axis('equal')
		title = f"{sub_titles[subplot_idx]}\nEdge Number: {num_edges}, Precision: {precision:.2f}, Recall: {recall:.2f}"
		ax.set_title(title)
	
	plt.tight_layout()
	plt.savefig(os.path.join(log_dir, f"edge_history.png"))

def save_vis_kf_removal(log_dir, img_id, rgb_img, prob):
	(log_dir/"preds/kf_vis").mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(1, 1, figsize=(4, 4))
	ax.imshow(rgb_img)
	ax.set_title(f'Remove New Keyframe {img_id}')
	ax.axis('off')
	plt.savefig(str(log_dir/"preds/kf_vis"/f"kf_rejection_query_{img_id}_{prob:.3f}.jpg"))
	plt.close()

def save_vis_kf_replacement(log_dir, img0_id, img1_id, rgb_img0, rgb_img1, prob):
	(log_dir/"preds/kf_vis").mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	ax[0].imshow(rgb_img0)
	ax[0].set_title(f'Old Keyframe {img0_id}')
	ax[0].axis('off')
	ax[1].imshow(rgb_img1)
	ax[1].set_title(f'New Keyframe {img1_id}')
	ax[1].axis('off')
	plt.savefig(str(log_dir/"preds/kf_vis"/f"kf_replacement_{img0_id}_{img1_id}_{prob:.3f}.jpg"))
	plt.close()

def save_query_result(log_dir, query_result_info, query_submap_id):
	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	for i in range(query_result_info.shape[0]):
		query_id, prob, score, succ = i, query_result_info[i, 0], query_result_info[i, 1], query_result_info[i, 2]
		if prob < VPR_MATCH_THRESHOLD:
			ax[0].bar(query_id, prob, width=0.6, alpha=0.7, label='VPR Score', color='g')
		else:
			ax[0].bar(query_id, prob, width=0.6, alpha=0.7, label='VPR Score', color='r')
		if score > REFINE_CONF_THRESHOLD:
			ax[1].bar(query_id, score, width=0.6, alpha=0.7, label='Edge Score/Loss', color='g')
		else:
			ax[1].bar(query_id, score, width=0.6, alpha=0.7, label='Edge Score/Loss', color='r')
	ax[0].grid(ls='--', color='0.7')
	ax[0].set_title('VPR Score/Loss')
	ax[1].grid(ls='--', color='0.7')
	ax[1].set_title('Edge Score (Green: High Score. Red: Low Score)')
	fig.tight_layout()
	plt.savefig(os.path.join(log_dir, f"preds/results_{query_submap_id}_query_result.png"))

def parse_arguments():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--input_submap_path", type=str, default=None, nargs="+", help="Path to input submaps")
	parser.add_argument("--output_map_path", type=str, default=None, help="Path to output final map")
	parser.add_argument("--image_size", type=int, default=None, nargs="+",
										help="Resizing shape for images (WxH). If a single int is passed, set the"
											 "longest edge of all images to this value, while keeping aspect ratio")

	parser.add_argument("--vpr_match_model", type=str, default="sequence_match", 
						help="single_match, topo_filter, sequence_match, sequence_match_ransac, sequence_match_adaptive")
	parser.add_argument("--vpr_match_seq_len", type=int, default=10, help="Sequence length for VPR")
	parser.add_argument("--pose_estimation_method", type=str, default="master", help=f"{available_models}")
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="cuda (gpu) or cpu")
	parser.add_argument("--color_correct", action="store_true", help="Flag to correct collor temperature")
	parser.add_argument("--prune_keyframe_forward", action="store_true", 
					 	help="Flag to prune keyframes by checking quality and information gain of newly inserted keyframes")
	parser.add_argument("--prune_keyframe_backward", action="store_true", 
					 	help="Flag to prune keyframes by checking quality and information gain of old keyframes")
	parser.add_argument("--warning", action="store_true", help="Logging level")
	parser.add_argument("--viz", action="store_true", help="Flag to plot results")
	args = parser.parse_args()

	return args

