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

# (global localization) Geometric Verification Threshold
REFINE_GV_SCORE_THRESHOLD = 100.0
MAX_LOSS = 10.0 
# (local localization) Confidence Map Threshold
RELIABLE_CONF_THRESHOLD = 0.1
REFINE_CONF_THRESHOLD = 0.6 # threshold to select good refinement: out-of-range image, wrong coarse localization; bad edges cluster just above 0.5 (e.g. merge step 40's 176-deg flip edge at conf 0.519)
assert RELIABLE_CONF_THRESHOLD < REFINE_CONF_THRESHOLD
# Same Place Threshold
TRANS_THRESHOLD = 7.5
ORI_THRESHOLD = 75.0
# (pose graph optimization) Minimum GNC weight for a loop edge to be accepted
# into the merged map. TLS weights are near 0 or near 1, so the midpoint is a
# safe cut.
PGO_INLIER_WEIGHT_THRESHOLD = 0.5

def is_same_place(nodeA, nodeB):
	dis_tsl, dis_angle = nodeA.compute_gt_distance(nodeB)
	return dis_tsl < TRANS_THRESHOLD and dis_angle < ORI_THRESHOLD

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

##### NOTE(gogojjh): This is not used in the map merging pipeline
def initialize_vpr_model(method, backbone, descriptors_dimension, device):
	# """Initialize and return the model."""
	# model = vpr_models.get_model(method, backbone, descriptors_dimension)
	# return model.eval().to(device)
	pass

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

def save_vis_edge_history(log_dir, db_submap, query_submap, edge_history):
	"""
	Save visualization of graph-based map with nodes and edges.
	Plot the trajectory onto the X-Z plane.
	"""
	fig, axes = plt.subplots(1, 4, figsize=(32, 8))
	# --- Helper to plot base map ---
	def plot_base(ax):
		for _, node in db_submap.nodes.items():
			ax.plot(node.trans_gt[0], node.trans_gt[1], 'bo', markersize=5)
			for edge in node.edges.values():
				next_node = edge[0]
				ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=0.5)
		for _, node in query_submap.nodes.items():
			ax.plot(node.trans_gt[0], node.trans_gt[1], 'ro', markersize=5)
			for edge in node.edges.values():
				next_node = edge[0]
				ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=0.5)

	# --- Edge filters for each subplot: edges still alive entering each stage ---
	# An edge's action holds only its FINAL state, so an edge is alive at a
	# stage iff it was not removed by that stage or any earlier one. Late
	# removals (removed_by_pgo, removed_by_low_connectivity) happen after CCM
	# and must still appear in the first three panels.
	edge_selector = [
		lambda action: True,
		lambda action: action != 'removed_by_gv',
		lambda action: action not in ('removed_by_gv', 'removed_by_ccm'),
		lambda action: action == 'added_by_vpr',
	]
	sub_titles = [
		"Edges: VPR",
		"Edges: VPR -> GV",
		"Edges: VPR -> GV -> CCM",
		"Edges: VPR -> GV -> CCM -> PGO"
	]
	precision_list, recall_list, edge_count_list = [], [], []
	
	# --- Generate binary labels ---
	y_true = [0] * len(edge_history)
	for query_idx in range(len(edge_history)):
		nodeB = query_submap.get_node(query_idx)
		for nodeA in db_submap.nodes.values():
			if is_same_place(nodeA, nodeB):
				y_true[query_idx] = 1
				break

	for subplot_idx, ax in enumerate(axes):
		plot_base(ax)
		num_edges = 0
		y_pred = [0] * len(edge_history)
		for key, value in edge_history.items():
			if not edge_selector[subplot_idx](value['action']): 
				continue
			db_idx, query_idx = key[0], key[1]
			nodeA = db_submap.get_node(db_idx)
			nodeB = query_submap.get_node(query_idx)
			ax.plot([nodeA.trans_gt[0], nodeB.trans_gt[0]],
					[nodeA.trans_gt[1], nodeB.trans_gt[1]],
					'g-', linewidth=2)
			num_edges += 1

			if y_true[query_idx]:
				if is_same_place(nodeA, nodeB):
					y_pred[query_idx] = 1 # true positive
				else:
					y_pred[query_idx] = 0 # false negative
			else:
				y_pred[query_idx] = 1 # false positive

		precision = precision_score(y_true, y_pred, zero_division=0)
		recall = recall_score(y_true, y_pred, zero_division=0)
		precision_list.append(precision)
		recall_list.append(recall)
		edge_count_list.append(num_edges)

		ax.grid(ls='--', color='0.7')
		ax.set_xlabel('X [m]')
		ax.set_ylabel('Y [m]')
		ax.axis('equal')
		title = f"{sub_titles[subplot_idx]}\nEdge Number: {num_edges}, Precision: {precision:.2f}, Recall: {recall:.2f}"
		ax.set_title(title)
	
	plt.tight_layout()
	plt.savefig(os.path.join(log_dir, f"edge_history.png"))

	return precision_list, recall_list, edge_count_list

def save_vis_kf_removal(log_dir, query_id, query_img, prob, db_id=None, db_img=None):
	(log_dir/"preds/kf_vis").mkdir(parents=True, exist_ok=True)
	if db_img is None:
		fig, ax = plt.subplots(1, 1, figsize=(4, 4))
		ax.imshow(query_img)
		ax.set_title(f'Remove Query Keyframe {query_id}')
		ax.axis('off')
		plt.savefig(str(log_dir/"preds/kf_vis"/f"kf_rejection_query_{query_id}_{prob:.3f}.jpg"))
		plt.close()
	else:
		fig, ax = plt.subplots(1, 2, figsize=(10, 4))
		ax[0].imshow(query_img)
		ax[0].set_title(f'Remove Query Keyframe {query_id}')
		ax[0].axis('off')
		ax[1].imshow(db_img)
		ax[1].set_title(f'Database Keyframe {db_id}')
		ax[1].axis('off')
		plt.savefig(str(log_dir/"preds/kf_vis"/f"kf_rejection_query_{query_id}_{db_id}_{prob:.3f}.jpg"))
		plt.close()

def save_vis_kf_replacement(log_dir, db_id, query_id, db_img, query_img, prob):
	(log_dir/"preds/kf_vis").mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	ax[0].imshow(db_img)
	ax[0].set_title(f'Remove Database Keyframe {db_id}')
	ax[0].axis('off')
	ax[1].imshow(query_img)
	ax[1].set_title(f'Query Keyframe {query_id}')
	ax[1].axis('off')
	plt.savefig(str(log_dir/"preds/kf_vis"/f"kf_replacement_{db_id}_{query_id}_{prob:.3f}.jpg"))
	plt.close()

def parse_arguments():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--dataset_root", type=str, required=True,
						help="Dataset root directory containing submap data and orders file")
	parser.add_argument("--output_root", type=str, default=None,
						help="Output root directory for merged results (default: same as dataset_root)")
	parser.add_argument("--scene", type=str, required=True,
						help="Scene name, e.g. s00000")
	parser.add_argument("--data_dir", type=str, default=None,
						help="Submap data directory name under dataset-root (default: <scene>_aria_data_390)")
	parser.add_argument("--order_index", type=int, required=True,
						help="Order index in orders file (0=in, 1=r0, ...)")
	parser.add_argument("--method", type=str, required=True,
						help="Method name for result directory naming, e.g. spgo_cc_seqmatch_master")
	parser.add_argument("--max_submaps", type=int, default=None,
						help="Maximum number of submaps to merge (default: all)")
	parser.add_argument("--image_size", type=int, default=None, nargs="+",
										help="Resizing shape for images (WxH). If a single int is passed, set the"
											 "longest edge of all images to this value, while keeping aspect ratio")

	parser.add_argument("--vpr_match_model", type=str, default="vpr_dp",
						help="single_match, seqslam, vpr_dp")
	parser.add_argument("--vpr_match_seq_len", type=int, default=10, help="Sequence length for VPR")
	parser.add_argument("--pose_estimation_method", type=str, default="master", help=f"{available_models}")
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="cuda (gpu) or cpu")
	# Ablation study flags for node culling factors
	parser.add_argument("--use_iqa", action="store_true", help="Use image quality assessment in node culling")
	parser.add_argument("--use_ig", action="store_true", help="Use information gain in node culling")
	parser.add_argument("--use_td", action="store_true", help="Use temporal difference in node culling")
	# Robust pose graph optimization
	parser.add_argument("--pgo_robust", type=str, default="gnc_tls",
		choices=["none", "huber", "gnc_tls", "gnc_gm"],
		help="Robust back-end for pose graph optimization")
	parser.add_argument("--pgo_gnc_barc_prob", type=float, default=0.99,
		help="Chi-squared probability for the GNC inlier cost threshold")
	parser.add_argument("--pgo_persistent_loops", action="store_true",
		help="Keep accepted inter-submap loop edges as loop factors across merge "
		     "steps so GNC can re-classify them instead of welding them into odometry")
	parser.add_argument("--pgo_min_loop_edges", type=int, default=1,
		help="Minimum number of refined loop edges required to merge a submap "
		     "this step; below this the merge is deferred (the submap keeps its "
		     "own anchor prior). 1 disables the guard. With a single edge PGO "
		     "zeroes the residual by rigid motion, so GNC cannot reject it")
	parser.add_argument("--pgo_loop_sigma_trans", type=float, default=0.1,
		help="Translation std [m] of an inter-submap loop factor; drives the GNC "
		     "TLS inlier threshold")
	parser.add_argument("--pgo_loop_sigma_rot", type=float, default=1.0,
		help="Rotation std [deg] of an inter-submap loop factor")
	parser.add_argument("--pgo_loop_conf_scaling", type=str, default="inverse",
		choices=["inverse", "none"],
		help="How matcher confidence scales a loop factor's sigma: 'inverse' "
		     "divides by conf (tightens high-confidence edges), 'none' keeps it flat")
	parser.add_argument("--pgo_seq_inlier_time_gap", type=float, default=0.0,
		help="Loop edges whose endpoint capture times differ by at most this "
		     "many seconds are declared GNC known inliers (sequentially "
		     "adjacent submaps overlap in time and are backed by odometry "
		     "continuity, while true revisit loops are minutes to days apart). "
		     "0 disables the exemption")
	# Logging and visualization flags
	parser.add_argument("--warning", action="store_true", help="Logging level")
	parser.add_argument("--viz", action="store_true", help="Flag to plot results")
	parser.add_argument("--rerun-viz", action="store_true", help="Record map merge Rerun visualization")
	parser.add_argument("--rerun-output", type=str, default=None, help="Output .rrd path for Rerun visualization")
	parser.add_argument("--rerun-viz-dir", type=str, default=None, help="Output directory for runtime Rerun visualization events")
	parser.add_argument("--rerun-image-format", type=str, default="jpg", choices=["jpg", "png"])
	parser.add_argument("--rerun-jpeg-quality", type=int, default=85)
	parser.add_argument("--rerun-dmatrix-format", type=str, default="png", choices=["png"])
	parser.add_argument("--rerun-max-match-images-per-step", type=int, default=200)
	parser.add_argument("--rerun-axis-scale", type=str, default="auto", choices=["auto"])
	args = parser.parse_args()

	return args
