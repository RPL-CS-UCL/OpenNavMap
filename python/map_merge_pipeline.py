#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pathlib
import numpy as np
import logging
import gtsam
import matplotlib
from tqdm import tqdm
import pathlib
from typing import List, Tuple, Dict
from codetiming import Timer

from utils.utils_vpr_method import initialize_match_model
from utils_map_merging import *
from utils.utils_geom import convert_vec_to_matrix, convert_matrix_to_vec, compute_pose_error
from utils.utils_geom import convert_vec_gtsam_pose3, convert_matrix_gtsam_pose3
from utils.gtsam_pose_graph import PoseGraph
from utils.utils_image import to_numpy
from utils.utils_image_matching_method import save_visualization
from benchmark_kf_selection.metric.landmark_selector import LandmarkSelector

from map_manager import MapManager
from image_graph import ImageGraph
from image_node import ImageNode
from visualization.map_merge_runtime_event_recorder import MapMergeRuntimeEventRecorder

from colorama import Fore, init
init(autoreset=True)

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):	matplotlib.use("Agg")

def update_edge_history(edge_history, key, action: str, db_row=None, query_row=None):
	if key not in edge_history:
		assert db_row is not None or query_row is not None, "db_row and query_row must be provided"
		value = {'action': action, 'db_row': db_row, 'query_row': query_row}
		edge_history[key] = value
		logging.warning(f"Add Edge history: DB {key[0]} -> Query {key[1]}: {value}")
	else:
		value = edge_history[key]
		value['action'] = action
		logging.warning(f"Update Edge history: DB {key[0]} -> Query {key[1]}: {value}")

class MergePipeline:
	def __init__(self, args, log_dir: pathlib.Path):
		self.args = args
		self.log_dir = log_dir
		self.frame_id_map = 'map'

		self.submaps = []
		self.id_offset = 0
		self.runtime_viz_recorder = None
		self.runtime_merge_step = -1

		self.lm_selector = LandmarkSelector()

		self.graph_configs = {
			'odom': {},
			'trav': {},
			'covis': {
				'resize': self.args.image_size,
				'depth_scale': 0.0,
				'load_rgb': True,
				'load_depth': False,
				'normalized': False,
			},
		}

		self.est_opts = {
			'known_extrinsics': True,
			'known_intrinsics': False,
			'niter': 300,
			'two_stage_opt_niter': 50,
			'crop_image_to_database': False
		}

	def init_vpr_match_model(self):
		self.vpr_match_model = initialize_match_model(self.args.vpr_match_model, self.args.vpr_match_seq_len)		
		logging.info(f"VPR Match Model: {self.args.vpr_match_model}")

	def init_pose_estimator(self):
		self.pose_estimator = initialize_pose_estimator(
			self.args.pose_estimation_method, 
			self.args.device
		)
		self.pose_estimator.verbose = False
		logging.info(f"Pose Estimator: {self.args.pose_estimation_method}")

	def read_map_from_file(self):
		for submap_path in self.args.input_submap_path:
			submap_id = len(self.submaps)
			submap = MapManager(pathlib.Path(submap_path), submap_id)
			submap.load_graphs(merger.graph_configs)
			self.submaps.append(submap)

			logging.info(f"Loaded {submap.map_id} from {submap_path} with info: {submap}")

	def create_pose_graph_from_map(
		self, 
		graph_odom_a,     # The odometry graph 
		graph_odom_b,     # The odometry graph 
		inter_edges_covis # inter_edges_covis own the same node id with odom
	):
		# Set basic std for factors
		prior_sigma = np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3) / 100
		odom_sigma = np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3) / 10
		loop_sigma = np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3)
		pose_graph = PoseGraph()
		I_pose3 = convert_matrix_gtsam_pose3(np.eye(4))

		# Create a pose graph from graph_odom_a/b by adding internal edges of graph_odom_a/b
		for graph, offset in [(graph_odom_a, 0), (graph_odom_b, self.id_offset)]:
			for node in graph.nodes.values():
				pose = convert_vec_gtsam_pose3(node.trans, node.quat)
				pose_graph.add_init_estimate(node.id + offset, pose)
				for next_node in (edge[0] for edge in node.edges.values() if node.id < edge[0].id):
					next_pose = convert_vec_gtsam_pose3(next_node.trans, next_node.quat)
					pose_graph.add_odometry_factor(
						node.id + offset, pose,
						next_node.id + offset, next_pose,
						odom_sigma
					)
		
		# Add the loop factor
		for edge in inter_edges_covis:
			nodeA, nodeB, T_AB, conf = edge[:4]
			trans, quat = convert_matrix_to_vec(T_AB)
			next_pose3 = convert_vec_gtsam_pose3(trans, quat)
			update_loop_sigma = loop_sigma / conf
			pose_graph.add_odometry_factor(
				nodeA.id, I_pose3, 
				nodeB.id+self.id_offset, next_pose3, 
				update_loop_sigma
			)

		# Add prior factor to each disconnected subgraph
		subgraph_keys = PoseGraph.find_connected_components(pose_graph.get_factor_graph())
		for graph_id, keys in enumerate(subgraph_keys):
			curr_pose3 = pose_graph.get_initial_estimate().atPose3(keys[0])
			pose_graph.add_prior_factor(keys[0], curr_pose3, prior_sigma)
			print(f"Add prior: {keys[0]} to the {graph_id} subgraph with node number {len(keys)}")

		return pose_graph, subgraph_keys

	def merge_and_update_submaps(
		self, 
		submap_a: MapManager, 
		submap_b: MapManager, 
		estimate_pose
	):
		"""Merge two submaps and update all relevant graphs"""
		submap_b.adjust_all_ids(self.id_offset)
		
		submap_a.update_node_poses(estimate_pose)
		submap_b.update_node_poses(estimate_pose)

		submap_a.covis.copy_sensor_data(submap_b.covis)
		submap_a.merge_graphs_from(submap_b)

		logging.warning(f"Merged map info - {submap_a}")


def _node_payload(graph, node):
	image_path = graph.map_root / node.rgb_img_name if getattr(node, 'rgb_img_name', None) else None
	payload = {
		"node_id": node.id,
		"time": getattr(node, 'time', None),
		"position": node.trans,
		"quat_xyzw": node.quat,
		"rgb_img_name": getattr(node, 'rgb_img_name', None),
		"rgb_img_path": str(image_path) if image_path is not None else None,
	}
	for attr_name in ("raw_K", "K", "raw_img_size", "img_size"):
		if hasattr(node, attr_name):
			value = getattr(node, attr_name)
			if value is not None:
				payload[attr_name] = np.asarray(value).tolist()
	return payload


def _iter_unique_edges(graph):
	for node in graph.nodes.values():
		for neighbor, weight in node.edges.values():
			if node.id < neighbor.id:
				yield node, neighbor, weight


def _scene_confidence_maps(scene):
	"""Return confidence maps from current pose-estimator scenes with legacy fallback."""
	if hasattr(scene, 'conf_i') and hasattr(scene, 'conf_j'):
		return scene.conf_i, scene.conf_j
	return scene.weight_i, scene.weight_j


def _record_graph_edges(recorder, merge_step: int, submap_id: int, edge_type: str, graph):
	if recorder is None or graph is None:
		return
	for node_a, node_b, weight in _iter_unique_edges(graph):
		recorder.record_event(
			merge_step=merge_step,
			stage="graph_edge_observed",
			event_type=f"{edge_type}_edge_observed",
			submap_id=submap_id,
			keyframe_id=max(node_a.id, node_b.id),
			payload={
				"edge_type": edge_type,
				"nodeAid": node_a.id,
				"nodeBid": node_b.id,
				"weight": weight,
				"position_a": node_a.trans,
				"position_b": node_b.trans,
			},
		)


def _record_submap_loaded(merger: MergePipeline, merge_step: int, submap: MapManager):
	recorder = merger.runtime_viz_recorder
	if recorder is None:
		return
	recorder.record_event(
		merge_step=merge_step,
		stage="submap_loaded",
		event_type="submap_loaded",
		submap_id=submap.map_id,
		keyframe_id=None,
		payload={
			"map_root": str(submap.map_root),
			"num_covis_nodes": submap.covis.get_num_node() if submap.covis else 0,
			"num_odom_nodes": submap.odom.get_num_node() if submap.odom else 0,
		},
	)
	if submap.covis:
		for node in submap.covis.nodes.values():
			recorder.record_event(
				merge_step=merge_step,
				stage="vio_node_observed",
				event_type="vio_node_observed",
				submap_id=submap.map_id,
				keyframe_id=node.id,
				payload=_node_payload(submap.covis, node),
			)
	_record_graph_edges(recorder, merge_step, submap.map_id, "odom", submap.odom)
	_record_graph_edges(recorder, merge_step, submap.map_id, "covis", submap.covis)
	_record_graph_edges(recorder, merge_step, submap.map_id, "trav", submap.trav)


def _plot_runtime_dmatrix_panels(
	D_all: np.ndarray,
	panels: List[Tuple[str, List[Tuple[int, int]], tuple]],
	output_path: pathlib.Path,
	figsize: Tuple[float, float],
) -> None:
	"""Plot D-matrix panels using the paper VPR visualization style."""
	import matplotlib.pyplot as plt
	from utils.utils_setting_color_font import (
		acquire_color_palette,
		acquire_marker,
		setting_font,
		acquire_linestyle,
	)

	setting_font(fontsize=14, titlesize=14, legend_fontsize=14, font_family="Palatino")
	plt.rcParams["text.usetex"] = False
	plt.rcParams["font.serif"] = ["DejaVu Serif"]
	palette = acquire_color_palette()
	markers = acquire_marker()
	linestyles = acquire_linestyle()
	label_fontsize = 14
	title_fontsize = 14
	colorbar_ticksize = 12

	fig, axes = plt.subplots(1, len(panels), figsize=figsize)
	axes = np.atleast_1d(axes)

	for panel_idx, (ax, (title, pairs, color)) in enumerate(zip(axes, panels)):
		panel_color = color if color is not None else palette[panel_idx]
		im = ax.imshow(D_all, cmap="Greys", aspect="auto")
		if pairs:
			query_indices, db_indices = zip(*pairs)
			ax.plot(
				query_indices,
				db_indices,
				color=panel_color,
				linestyle=linestyles[0],
				linewidth=1.2,
				alpha=0.75,
			)
			ax.scatter(
				query_indices,
				db_indices,
				c=[panel_color],
				s=16,
				alpha=1.0,
				marker=markers[0],
			)
		colorbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		colorbar.ax.tick_params(labelsize=colorbar_ticksize)
		im.set_clim(0.0, 1.0)
		ax.set_xlabel("Query Index", fontsize=label_fontsize)
		ax.set_ylabel("Reference Index", fontsize=label_fontsize)
		ax.set_title(title, fontsize=title_fontsize)
		ax.tick_params(axis="both", labelsize=colorbar_ticksize)

	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def _save_dmatrix_artifact(recorder, merge_step: int, D_matrix: np.ndarray) -> pathlib.Path:
	from utils.utils_setting_color_font import acquire_color_palette

	artifact_path = recorder.artifact_path(merge_step, "dmatrix.png")
	palette = acquire_color_palette()
	_plot_runtime_dmatrix_panels(
		D_all=D_matrix,
		panels=[("Difference Matrix", [], palette[0])],
		output_path=artifact_path,
		figsize=(6, 5),
	)
	return artifact_path


def _record_pgo_event(merger: MergePipeline, stage: str, event_type: str, g2o_path: str, error: float = None):
	recorder = merger.runtime_viz_recorder
	if recorder is None:
		return
	payload = {"g2o_path": g2o_path}
	if error is not None:
		payload["error"] = error
	recorder.record_event(
		merge_step=merger.runtime_merge_step,
		stage=stage,
		event_type=event_type,
		submap_id=None,
		keyframe_id=None,
		payload=payload,
		artifacts={"g2o": pathlib.Path(g2o_path)},
	)


def _record_stage_annotation(
	merger: MergePipeline,
	merge_step: int,
	submap_id: int,
	stage_index: int,
	title: str,
	subtitle: str,
):
	recorder = merger.runtime_viz_recorder
	if recorder is None:
		return
	recorder.record_stage_annotation(
		merge_step=merge_step,
		submap_id=submap_id,
		stage_index=stage_index,
		stage_total=8,
		title=title,
		subtitle=subtitle,
	)
		
def compute_lm_pairwise(
	db_nodes, 
	query_node, 
	estimator,
	device
) -> Dict[Tuple[ImageNode, ImageNode], float]:
	K = estimator.scene.get_intrinsics()
	cams = torch.linalg.inv(estimator.scene.get_im_poses())
	depthmaps = estimator.scene.get_depthmaps()
	all_pts3d = estimator.scene.get_pts3d() # all pts3d in the world frame
	H, W = depthmaps[0].shape
	all_nodes = db_nodes + [query_node]
	msk_conf = estimator.scene.get_masks()
	assert len(all_pts3d) == len(all_nodes)

	# each element ('db'/'query', node_i, node_j, gain) meaning that
	# information gain by the node_i w.r.t. node_j, and node_i is the db/query node
	lm_gain_pw = list()
	for i in range(len(all_pts3d)):
		for j in range(len(all_pts3d)):
			# Only consider the overlapping between db_nodes and the query_node
			if i == j or (i != len(all_pts3d) - 1 and j != len(all_pts3d) - 1):
				continue

			# Project depth of camera i into camera j
			pts3d_flat = all_pts3d[i].reshape(-1, 3)
			proj = pts3d_flat @ cams[j][:3, :3].T + cams[j][:3, 3].reshape(1, 3)
			proj_depth = proj[:, 2]
			uv_hom = (proj / proj_depth[:, None])  # Add dimension for broadcasting
			u, v = (uv_hom[:, :2] @ K[j][:2, :2].T + K[j][:2, 2]).round().long().unbind(-1)			

			# Generate the projected depth map
			valid_mask = (proj_depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
			u, v, proj_depth = u[valid_mask], v[valid_mask], proj_depth[valid_mask]
			proj_depth_map = torch.zeros(H, W, device=device)
			proj_depth_map[v, u] = torch.maximum(proj_depth_map[v, u], proj_depth)

			# Consider as seen if the projected depth is close to the original depth
			u, v = torch.stack([u, v]).unique(dim=1)
			depth_diff_msk = torch.abs(proj_depth_map[v, u] - depthmaps[j][v, u]) < 0.5 * depthmaps[j][v, u]
			# Consider as seen if the region is with high confidence of the camera j
			conf_msk = msk_conf[j][v, u]
			msk = depth_diff_msk & conf_msk

			num_valid_reg_j = torch.sum(msk_conf[j]).float()
			redu = torch.sum(msk).float() / num_valid_reg_j if num_valid_reg_j > 0 else 0.0
			if j == len(all_pts3d) - 1:
				lm_gain_pw.append(('db', all_nodes[i], all_nodes[j], 1.0 - redu))
			else:
				lm_gain_pw.append(('query', all_nodes[i], all_nodes[j], 1.0 - redu))

			# DEBUG(gogojjh): Visualize the projected depth map
			# import matplotlib.pyplot as plt
			# fig, axs = plt.subplots(1, 2, figsize=(16, 12))
			# im0 = axs[0].imshow(depthmaps[j].detach().cpu().numpy(), cmap='turbo')
			# axs[0].set_title(f'Original Depth Camera {j} onto Camera {j}')
			# plt.colorbar(im0, ax=axs[0], label='Depth')			
			# im1 = axs[1].imshow(proj_depth_map.detach().cpu().numpy(), cmap='turbo')
			# axs[1].set_title(f'Projected Depth of Camera {i} onto Camera {j})')
			# plt.colorbar(im1, ax=axs[1], label='Depth')
			# plt.tight_layout()
			# plt.savefig(os.path.join('/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_aria/s00001/out_map_test/preds', f'depth_maps_{i}_to_{j}.jpg'))
			# plt.close()

	return lm_gain_pw

def perform_global_loc(
	merger: MergePipeline,
	final_graph: ImageGraph,
	cur_graph: ImageGraph,
	cur_graph_id: int,
	edge_history: dict = None
) -> List[Tuple[ImageNode, ImageNode, np.ndarray, float]]:
	"""Performs coarse localization between a reference map and a query submap.
	
	This function uses a VPR model to find coarse correspondences between nodes
	in the reference map and the query submap. It optionally applies RANSAC-based
	outlier rejection and saves visualization results.
	
	Args:
		merger: The merger object containing the VPR model and configuration.
		final_graph: The reference map containing database nodes.
		cur_graph: The query submap to localize within the reference map.
		cur_graph_id: Identifier for the current submap for logging purposes.
		edge_change_log: List of edges that recorded history of change.
		
	Returns:
		A list of edges representing potential matches between database and query
		nodes. Each edge is a tuple (db_node, query_node, T_A2B, score).
	"""
	# Load descriptors from database and query nodes
	db_descriptors = np.array(
		[node.get_descriptor() for node in final_graph.nodes.values()], dtype=np.float32
	)
	db_node_ids = [node.id for node in final_graph.nodes.values()]

	query_descriptors = np.array(
		[node.get_descriptor() for node in cur_graph.nodes.values()], dtype=np.float32
	)
	query_node_ids = [node.id for node in cur_graph.nodes.values()]
	recorder = merger.runtime_viz_recorder
	if recorder is not None:
		recorder.record_event(
			merge_step=merger.runtime_merge_step,
			stage="descriptor_computed",
			event_type="descriptor_computed",
			submap_id=cur_graph_id,
			keyframe_id=None,
			payload={
				"reference_descriptor_shape": db_descriptors.shape,
				"query_descriptor_shape": query_descriptors.shape,
				"reference_node_ids": db_node_ids,
				"query_node_ids": query_node_ids,
			},
		)

	with Timer(name="Global Localization", text=Fore.GREEN + "{name} costs: {milliseconds:.3f} ms"):
		merger.vpr_match_model.initialize_model(db_descriptors)
		D_all = merger.vpr_match_model.compute_diff_matrix(query_descriptors)
		logging.info(f"D_all shape: {D_all.shape}")
		if recorder is not None:
			dmatrix_path = _save_dmatrix_artifact(recorder, merger.runtime_merge_step, D_all)
			recorder.record_event(
				merge_step=merger.runtime_merge_step,
				stage="dmatrix_computed",
				event_type="dmatrix_computed",
				submap_id=cur_graph_id,
				keyframe_id=None,
				payload={"shape": D_all.shape},
				artifacts={"dmatrix_png": dmatrix_path},
			)
			_record_stage_annotation(
				merger,
				merge_step=merger.runtime_merge_step,
				submap_id=cur_graph_id,
				stage_index=4,
				title=f"VPR Sequence Matching - Reference Map-Submap {cur_graph_id}",
				subtitle="Search the difference matrix for topological matches between reference and query frames.",
			)
		
		# VPR sequence matching for all query nodes
		connected_db_query_indices = []
		if 'PlaceRecognitionGraphSearch' in type(merger.vpr_match_model).__name__:
			pred_db_query_rows, score = merger.vpr_match_model.match(query_descriptors)
			for db_row, query_row in pred_db_query_rows:
				db_idx, query_idx = db_node_ids[db_row], query_node_ids[query_row]
				connected_db_query_indices.append((db_idx, query_idx, score))
				if recorder is not None:
					recorder.record_event(
						merge_step=merger.runtime_merge_step,
						stage="vpr_candidate",
						event_type="vpr_candidate",
						submap_id=cur_graph_id,
						keyframe_id=query_idx,
						payload={"db_node_id": db_idx, "query_node_id": query_idx, "db_row": db_row, "query_row": query_row, "score": score},
					)
				update_edge_history(
					edge_history, 
					(db_idx, query_idx), 
					action='added_by_vpr', db_row=db_row, query_row=query_row
				)
		elif 'PlaceRecognitionSeqMatching' in type(merger.vpr_match_model).__name__:
			for query_row in range(len(query_descriptors)):
				query_descs = query_descriptors[max(0, query_row-merger.vpr_match_model.seqLen+1) : query_row + 1]
				_, db_row, score = merger.vpr_match_model.match(query_descs)
				db_idx, query_idx = db_node_ids[db_row], query_node_ids[query_row]
				connected_db_query_indices.append((db_idx, query_idx, score))
				if recorder is not None:
					recorder.record_event(
						merge_step=merger.runtime_merge_step,
						stage="vpr_candidate",
						event_type="vpr_candidate",
						submap_id=cur_graph_id,
						keyframe_id=query_idx,
						payload={"db_node_id": db_idx, "query_node_id": query_idx, "db_row": db_row, "query_row": query_row, "score": score},
					)
				update_edge_history(
					edge_history, 
					(db_idx, query_idx), 
					action='added_by_vpr', db_row=db_row, query_row=query_row
				)
		else:
			raise ValueError(f"Unsupported VPR match model: {type(merger.vpr_match_model).__name__}")

		# Geometric Verification
		_record_stage_annotation(
			merger,
			merge_step=merger.runtime_merge_step,
			submap_id=cur_graph_id,
			stage_index=5,
			title=f"Geometric Verification - Reference Map-Submap {cur_graph_id}",
			subtitle="Reject false positive topological matches using feature inlier checks.",
		)
		coarse_edges = []
		for db_idx, query_idx, _ in connected_db_query_indices:
			db_node = final_graph.get_node(db_idx)	
			query_node = cur_graph.get_node(query_idx)
			result = merger.pose_estimator.get_matched_kpts(
				final_graph.map_root, 
				db_node.rgb_image, 
				query_node.rgb_image
			)
			num_inlier = result['num_inliers']
			warning_str = Fore.GREEN + f"Query {query_node.rgb_img_name}-DB {db_node.rgb_img_name}-Matched Kpts: {num_inlier}"
			logging.warning(warning_str)

			################# DEBUG(gogojjh): Visualize the matched keypoints
			# save_visualization(
			# 	to_numpy(db_node.rgb_image.permute(1, 2, 0)), to_numpy(query_node.rgb_image.permute(1, 2, 0)), 
			# 	result['inlier_kpts0'], result['inlier_kpts1'], merger.log_dir / 'preds/match_vis', 
			# 	query_idx
			# )
			#################
			if num_inlier >= REFINE_GV_SCORE_THRESHOLD: 
				coarse_edges.append((db_node, query_node, np.eye(4), num_inlier))
				accepted_by_gv = True
			else:
				update_edge_history(edge_history, (db_idx, query_idx), action='removed_by_gv')
				accepted_by_gv = False
			if recorder is not None:
				recorder.record_event(
					merge_step=merger.runtime_merge_step,
					stage="gv_candidate",
					event_type="gv_candidate",
					submap_id=cur_graph_id,
					keyframe_id=query_idx,
					payload={
						"db_node_id": db_idx,
						"query_node_id": query_idx,
						"num_inliers": num_inlier,
						"accepted": accepted_by_gv,
						"threshold": REFINE_GV_SCORE_THRESHOLD,
					},
				)
	
	return coarse_edges, D_all

def perform_local_loc(
	edges_nodeA_to_nodeB_coarse: List[Tuple[ImageNode, ImageNode, np.ndarray, float]],
	merger: MergePipeline,
	final_graph: ImageGraph,
	cur_graph: ImageGraph,
	cur_graph_id: int,
	edge_history: dict = None
) -> Tuple[List[Tuple[ImageNode, ImageNode, np.ndarray, float]],
		   Dict[ImageNode, Dict[ImageNode, float]],
		   Dict[ImageNode, Dict[ImageNode, float]]]:
	"""Performs fine-grained localization using pose estimation on coarse matches.
	
	Args:
		edges_nodeA_to_nodeB_coarse: List of coarse matches (db_node, query_node, T_A2B, score)
		final_graph: Reference map containing database nodes
		cur_graph: Query submap to localize
		merger: Merger object with pose estimator and configuration
		
	Returns:
		refined_edges: List of refined matches (represented as image node) with relative pose estimates
	"""
	# lm_gain_db[nodeA][nodeB] meaning how much information is gained of nodeA w.r.t. nodeB, and nodeA is a db_node
	# lm_gain_query[nodeA][nodeB] meaning how much information is gained of nodeA w.r.t. nodeB, and nodeA is a query_node
	lm_gain_db, lm_gain_query = dict(), dict()
	# Each element of edges_nodeAB_refine_covis: [nodeA, nodeB, T_rel, conf, overlapping_score]
	refined_edges = []
	lloc_history = dict()
	for edge in tqdm(edges_nodeA_to_nodeB_coarse):
		db_node, query_node = edge[:2]
		# Check whether the node has more than one edge
		if len(db_node.edges) == 0: 
			continue	
		
		try:
			# Prepare database references
			other_db_node, min_dis = None, float('inf')
			for node_weight in db_node.edges.values():
				desc_dis = np.linalg.norm(
					node_weight[0].global_descriptor - query_node.global_descriptor
				)
				if desc_dis < min_dis:
					min_dis = desc_dis
					other_db_node = node_weight[0]

			db_node_pair = [db_node, other_db_node]
			db_names = [n.rgb_img_name for n in db_node_pair]
			db_poses = [torch.from_numpy(convert_vec_to_matrix(n.trans, n.quat, 'xyzw')) for n in db_node_pair]
			db_intrs = [{
				'K': torch.from_numpy(n.raw_K),
				'im_size': torch.from_numpy(n.raw_img_size)
			} for n in db_node_pair]

			# Prepare query data
			query_name = query_node.rgb_img_name
			query_intr = {
				'K': torch.from_numpy(query_node.raw_K),
				'im_size': torch.from_numpy(query_node.raw_img_size)
			}

			# Perform pose estimation with timing
			with Timer(name="Pose Estimation", text=Fore.GREEN + "{name} costs: {milliseconds:.3f} ms"):
				result = merger.pose_estimator(
					final_graph.map_root,
					[node.rgb_image for node in db_node_pair],
					query_node.rgb_image,
					db_poses, db_intrs, 
					query_intr, 
					merger.est_opts
				)				
				im_pose = result["im_pose"] # camera pose in the world frame
				if im_pose is None: 
					raise ValueError(f"{merger.pose_estimator} - Estimated pose is None.")
				elif np.isnan(im_pose).any():
					raise ValueError("Estimated pose is NaN or infinite.")
				
				T_db_est = convert_vec_to_matrix(db_node.trans, db_node.quat, 'xyzw')
				T_rel_est = np.linalg.inv(T_db_est) @ im_pose

				##############################
				T_nodeA_gt = convert_vec_to_matrix(db_node.trans_gt, db_node.quat_gt, 'xyzw')
				T_query_gt = convert_vec_to_matrix(query_node.trans_gt, query_node.quat_gt, 'xyzw')
				T_rel_gt = np.linalg.inv(T_nodeA_gt) @ T_query_gt
				logging.warning(f"EST Rel Pose: {T_rel_est[:3, 3:4].T}")
				logging.warning(f"GT Rel Pose: {T_rel_gt[:3, 3:4].T}")
				dis_tsl, dis_angle = compute_pose_error(T_rel_est, T_rel_gt, 'matrix')
				logging.warning(f"Error in translation: {dis_tsl:.3f} [m] and rotation {dis_angle:.3f} [deg]")
				##############################

				# Applicable to master and duster
				if hasattr(merger.pose_estimator, 'get_minimum_spanning_tree'):
					top_k_matches = len(db_names) # default: 2
					msp_edges = merger.pose_estimator.get_minimum_spanning_tree()
					conf_i, conf_j = _scene_confidence_maps(merger.pose_estimator.scene)
					for edge in msp_edges:
						if edge[0] == top_k_matches or edge[1] == top_k_matches: # confidence of the query image
							edge_str = f"{edge[0]}_{edge[1]}"
							conf = (conf_i[edge_str].mean() * conf_j[edge_str].mean()).detach().cpu().item()

					logging.warning(Fore.GREEN + f"{db_names[0]} {db_names[1]} - {query_name} with conf: {conf:.3f}")
					##### Only reliable db-query pairs are considered for keyframe selection
					lloc_history[(db_node.id, query_node.id)] = {'conf': conf, 'trans_err': dis_tsl, 'rot_err': dis_angle}
					if merger.runtime_viz_recorder is not None:
						merger.runtime_viz_recorder.record_event(
							merge_step=merger.runtime_merge_step,
							stage="metric_localization_result",
							event_type="metric_localization_result",
							submap_id=cur_graph_id,
							keyframe_id=query_node.id,
							payload={
								"db_node_id": db_node.id,
								"query_node_id": query_node.id,
								"conf": conf,
								"accepted": conf > REFINE_CONF_THRESHOLD,
								"reliable": conf > RELIABLE_CONF_THRESHOLD,
								"trans_err": dis_tsl,
								"rot_err": dis_angle,
								"relative_pose": T_rel_est,
							},
						)
					if conf < REFINE_CONF_THRESHOLD:
						update_edge_history(edge_history, (db_node.id, query_node.id), action='removed_by_ccm')
					
					if conf > RELIABLE_CONF_THRESHOLD:
						lm_gain_pw = compute_lm_pairwise(
							db_node_pair,
							query_node, 
							merger.pose_estimator, 
							merger.args.device
						)
						for idr, node_i, node_j, gain in lm_gain_pw:
							if idr == 'db':
								if node_i not in lm_gain_db:
									lm_gain_db[node_i] = dict() 
								lm_gain_db[node_i][node_j] = gain
							elif idr == 'query':
								if node_i not in lm_gain_query:
									lm_gain_query[node_i] = dict()
								lm_gain_query[node_i][node_j] = gain

					if conf > REFINE_CONF_THRESHOLD:
						refined_edges.append(
							(db_node, query_node, T_rel_est, conf, 1.0-lm_gain_db[db_node][query_node])
						)
						if merger.runtime_viz_recorder is not None:
							merger.runtime_viz_recorder.record_event(
								merge_step=merger.runtime_merge_step,
								stage="metric_edge_added",
								event_type="metric_edge_added",
								submap_id=cur_graph_id,
								keyframe_id=query_node.id,
								payload={
									"db_node_id": db_node.id,
									"query_node_id": query_node.id,
									"conf": conf,
									"trans_err": dis_tsl,
									"rot_err": dis_angle,
									"relative_pose": T_rel_est,
								},
							)
				# Applicable to other estimators
				else:
					conf = MAX_LOSS - result["loss"]
					if merger.runtime_viz_recorder is not None:
						merger.runtime_viz_recorder.record_event(
							merge_step=merger.runtime_merge_step,
							stage="metric_localization_result",
							event_type="metric_localization_result",
							submap_id=cur_graph_id,
							keyframe_id=query_node.id,
							payload={
								"db_node_id": db_node.id,
								"query_node_id": query_node.id,
								"conf": conf,
								"accepted": True,
								"relative_pose": T_rel_est,
							},
						)
					refined_edges.append((db_node, query_node, T_rel_est, conf, conf / MAX_LOSS))
					if merger.runtime_viz_recorder is not None:
						merger.runtime_viz_recorder.record_event(
							merge_step=merger.runtime_merge_step,
							stage="metric_edge_added",
							event_type="metric_edge_added",
							submap_id=cur_graph_id,
							keyframe_id=query_node.id,
							payload={
								"db_node_id": db_node.id,
								"query_node_id": query_node.id,
								"conf": conf,
								"relative_pose": T_rel_est,
							},
						)
			
		except Exception as e:
			update_edge_history(edge_history, (db_node.id, query_node.id), action='removed_by_ccm')
			if merger.runtime_viz_recorder is not None:
				merger.runtime_viz_recorder.record_event(
					merge_step=merger.runtime_merge_step,
					stage="metric_localization_result",
					event_type="metric_localization_result",
					submap_id=cur_graph_id,
					keyframe_id=query_node.id,
					payload={
						"db_node_id": db_node.id,
						"query_node_id": query_node.id,
						"accepted": False,
						"error": str(e),
					},
				)
			logging.warning(f"{Fore.RED} Pose estimation failed: {str(e)}")
			continue

	return refined_edges, lm_gain_db, lm_gain_query, lloc_history

def perform_keyframe_culling(
	merger: MergePipeline,
	args,
	cur_submap: MapManager,
	lm_gain_query: Dict[ImageNode, Dict[ImageNode, float]],
	lm_gain_db: Dict[ImageNode, Dict[ImageNode, float]]
) -> Tuple[List, List, List]:
	"""Perform keyframe culling based on quality and information gain factors.
	
	This function implements forward and backward culling strategies:
	- Forward culling: Reject newly inserted keyframes with low quality or information gain
	- Backward culling: Replace old keyframes with newer ones that have better quality/gain
	
	Args:
		merger: The merger object containing selector and configuration
		args: Command line arguments with culling flags
		cur_submap: Current submap being merged
		lm_gain_query: Information gain for query nodes (new keyframes)
		lm_gain_db: Information gain for database nodes (old keyframes)
		
	Returns:
		Tuple of (nodes_query_to_cull, nodes_db_to_cull, nodes_to_cull_info)
	"""
	nodes_query_to_cull, nodes_db_to_cull = [], []
	nodes_to_cull_info, nodes_to_not_cull_info = [], []
	
	##### Forward Pass Culling #####
	# Factor: IQA to each node in the current map (cur_graph)
	if args.use_iqa:
		# Go through all nodes in the current map, check IQA probability
		for node_query in cur_submap.covis.nodes.values():
			acc_prob = merger.lm_selector.quality_probability(node_query.iqa_data)
			prob_str = merger.lm_selector.print_prefilter_prob(node_query.iqa_data, use_iqa=args.use_iqa)
			if acc_prob < merger.lm_selector.P_iqa_th:
				nodes_query_to_cull.append(node_query)
				nodes_to_cull_info.append({
					'node_id': node_query.id,
					'type': 'query',
					'prob': acc_prob,
					'method': 'culled_by_iqa',
					'prob_str': prob_str
				})
				if args.viz:
					save_vis_kf_removal(
						merger.log_dir, node_query.id,
						to_numpy(node_query.rgb_image.detach().squeeze(0).permute(1, 2, 0)),
						acc_prob
					)
			else:
				nodes_to_not_cull_info.append({
					'node_id': node_query.id,
					'type': 'query',
					'prob': acc_prob,
					'method': 'not_culled_by_iqa',
					'prob_str': prob_str
				})

	# Factor: IQA + IG + TD to each node in the current map
	# Accept the new keyframe with high information gain, even if it has low image quality
	for node_query, data in lm_gain_query.items():
		acc_prob, node_rep = 1.0, None
		prob_str = ''
		for node_db, gain in data.items():
			prob = merger.lm_selector.compute_forward_prob(
				node_query.iqa_data, gain, node_query.time - node_db.time,
				use_iqa=args.use_iqa, use_ig=args.use_ig, use_td=args.use_td
			)
			if prob < acc_prob:
				acc_prob, node_rep = prob, node_db
				prob_str = merger.lm_selector.print_each_forward_prob(
					node_query.iqa_data, gain, node_query.time - node_db.time,
					use_iqa=args.use_iqa, use_ig=args.use_ig, use_td=args.use_td
				)
		
		if acc_prob < merger.lm_selector.P_acc_th:
			if node_query not in nodes_query_to_cull:
				nodes_query_to_cull.append(node_query)
				nodes_to_cull_info.append({
					'node_id': node_query.id,
					'type': 'query',
					'replaced_by': node_rep.id,
					'prob': acc_prob,
					'method': 'culled_by_forward',
					'prob_str': prob_str
				})
				if args.viz:
					save_vis_kf_removal(
						merger.log_dir, 
						node_query.id, to_numpy(node_query.rgb_image.detach().squeeze(0).permute(1, 2, 0)), 
						acc_prob,
						node_rep.id, to_numpy(node_rep.rgb_image.detach().squeeze(0).permute(1, 2, 0))
					)
		elif node_rep is not None:
			nodes_to_not_cull_info.append({
				'node_id': node_query.id,
				'type': 'query',
				'compared_to': node_rep.id,
				'prob': acc_prob,
				'method': 'not_culled_by_forward',
				'prob_str': prob_str
			})

	##### Backward Pass Culling #####
	# Factor: IG + TD to each node in the final map
	# Cull the old keyframe with low information gain and low image quality
	for node_db, data in lm_gain_db.items():
		acc_prob, node_rep = 1.0, None
		prob_str = ''
		for node_query, gain in data.items():
			if node_query in nodes_query_to_cull: 
				continue
			prob = merger.lm_selector.compute_backward_prob(
				gain, node_query.time - node_db.time,
				use_ig=args.use_ig, use_td=args.use_td
			)
			if prob < acc_prob:
				acc_prob, node_rep = prob, node_query
				prob_str = merger.lm_selector.print_each_backward_prob(
					lm_gain_db[node_db][node_rep], 
					node_rep.time - node_db.time,
					use_ig=args.use_ig, use_td=args.use_td
				)

		if acc_prob < merger.lm_selector.P_keep_th:
			nodes_db_to_cull.append(node_db)
			nodes_to_cull_info.append({
				'node_id': node_db.id,
				'type': 'db',
				'prob': acc_prob,
				'method': 'culled_by_backward',
				'replaced_by': node_rep.id,
				'prob_str': prob_str
			})
			if args.viz:
				save_vis_kf_replacement(
					merger.log_dir, 
					node_db.id, node_rep.id,
					to_numpy(node_db.rgb_image.detach().squeeze(0).permute(1, 2, 0)), 
					to_numpy(node_rep.rgb_image.detach().squeeze(0).permute(1, 2, 0)),
					acc_prob
				)
		elif node_rep is not None:
			nodes_to_not_cull_info.append({
				'node_id': node_db.id,
				'type': 'db',
				'prob': acc_prob,
				'method': 'not_culled_by_backward',
				'prob_str': prob_str
			})

	return nodes_query_to_cull + nodes_db_to_cull, nodes_to_cull_info, nodes_to_not_cull_info

def perform_submap_merging(merger: MergePipeline, args):
	"""Main loop for processing submap merging"""
	assert len(merger.submaps) > 0, "No submaps loaded."
	logging.info(f"Processing {len(merger.submaps)} submaps.")
	if args.rerun_viz:
		rerun_viz_dir = pathlib.Path(args.rerun_viz_dir) if args.rerun_viz_dir else pathlib.Path(args.output_map_path) / "rerun_viz"
		merger.runtime_viz_recorder = MapMergeRuntimeEventRecorder(rerun_viz_dir)
		merger.runtime_viz_recorder.write_metadata(
			{
				"input_submap_path": args.input_submap_path,
				"output_map_path": args.output_map_path,
				"vpr_match_model": args.vpr_match_model,
				"vpr_match_seq_len": args.vpr_match_seq_len,
				"pose_estimation_method": args.pose_estimation_method,
			}
		)
		merger.runtime_viz_recorder.record_event(
			merge_step=-1,
			stage="recording_started",
			event_type="recording_started",
			submap_id=None,
			keyframe_id=None,
			payload={"output_dir": str(rerun_viz_dir)},
		)
	
	# Initialize the final submap
	final_map = MapManager(merger.log_dir)
	final_map.init_graphs(merger.graph_configs)

	# Incrementally merge each submap to the final map, only care about the first two submaps
	for merge_step, cur_submap in enumerate(merger.submaps):
		merger.runtime_merge_step = merge_step
		if merge_step == 0:
			_record_stage_annotation(
				merger,
				merge_step=merge_step,
				submap_id=cur_submap.map_id,
				stage_index=1,
				title="Load Reference Map",
				subtitle="Replay keyframes and odom/covis/trav graph edges for the reference submap.",
			)
		elif merge_step == 1:
			_record_stage_annotation(
				merger,
				merge_step=merge_step,
				submap_id=cur_submap.map_id,
				stage_index=2,
				title=f"Load Submap {cur_submap.map_id}",
				subtitle="Replay keyframes and odom/covis/trav graph edges for the query submap.",
			)
		_record_submap_loaded(merger, merge_step, cur_submap)
		# The offset of node.id in cur_submap 
		merger.id_offset = final_map.get_max_node_id() + 1

		if not final_map.is_empty:
			edge_history = dict()
			# Identify coarse covisibility relationship 
			_record_stage_annotation(
				merger,
				merge_step=merge_step,
				submap_id=cur_submap.map_id,
				stage_index=3,
				title=f"Compute Difference Matrix - Reference Map-Submap {cur_submap.map_id}",
				subtitle="Compare query and reference descriptors to build the localization cost matrix.",
			)
			edges_nodeAB_coarse_covis, D_matrix = perform_global_loc(
				merger,
				final_map.covis, 
				cur_submap.covis, 
				cur_submap.map_id,
				edge_history
			)
			# Identify strong covisibility relationship 
			_record_stage_annotation(
				merger,
				merge_step=merge_step,
				submap_id=cur_submap.map_id,
				stage_index=6,
				title=f"Metric Localization - Reference Map-Submap {cur_submap.map_id}",
				subtitle="Estimate metric inter-submap constraints from geometrically verified candidates.",
			)
			edges_nodeAB_refine_covis, lm_gain_db, lm_gain_query, lloc_history = perform_local_loc(
				edges_nodeAB_coarse_covis,
				merger, 
				final_map.covis,
				cur_submap.covis, 
				cur_submap.map_id,
				edge_history
			)

			##### Perform Pose Graph Optimization #####
			_record_stage_annotation(
				merger,
				merge_step=merge_step,
				submap_id=cur_submap.map_id,
				stage_index=7,
				title=f"Pose Graph Optimization - Reference Map-Submap {cur_submap.map_id}",
				subtitle="Optimize the combined pose graph before committing the merged map.",
			)
			# Initialize the pose graph
			logging.info(Fore.GREEN + f'Performing PGO for Submap {cur_submap.map_id}' + Fore.RESET)
			pose_graph, subgraph_keys = merger.create_pose_graph_from_map(
				final_map.odom,
				cur_submap.odom, 
				edges_nodeAB_refine_covis
			)	
			g2o_path = str(merger.log_dir/"preds/initial_pose_graph.g2o")
			gtsam.writeG2o(pose_graph.get_factor_graph(), pose_graph.get_initial_estimate(), g2o_path)
			_record_pgo_event(
				merger,
				stage="pgo_before",
				event_type="pgo_before",
				g2o_path=g2o_path,
				error=pose_graph.get_factor_graph().error(pose_graph.get_initial_estimate()),
			)
			
			# Optimize the pose graph
			logging.info(f"PGO: initial error: {pose_graph.get_factor_graph().error(pose_graph.get_initial_estimate()):.3f}")
			result_pgo = PoseGraph.optimize_pose_graph_with_LM(
				pose_graph.get_factor_graph(), 
				pose_graph.get_initial_estimate(), 
				verbose=False,
				robust_kernel=True
			)
			logging.info(f"PGO: final error: {pose_graph.get_factor_graph().error(result_pgo):.3f}")

			##### Visualization #####
			if args.viz:
				save_dir = str(merger.log_dir / "preds")
                ### Visualize the difference matrix (with/without GV)
				db_query_rows = [
					(value['db_row'], value['query_row']) for value in edge_history.values()
				]
				merger.vpr_match_model.viz_diff_matrix(
					os.path.join(save_dir, 'D_matrix_vpr.jpg'), D_matrix, db_query_rows
				)
				db_query_rows = [
					(value['db_row'], value['query_row']) for value in edge_history.values()
					if 'removed_by_gv' not in value['action']
				]
				merger.vpr_match_model.viz_diff_matrix(
					os.path.join(save_dir, 'D_matrix_gv.jpg'), D_matrix, db_query_rows
				)
				### Visualize the edge connections (with/without GV/CCM)
				precision_list, recall_list = save_vis_edge_history(
					save_dir, final_map.covis, cur_submap.covis, edge_history
				)
				### Visualize the optimized pose graph
				pose_graph.plot_pose_graph(
					save_dir, pose_graph.get_factor_graph(), 
					[pose_graph.get_initial_estimate(), result_pgo],
					['Before PGO', 'After PGO'], mode='2d', 
					subgraph_keys=subgraph_keys
				)

				total_num_edges = len(edge_history)
				num_edge_added_by_vpr, num_edge_removed_by_gv, num_edge_removed_by_ccm = 0, 0, 0
				for key, value in edge_history.items():
					db_idx, query_idx = int(key[0]), int(key[1])
					action = value['action'] if isinstance(value, dict) else value
					if 'added_by_vpr' in action:
						num_edge_added_by_vpr += 1
					elif 'removed_by_gv' in action:
						num_edge_removed_by_gv += 1
					elif 'removed_by_ccm' in action:
						num_edge_removed_by_ccm += 1
				
				edge_history_path = str(merger.log_dir / "preds" / "edge_history.txt")
				with open(edge_history_path, 'w') as f:
					f.write(f"Number of edges added by VPR: {total_num_edges}\n")
					f.write(f"Number of edges removed by GV: {num_edge_removed_by_gv} ({num_edge_removed_by_gv/total_num_edges*100:.2f}%)\n")
					f.write(f"Number of edges removed by CCM: {num_edge_removed_by_ccm} ({num_edge_removed_by_ccm/total_num_edges*100:.2f}%)\n")
					f.write(f"Number of edges retained: {num_edge_added_by_vpr} ({num_edge_added_by_vpr/total_num_edges*100:.2f}%)\n")
					f.write(f"Precision: " + ",".join([f"{precision:.2f}" for precision in precision_list]) + "\n")
					f.write(f"Recall: " + ",".join([f"{recall:.2f}" for recall in recall_list]) + "\n")
					for key, value in edge_history.items():
						db_idx, query_idx = key[0], key[1]
						action = value['action']
						f.write(f"{db_idx},{query_idx},{action}\n")

				lloc_history_path = str(merger.log_dir / "preds" / "lloc_history.txt")
				with open(lloc_history_path, 'w') as f:
					sorted_lloc_history = sorted(
						lloc_history.items(), 
						key=lambda item: item[1]['conf'],
						reverse=True
					)
					for key, value in sorted_lloc_history:
						db_idx, query_idx = key[0], key[1]
						f.write(f"{db_idx},{query_idx},Conf: {value['conf']:.3f} - Error: {value['trans_err']:.3f} [m] and {value['rot_err']:.3f} [deg]\n")
			########################

			for key in result_pgo.keys():
				update_estimate = result_pgo.atPose3(key)
				pose_graph.add_init_estimate(key, update_estimate)
			g2o_path = str(merger.log_dir / "preds" / "refine_pose_graph.g2o")
			gtsam.writeG2o(pose_graph.get_factor_graph(), pose_graph.get_initial_estimate(), g2o_path)
			_record_pgo_event(
				merger,
				stage="pgo_after",
				event_type="pgo_after",
				g2o_path=g2o_path,
				error=pose_graph.get_factor_graph().error(pose_graph.get_initial_estimate()),
			)
			
			##### Perform keyframe culling #####
			# Culling nodes in covis graph only, nodes in odom and trav graph are kept
			# The id of culled nodes will not be replaced by new nodes to avoid conflict with odom and trav graph
			cull_keyframe = args.use_iqa or args.use_ig or args.use_td
			if cull_keyframe:
				print(Fore.GREEN + "Performing keyframe culling..." + Fore.RESET)
				nodes_to_cull, nodes_to_cull_info, nodes_to_not_cull_info = perform_keyframe_culling(
					merger, args, cur_submap, lm_gain_query, lm_gain_db
				)

				# Write cull node information to log file
				info_path = merger.log_dir / "preds" / "cull_node_info.txt"
				info_path.parent.mkdir(parents=True, exist_ok=True)
				with open(info_path, 'w') as f:
					f.write(f"# Ablation Study Configuration:\n")
					f.write(f"# IQA: {args.use_iqa}\n")
					f.write(f"# IG: {args.use_ig}\n")
					f.write(f"# TD: {args.use_td}\n")
					f.write("node_id,type,replaced_by,prob,method,prob_str\n")
					for record in nodes_to_cull_info:
						node_id = record['node_id']
						type_ = record['type']
						replaced_by = record.get('replaced_by', "")
						prob = record['prob']
						method = record['method']
						prob_str = record.get('prob_str', "")
						f.write(f"{node_id},{type_},{replaced_by},{prob:.3f},{method},{prob_str}\n")

				info_path = merger.log_dir / "preds" / "not_cull_node_info.txt"
				info_path.parent.mkdir(parents=True, exist_ok=True)
				with open(info_path, 'w') as f:
					f.write(f"# Ablation Study Configuration:\n")
					f.write(f"# IQA: {args.use_iqa}\n")
					f.write(f"# IG: {args.use_ig}\n")
					f.write(f"# TD: {args.use_td}\n")
					f.write("node_id,type,compared_to,prob,method,prob_str\n")
					for record in nodes_to_not_cull_info:
						node_id = record['node_id']
						type_ = record['type']
						compared_to = record.get('compared_to', "")
						prob = record['prob']
						method = record['method']
						prob_str = record.get('prob_str', "")
						f.write(f"{node_id},{type_},{compared_to},{prob:.3f},{method},{prob_str}\n")						
			else:
				nodes_to_cull, nodes_to_cull_info, nodes_to_not_cull_info = [], [], []
			if merger.runtime_viz_recorder is not None:
				for record in nodes_to_cull_info:
					merger.runtime_viz_recorder.record_event(
						merge_step=merger.runtime_merge_step,
						stage="keyframe_culling_decision",
						event_type="keyframe_culling_decision",
						submap_id=cur_submap.map_id,
						keyframe_id=record.get("node_id"),
						payload=record,
					)

			##### Perform map update and merging
			_record_stage_annotation(
				merger,
				merge_step=merge_step,
				submap_id=cur_submap.map_id,
				stage_index=8,
				title=f"Submap Merging - Reference Map-Submap {cur_submap.map_id}",
				subtitle="Merge the optimized query submap into the reference map and update graph edges.",
			)
			# Merge two submap into one with optimized poses
			merger.merge_and_update_submaps(final_map, cur_submap, pose_graph.get_initial_estimate())
			# Enforce all node id in edges_nodeAB_refine_covis to be adjusted
			final_map.covis.remove_node_list(nodes_to_cull)
			final_map.covis.remove_invalid_edges(nodes_to_cull)
			final_map.covis.rm_sensor_data(nodes_to_cull)

			# Update edges from the src_edges for different types of graphs
			# Nodes are merged and reflected on the updated graph
			# node_a = final_map.graphs[graph_type].get_node(edges_nodeAB[0])
			# node_b = final_map.graphs[graph_type].get_node(edges_nodeAB[1])
			weight_func1 = (lambda edge: edge[4]) # overlapping score
			weight_func2 = (lambda edge: np.linalg.norm(edge[0].trans - edge[1].trans)) 
			for dst_graph_type, src_edges, weight_func in [
				("covis", edges_nodeAB_refine_covis, weight_func1),
				("odom", edges_nodeAB_refine_covis, weight_func2),
				("trav", edges_nodeAB_refine_covis, weight_func2)
			]:
				dst_edges = final_map.update_edges(src_edges, dst_graph_type)
				final_map.graphs[dst_graph_type].add_inter_edges(dst_edges, weight_func)
			
			logging.info(f"Final map info:\n{final_map}")
			if merger.runtime_viz_recorder is not None:
				merger.runtime_viz_recorder.record_event(
					merge_step=merger.runtime_merge_step,
					stage="map_committed",
					event_type="map_committed",
					submap_id=cur_submap.map_id,
					keyframe_id=None,
					payload={
						"num_final_covis_nodes": final_map.covis.get_num_node(),
						"num_final_odom_nodes": final_map.odom.get_num_node(),
						"num_culled_nodes": len(nodes_to_cull),
						"nodes": [
							{"node_id": node.id, "position": node.trans.tolist(), "quat_xyzw": node.quat.tolist()}
							for node in final_map.covis.nodes.values()
						],
						"edges": {
							etype: [
								[int(node_a.id), int(node_b.id)]
								for node_a, node_b, _ in _iter_unique_edges(getattr(final_map, etype))
							]
							for etype in ("odom", "covis", "trav")
							if getattr(final_map, etype, None) is not None
						},
					},
				)
		else:
			pose_graph, _ = merger.create_pose_graph_from_map(
				final_map.odom, 
				cur_submap.odom, 
				[]
			)
			g2o_path = str(merger.log_dir / "preds" / "initial_pose_graph.g2o")
			gtsam.writeG2o(pose_graph.get_factor_graph(), pose_graph.get_initial_estimate(), g2o_path)
			_record_pgo_event(
				merger,
				stage="pgo_before",
				event_type="pgo_before",
				g2o_path=g2o_path,
				error=pose_graph.get_factor_graph().error(pose_graph.get_initial_estimate()),
			)
			
			if args.viz:
				save_dir = str(merger.log_dir / "preds")
				# Visualize the edge connections (without GV/CCM)
				save_vis_edge_history(
					save_dir, final_map.covis, cur_submap.covis, dict()
				)
				# Visualize the pose graph
				pose_graph.plot_pose_graph(
					save_dir, pose_graph.get_factor_graph(), 
					[pose_graph.get_initial_estimate(), pose_graph.get_initial_estimate()],
					['Before PGO', 'Before PGO'], mode='2d'
				)

			merger.merge_and_update_submaps(final_map, cur_submap, pose_graph.get_initial_estimate())
			if merger.runtime_viz_recorder is not None:
				merger.runtime_viz_recorder.record_event(
					merge_step=merger.runtime_merge_step,
					stage="map_committed",
					event_type="map_committed",
					submap_id=cur_submap.map_id,
					keyframe_id=None,
					payload={
						"num_final_covis_nodes": final_map.covis.get_num_node(),
						"num_final_odom_nodes": final_map.odom.get_num_node(),
						"num_culled_nodes": 0,
						"nodes": [
							{"node_id": node.id, "position": node.trans.tolist(), "quat_xyzw": node.quat.tolist()}
							for node in final_map.covis.nodes.values()
						],
						"edges": {
							etype: [
								[int(node_a.id), int(node_b.id)]
								for node_a, node_b, _ in _iter_unique_edges(getattr(final_map, etype))
							]
							for etype in ("odom", "covis", "trav")
							if getattr(final_map, etype, None) is not None
						},
					},
				)

	if not final_map.is_empty:
		final_map.save_to_file()
	if merger.runtime_viz_recorder is not None:
		merger.runtime_viz_recorder.record_event(
			merge_step=merger.runtime_merge_step,
			stage="recording_finished",
			event_type="recording_finished",
			submap_id=None,
			keyframe_id=None,
			payload={"output_map_path": args.output_map_path},
		)

if __name__ == '__main__':
	args = parse_arguments()

	if args.warning:
		logging_level = logging.WARNING
	else:
		logging_level = logging.INFO
	logging.basicConfig(
		level=logging_level,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[logging.StreamHandler()]
	)
	log_dir = setup_log_environment(pathlib.Path(args.output_map_path), args)

	# Initialize the map merging pipeline
	merger = MergePipeline(args, log_dir)
	merger.init_vpr_match_model()
	merger.init_pose_estimator()
	merger.read_map_from_file()

	perform_submap_merging(merger, args)
