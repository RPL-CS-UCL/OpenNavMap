#! /usr/bin/env python

import os
import sys
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
from utils.utils_map_merging import *
from utils.utils_geom import convert_vec_to_matrix, convert_matrix_to_vec, compute_pose_error
from utils.utils_geom import convert_vec_gtsam_pose3, convert_matrix_gtsam_pose3
from utils.gtsam_pose_graph import PoseGraph
from benchmark_kf_selection.metric.landmark_selector import LandmarkSelector

from map_manager import MapManager
from image_graph import ImageGraphLoader as GraphLoader
from image_graph import ImageGraph
from image_node import ImageNode

from colorama import Fore, init
init(autoreset=True)

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):	matplotlib.use("Agg")

class MergePipeline:
	def __init__(self, args, log_dir: pathlib.Path):
		self.args = args
		self.log_dir = log_dir
		self.frame_id_map = 'map'

		self.submaps = []
		self.id_offset = 0

		self.lm_selector = LandmarkSelector()

		self.graph_configs = {
			'odom': {},
			'trav': {},
			'covis': {
				'resize': self.args.image_size,
				'depth_scale': 0.0,
				'load_rgb': True,
				'load_depth': False,
				'normalized': False
			},
		}

		self.est_opts = {
			'known_extrinsics': True, 
			'known_intrinsics': False, 
			'niter': 300
		}

	def init_vpr_match_model(self):
		self.vpr_match_model = initialize_match_model(self.args.vpr_match_model, self.args.vpr_match_seq_len)		
		logging.info(f"VPR Match Model: {self.args.vpr_match_model}")

	def init_pose_estimator(self):
		self.pose_estimator = initialize_pose_estimator(
			self.args.pose_estimation_method, 
			self.args.device
		)
		logging.info(f"Pose Estimator: {self.args.pose_estimation_method}")

	def initalize_ros(self):
		# self.pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
		# self.pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)
		
		# self.pub_odom = rospy.Publisher('/vloc/odometry', Odometry, queue_size=10)
		# self.pub_path = rospy.Publisher('/vloc/path', Path, queue_size=10)
		# self.pub_path_gt = rospy.Publisher('/vloc/path_gt', Path, queue_size=10)
		# self.pub_map_obs = rospy.Publisher('/vloc/image_map_obs', Image, queue_size=10)

		# self.br = tf2_ros.TransformBroadcaster()
		# self.path_msg = Path()
		# self.path_gt_msg = Path()
		pass

	def read_map_from_file(self):
		for submap_path in self.args.input_submap_path:
			submap_id = len(self.submaps)
			submap = MapManager(pathlib.Path(submap_path), submap_id)
			submap.load_graphs(merger.graph_configs)
			self.submaps.append(submap)

			print(f"Loaded {submap.map_id} from {submap_path}")

	def create_pose_graph_from_map(
		self, 
		graph_odom_a,     # The odometry graph 
		graph_odom_b,     # The odometry graph 
		inter_edges_covis # inter_edges_covis own the same node id with odom
	):
		# Set basic std for factors
		prior_sigma = np.array([1e-3] * 3 + [1e-2] * 3)
		odom_sigma = np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3)
		loop_sigma = np.array([np.deg2rad(3.0)] * 3 + [1.0] * 3)
		pose_graph = PoseGraph()
		I_pose3 = convert_matrix_gtsam_pose3(np.eye(4))

		# Create a pose graph from graph_odom_a by adding internal edges of graph_odom_a
		for _, node in graph_odom_a.nodes.items():
			curr_pose3 = convert_vec_gtsam_pose3(node.trans, node.quat)
			pose_graph.add_init_estimate(node.id, curr_pose3)
			# Add prior factor
			if node.id == 0:
				pose_graph.add_prior_factor(node.id, curr_pose3, prior_sigma)
			# Add odometry factor
			for edge in node.edges:
				next_node = edge[0]
				# Avoid duplicate factors
				if node.id < next_node.id:
					next_pose3 = convert_vec_gtsam_pose3(next_node.trans, next_node.quat)
					pose_graph.add_odometry_factor(
						node.id, curr_pose3, 
						next_node.id, next_pose3, 
						odom_sigma
					)
		
		# Create a pose graph from graph_odom_b by adding internal edges of graph_odom_b
		for _, node in graph_odom_b.nodes.items():
			curr_pose3 = convert_vec_gtsam_pose3(node.trans, node.quat)
			pose_graph.add_init_estimate(node.id + self.id_offset, curr_pose3)
			# Add odometry factor
			for edge in node.edges:
				next_node = edge[0]
				# Avoid duplicate factors
				if node.id < next_node.id:
					next_pose3 = convert_vec_gtsam_pose3(next_node.trans, next_node.quat)
					pose_graph.add_odometry_factor(
						node.id+self.id_offset, curr_pose3, 
						next_node.id+self.id_offset, next_pose3, 
						odom_sigma
					)
		
		# Add the loop factor
		for edge in inter_edges_covis:
			nodeA, nodeB, T_AB, conf = edge
			trans, quat = convert_matrix_to_vec(T_AB)
			next_pose3 = convert_vec_gtsam_pose3(trans, quat)
			update_loop_sigma = loop_sigma / conf
			pose_graph.add_odometry_factor(
				nodeA.id, I_pose3, 
				nodeB.id+self.id_offset, next_pose3, 
				update_loop_sigma
			)

		return pose_graph					

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

		print(f"Merged map info - {submap_a}")
		
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
	# msk_conf = estimator.scene.get_masks()
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

			# Mask for overlapping points
			valid_mask = (proj_depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
			proj_depth_map = torch.zeros(H, W, device=device)
			proj_depth_map[v[valid_mask], u[valid_mask]] = proj_depth[valid_mask]
			u, v = u[valid_mask], v[valid_mask]
			proj_depth = proj_depth[valid_mask]
			msk = torch.abs(proj_depth - depthmaps[j][v, u].reshape(1, -1)) < 0.5 * depthmaps[j][v, u].reshape(1, -1)

			redu = np.sum(msk.detach().cpu().numpy()) / (len(pts3d_flat))
			if j == len(all_pts3d) - 1:
				lm_gain_pw.append(('db', all_nodes[i], all_nodes[j], 1.0 - redu))
			else:
				lm_gain_pw.append(('query', all_nodes[i], all_nodes[j], 1.0 - redu))

			# DEBUG(gogojjh):
			if False:
				import matplotlib.pyplot as plt
				fig, axs = plt.subplots(1, 2, figsize=(16, 12))
				im0 = axs[0].imshow(depthmaps[j].detach().cpu().numpy(), cmap='turbo')
				axs[0].set_title(f'Original Depth Camera {j} onto Camera {j}')
				plt.colorbar(im0, ax=axs[0], label='Depth')
				
				im1 = axs[1].imshow(proj_depth_map.detach().cpu().numpy(), cmap='turbo')
				axs[1].set_title(f'Projected Depth of Camera {i} onto Camera {j})')
				plt.colorbar(im1, ax=axs[1], label='Depth')

				plt.tight_layout()
				plt.savefig(os.path.join('/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_aria/s00001/out_map_test/preds', f'depth_maps_{i}_to_{j}.jpg'))
				plt.close()

	return lm_gain_pw

def perform_global_loc(
	merger: MergePipeline,
	final_graph: ImageGraph,
	cur_graph: ImageGraph,
	cur_graph_id: int
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
		
	Returns:
		A list of edges representing potential matches between database and query
		nodes. Each edge is a tuple (db_node, query_node, T_A2B, score).
	"""
	# Load descriptors from database and query nodes
	num_db_nodes = final_graph.get_num_node()
	db_descriptors = np.array(
		[node.get_descriptor() for node in final_graph.nodes.values()], dtype=np.float32
	)
	db_node_ids = [node.id for node in final_graph.nodes.values()]

	num_query_nodes = cur_graph.get_num_node()
	query_descriptors = np.array(
		[node.get_descriptor() for node in cur_graph.nodes.values()], dtype=np.float32
	)
	query_node_ids = [node.id for node in cur_graph.nodes.values()]

	merger.vpr_match_model.initialize_model(db_descriptors)
	with Timer(name="Global Localization", text=Fore.GREEN + "{name} costs: {milliseconds:.3f} ms"):
		# VPR matching for all query nodes
		connected_indices = []
		for row in range(num_query_nodes):
			start_row = max(0, row - merger.vpr_match_model.seqLen + 1)
			query_descs = query_descriptors[start_row : row + 1]
			_, pred, score = merger.vpr_match_model.match(query_descs)
			connected_indices.append((db_node_ids[pred], query_node_ids[row], score))

		if hasattr(merger.vpr_match_model, 'compute_diff_matrix'):
			D_all = merger.vpr_match_model.compute_diff_matrix(query_descriptors)
		else:
			D_all = None
		
		# RANSAC-based outlier rejection on the difference matrix if enabled
		best_indices = connected_indices
		lines_coeff = cluster_data = cluster_labels = None
		if getattr(merger.vpr_match_model, 'ENABLE_RANSAC', False):
			filtered_indices, lines_coeff, cluster_data, cluster_labels = \
				merger.vpr_match_model.ransac_check_match(
					D_all, 
					connected_indices[merger.vpr_match_model.seqLen:]
				 )
			best_indices = connected_indices[:merger.vpr_match_model.seqLen] + filtered_indices
		
		# Geomtric Verification
		coarse_edges = []
		for db_idx, query_idx, _ in best_indices:
			db_node = final_graph.get_node(db_idx)	
			query_node = cur_graph.get_node(query_idx)
			result = merger.pose_estimator.get_matched_kpts(
				final_graph.map_root, db_node.rgb_image, query_node.rgb_image
			)
			num_inlier = result['num_inliers']
			print(Fore.GREEN + f"DB {db_node.id} - Query {query_node.id} - Number of matched kpts: {num_inlier}")
			if num_inlier > REFINE_GV_SCORE_THRESHOLD: 
				coarse_edges.append((db_node, query_node, np.eye(4), num_inlier))
	
	##### Save visualization and debug data
	save_dir = str(merger.log_dir/"preds")
	if D_all is not None:
		merger.vpr_match_model.save_diff_matrix_fitting(
			save_dir, connected_indices, best_indices, D_all, final_graph,
			cur_graph, lines_coeff, cluster_data, cluster_labels)
	save_vis_pose_graph(
		save_dir, final_graph, cur_graph, cur_graph_id, coarse_edges,
		suffix=f'{merger.args.vpr_match_model}_coarse')
	
	return coarse_edges

def perform_local_loc(
	edges_nodeA_to_nodeB_coarse: List[Tuple[ImageNode, ImageNode, np.ndarray, float]],
	merger: MergePipeline,
	final_graph: ImageGraph,
	cur_graph: ImageGraph,
	cur_graph_id: int,
) -> Tuple[
	List[Tuple[ImageNode, ImageNode, np.ndarray, float]],
	Dict[ImageNode, Dict[ImageNode, float]],
	Dict[ImageNode, Dict[ImageNode, float]]
]:
	"""Performs fine-grained localization using pose estimation on coarse matches.
	
	Args:
		edges_nodeA_to_nodeB_coarse: List of coarse matches (db_node, query_node, T_A2B, score)
		final_graph: Reference map containing database nodes
		cur_graph: Query submap to localize
		merger: Merger object with pose estimator and configuration
		
	Returns:
		List of refined matches (represented as image node) with relative pose estimates
	"""
	# lm_gain_db[nodeA][nodeB] meaning how much information is gained of nodeA w.r.t. nodeB, and nodeA is a db_node
	# lm_gain_query[nodeA][nodeB] meaning how much information is gained of nodeA w.r.t. nodeB, and nodeA is a query_node
	lm_gain_db, lm_gain_query = dict(), dict()
	
	refined_edges = []
	for edge in tqdm(edges_nodeA_to_nodeB_coarse):
		db_node, query_node, _, _ = edge
		# Check whether the node has more than one edge
		if not db_node.edges: 
			continue
		try:
			# Prepare database references
			db_node_pair = [db_node, db_node.edges[0][0]]
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
				##### DEBUG(gogojjh):				
				T_nodeA_gt = convert_vec_to_matrix(db_node.trans_gt, db_node.quat_gt, 'xyzw')
				T_query_gt = convert_vec_to_matrix(query_node.trans_gt, query_node.quat_gt, 'xyzw')
				T_rel_gt = np.linalg.inv(T_nodeA_gt) @ T_query_gt
				print('EST Rel Pose: ', T_rel_est[:3, 3:4].T)
				print('GT Rel Pose: ', T_rel_gt[:3, 3:4].T)
				dis_tsl, dis_angle = compute_pose_error(T_rel_est, T_rel_gt, 'matrix')
				print(f"Error in translation: {dis_tsl:.3f} [m] and rotation {dis_angle:.3f} [deg]")
				##############################

				##############################
				##### Store immedinate results for subsequent keyframe selection
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
				##############################

				# Add to refined matches if score exceeds threshold
				top_k_matches = len(db_names) # default: 2
				if hasattr(merger.pose_estimator, 'get_minimum_spanning_tree'):
					msp_edges = merger.pose_estimator.get_minimum_spanning_tree()
					weight_i, weight_j = merger.pose_estimator.scene.weight_i, merger.pose_estimator.scene.weight_j
					for edge in msp_edges:
						if edge[0] == top_k_matches or edge[1] == top_k_matches: # confidence of the query image
							edge_str = f"{edge[0]}_{edge[1]}"
							conf = \
								weight_i[edge_str].detach().cpu().numpy().mean() * \
								weight_j[edge_str].detach().cpu().numpy().mean()

					print(Fore.GREEN + f"{db_names[0]} {db_names[1]} - {query_name} with conf: {conf:.3f}")
					if conf > REFINE_CONF_THRESHOLD:
						refined_edges.append((db_node, query_node, T_rel_est, conf))
				else:
					conf = MAX_LOSS - result["loss"]
					refined_edges.append((db_node, query_node, T_rel_est, conf))
				
		except Exception as e:
			print(f"{Fore.RED} Pose estimation failed: {str(e)}")
			continue

	##### Save visualization and debug data
	save_dir = str(merger.log_dir/"preds")
	save_vis_pose_graph(
		save_dir, final_graph, cur_graph, cur_graph_id, refined_edges,
		suffix=f'{merger.args.vpr_match_model}_refine')

	return refined_edges, lm_gain_db, lm_gain_query

def perform_submap_merging(merger: MergePipeline, args):
	"""Main loop for processing submap merging"""
	assert len(merger.submaps) > 0, "No submaps loaded."
	print(f"Processing {len(merger.submaps)} submaps.")
	
	# Initialize the final submap
	final_map = MapManager(merger.log_dir)
	final_map.init_graphs(merger.graph_configs)

	# Incrementally merge each submap to the final map, only care about the first two submaps
	for cur_submap in merger.submaps:
		# The offset of node.id in cur_submap 
		merger.id_offset = final_map.get_max_node_id() + 1		
		
		if not final_map.is_empty:
			# Identify coarse covisibility relationship 
			edges_nodeAB_coarse_covis = perform_global_loc(
				merger,
				final_map.covis, 
				cur_submap.covis, 
				cur_submap.map_id
			)
			# Identify strong covisibility relationship 
			edges_nodeAB_refine_covis, lm_gain_db, lm_gain_query = perform_local_loc(
				edges_nodeAB_coarse_covis, 
				merger, 
				final_map.covis,
				cur_submap.covis, 
				cur_submap.map_id
			)

			##### Perform Pose Graph Optimization #####
			# Initialize the pose graph
			print(Fore.GREEN + f'Performing PGO for Submap {cur_submap.map_id}' + Fore.RESET)
			pose_graph = merger.create_pose_graph_from_map(
				final_map.odom,
				cur_submap.odom, 
				edges_nodeAB_refine_covis
			)	
			g2o_path = str(merger.log_dir/"preds/initial_pose_graph.g2o")
			gtsam.writeG2o(pose_graph.get_factor_graph(), pose_graph.get_initial_estimate(), g2o_path)
			
			# Optimize the pose graph
			print(f"PGO: initial error = {pose_graph.get_factor_graph().error(pose_graph.get_initial_estimate()):.3f}")
			result_pgo = PoseGraph.optimize_pose_graph_with_LM(
				pose_graph.get_factor_graph(), 
				pose_graph.get_initial_estimate(), 
				verbose=False
			)
			print(f"PGO: final error = {pose_graph.get_factor_graph().error(result_pgo):.3f}")
			for key in result_pgo.keys():
				update_estimate = result_pgo.atPose3(key)
				pose_graph.add_init_estimate(key, update_estimate)
			g2o_path = str(merger.log_dir/"preds/refine_pose_graph.g2o")
			gtsam.writeG2o(pose_graph.get_factor_graph(), pose_graph.get_initial_estimate(), g2o_path)
			
			##### Perform keyframe pruning #####
			# Steps: check pruning probability -> remove old nodes for covis graph
			# But nodes in odom and trav graph are kepts
			# The id of removed nodes will not replaced by new nodes to avoid conflict with odom and trav graph
			if args.select_keyframe:
				# Compute the acceptance probability:
				# 	Accept the new keyframe with high information gain, even it has low image quality
				nodes_query_to_remove = []
				for nodeA, data in lm_gain_query.items():
					acc_prob_dict = 0.0
					for _, gain in data.items():
						acc_prob_dict = max(
							merger.lm_selector.compute_accept_prob(nodeA.iqa_data, gain),
							acc_prob_dict
						)
					if acc_prob_dict < merger.lm_selector.P_acc_th:
						print(f"Remove Submap1 {nodeA.id} lower than Accept Prob:{acc_prob_dict:.3f}")
						save_vis_kf_removal(
							merger.log_dir, 
							nodeA.id,
							nodeA.rgb_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
						)
						nodes_query_to_remove.append(nodeA)

				# Compute the keeping probability:
				# 	Remove the old keyframe with the low information gain and low image quality
				nodes_db_to_remove = []
				for nodeA, data in lm_gain_db.items():
					min_prob, node_rep = 1.0, None
					print(f"DB: {nodeA.rgb_img_name}")
					for nodeB, gain in data.items():
						if nodeB in nodes_query_to_remove:
							continue
						prob = merger.lm_selector.compute_keep_prob(
							nodeA.iqa_data-nodeB.iqa_data, gain, nodeB.time-nodeA.time
						)
						if prob < min_prob:
							min_prob, node_rep = prob, nodeB

					if min_prob < merger.lm_selector.P_keep_th and node_rep:
						nodes_db_to_remove.append(nodeA)
						print(f"Replace Submap0 {nodeA.id} with Submap1 {node_rep.id} with Prob:{min_prob:.3f}")
						merger.lm_selector.print_each_prob(
							nodeA.iqa_data-node_rep.iqa_data, lm_gain_db[nodeA][node_rep], node_rep.time-nodeA.time
						)
						save_vis_kf_replacement(
							merger.log_dir, 
							nodeA.id,
							node_rep.id,
							nodeA.rgb_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), 
							node_rep.rgb_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
						)
				
				# Remove nodes and invalid edges from the graph
				print('Removing nodes from cur_submap covis')
				cur_submap.covis.remove_node_list(nodes_query_to_remove)
				cur_submap.covis.remove_invalid_edges()
				
				print('Removing nodes from cur_submap odom')
				final_map.covis.remove_node_list(nodes_db_to_remove)			
				final_map.covis.remove_invalid_edges()
				final_map.covis.rm_sensor_data(nodes_db_to_remove)

			##### Perform map update and merging
			# Merge two submap into one with optimized poses
			merger.merge_and_update_submaps(final_map, cur_submap, pose_graph.get_initial_estimate())

			# Update edges from the src_edges for different types of graphs
			# Nodes are merged and reflected on the updated graph
			# node_a = final_map.graphs[graph_type].get_node(edges_nodeAB[0])
			# node_b = final_map.graphs[graph_type].get_node(edges_nodeAB[1])
			weight_func1 = (lambda edge: edge[3])
			weight_func2 = (lambda edge: np.linalg.norm(edge[0].trans - edge[1].trans))
			for dst_graph_type, src_edges, weight_func in [
				("covis", edges_nodeAB_refine_covis, weight_func1),
				("odom", edges_nodeAB_refine_covis, weight_func2),
				("trav", edges_nodeAB_refine_covis, weight_func2)
			]:
				dst_edges = final_map.update_edges(src_edges, dst_graph_type)
				final_map.graphs[dst_graph_type].add_inter_edges(dst_edges, weight_func)

			print(f"Final map info:\n{final_map}")

			if args.viz:
				save_dir = str(merger.log_dir/"preds")
				pose_graph.plot_pose_graph(save_dir, pose_graph.get_factor_graph(), result_pgo)
		else:
			pose_graph = merger.create_pose_graph_from_map(
				final_map.odom, 
				cur_submap.odom, 
				[]
			)
			g2o_path = str(merger.log_dir/"preds/initial_pose_graph.g2o")
			gtsam.writeG2o(pose_graph.get_factor_graph(), pose_graph.get_initial_estimate(), g2o_path)
			merger.merge_and_update_submaps(final_map, cur_submap, pose_graph.get_initial_estimate())

	if not final_map.is_empty:
		final_map.save_to_file()

if __name__ == '__main__':
	import warnings
	warnings.filterwarnings("ignore", category=FutureWarning)

	args = parse_arguments()
	log_dir = setup_log_environment(pathlib.Path(args.output_map_path), args)

	# Initialize the map merging pipeline
	merger = MergePipeline(args, log_dir)
	# rospy.loginfo('Initialize Pose Estimator')
	merger.init_vpr_match_model()
	merger.init_pose_estimator()
	merger.read_map_from_file()

	# rospy.init_node('map_merge_pipeline_node', anonymous=True)
	# merger.initalize_ros()
	# loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	# loc_pipeline.child_frame_id = rospy.get_param('~child_frame_id', 'camera')

	perform_submap_merging(merger, args)
