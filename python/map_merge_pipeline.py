#! /usr/bin/env python

import os
import sys
import torch
import pathlib
import numpy as np
import logging
import gtsam
# import cv2
# import time
# import copy
# import random

import rospy
# from std_msgs.msg import Header
# from nav_msgs.msg import Odometry, Path
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import PoseArray
# from visualization_msgs.msg import MarkerArray
# import tf2_ros
import matplotlib
# import matplotlib.pyplot as plt
# from PIL import Image
# from tqdm import tqdm

from typing import List, Tuple

# import rospkg
# rospkg = rospkg.RosPack()
# pack_path = rospkg.get_path('litevloc')
# sys.path.append(os.path.join(pack_path, '../image_matching_models'))
# sys.path.append(os.path.join(pack_path, '../image_matching_models'))

# from estimator.utils import to_tensor, to_numpy

from utils.utils_vpr_method import initialize_match_model
from utils.utils_map_merging import *
from utils.utils_geom import convert_vec_to_matrix, convert_matrix_to_vec, convert_vec_gtsam_pose3, compute_pose_error
from image_graph import ImageGraphLoader as GraphLoader
from image_graph import ImageGraph
from image_node import ImageNode

from gtsam_pose_graph import PoseGraph
from codetiming import Timer

from colorama import Fore, init
init(autoreset=True)

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):	matplotlib.use("Agg")

class MergePipeline:
	def __init__(self, args, log_dir):
		self.args = args
		self.log_dir = log_dir
		self.frame_id_map = 'map'
		self.submaps = []

	def init_vpr_match_model(self):
		self.vpr_match_model = initialize_match_model(self.args.vpr_match_model, self.args.vpr_match_seq_len)		
		logging.info(f"VPR Match Model: {self.args.vpr_match_model}")

	def init_pose_estimator(self):
		self.pose_estimator = initialize_pose_estimator(self.args.pose_estimation_method, self.args.device)
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
			image_graph = GraphLoader.load_data(
				submap_path,
				resize=self.args.image_size,
				depth_scale=0.0,
				load_rgb=True,
				load_depth=False,
				normalized=False,
				edge_type='odometry'
			)
			self.submaps.append((submap_id, image_graph))
			print(f"Loaded {submap_id}th {image_graph} from {submap_path}")

	# TODO(gogojjh): Adjust the noise model
	def create_pose_graph_from_submaps(self, submapA, submapB, edges_nodeA_to_nodeB, std_rot_deg=1.0, std_tsl=0.01):
		# Convert the base graph to a gtsam pose graph
		pose_graph = PoseGraph()
		
		sigma = np.array([np.deg2rad(std_rot_deg)] * 3 + [std_tsl] * 3)
		prior_sigma = np.array(sigma)
		odom_sigma = np.array(sigma)
		loop_sigma = np.array(sigma)

		# Create a pose graph from submapA by adding internal edges of submapA
		for node in submapA.nodes.values():
			curr_pose3 = convert_vec_gtsam_pose3(node.trans, node.quat)
			pose_graph.add_init_estimate(node.id, curr_pose3)
			# Add prior factor
			if node.id == 0: pose_graph.add_prior_factor(node.id, curr_pose3, prior_sigma)
			# Add odometry factor
			for edge in node.edges:
				next_node = edge[0]
				# Avoid duplicate factors
				if node.id < next_node.id:
					next_pose3 = convert_vec_gtsam_pose3(next_node.trans, next_node.quat)
					pose_graph.add_odometry_factor(node.id, curr_pose3, next_node.id, next_pose3, odom_sigma)

		# Expand the pose graph from submapB by adding internal edges of submapA
		id_offset = max(submapA.get_all_id()) + 1
		print(f"id offset: {id_offset}")
		for node in submapB.nodes.values():
			curr_pose3 = convert_vec_gtsam_pose3(node.trans, node.quat)
			pose_graph.add_init_estimate(node.id + id_offset, curr_pose3)
			# Add odometry factor
			for edge in node.edges:
				next_node = edge[0]
				# Avoid duplicate factors
				if node.id < next_node.id:				
					next_pose3 = convert_vec_gtsam_pose3(next_node.trans, next_node.quat)
					pose_graph.add_odometry_factor(node.id + id_offset, curr_pose3, next_node.id + id_offset, next_pose3, odom_sigma)

		# Add the loop factor
		for edge in edges_nodeA_to_nodeB:
			nodeA, nodeB = edge[0], edge[1]
			I_pose3 = convert_vec_gtsam_pose3(np.zeros(3), np.array([0, 0, 0, 1]))
			trans, quat = convert_matrix_to_vec(edge[2], 'xyzw')
			next_pose3 = convert_vec_gtsam_pose3(trans, quat)
			pose_graph.add_odometry_factor(nodeA.id, I_pose3, nodeB.id + id_offset, next_pose3, loop_sigma)

		return pose_graph					

	def merge_and_update_submaps(self, submapA, submapB, edges_nodeA_nodeB_weight):
		id_offset = 0 if not submapA.get_all_id() else max(submapA.get_all_id()) + 1
		for node in submapB.nodes.values():
			node.id += id_offset

			if os.path.exists(os.path.join(submapB.map_root, node.rgb_img_name)):
				rgb_img_path = os.path.join(submapB.map_root, node.rgb_img_name)
				node.rgb_img_name = f"seq/{node.id:06d}.color.jpg"
				new_rgb_img_path = os.path.join(submapA.map_root, node.rgb_img_name)
				os.system(f'cp {rgb_img_path} {new_rgb_img_path}')

			if os.path.exists(os.path.join(submapB.map_root, node.depth_img_name)):
				depth_img_path = os.path.join(submapB.map_root, node.depth_img_name)
				node.depth_img_name = f"seq/{node.id:06d}.depth.png"
				new_depth_img_path = os.path.join(submapA.map_root, node.depth_img_name)
				os.system(f'cp {depth_img_path} {new_depth_img_path}')

			submapA.add_node(node)

		for edge in edges_nodeA_nodeB_weight:
			nodeA, nodeB, weight = edge[0], edge[1], edge[2]
			submapA.add_edge_undirected(nodeA, nodeB, weight)

		print(f"Final Map Info: {submapA}")

def perform_global_loc(
	merger: MergePipeline,
	final_map: ImageGraph,
	cur_submap: ImageGraph,
	cur_submap_id: int
) -> List[Tuple[ImageNode, ImageNode, np.ndarray, float]]:
	"""Performs coarse localization between a reference map and a query submap.
	
	This function uses a VPR model to find coarse correspondences between nodes
	in the reference map and the query submap. It optionally applies RANSAC-based
	outlier rejection and saves visualization results.
	
	Args:
		merger: The merger object containing the VPR model and configuration.
		final_map: The reference map containing database nodes.
		cur_submap: The query submap to localize within the reference map.
		cur_submap_id: Identifier for the current submap for logging purposes.
		
	Returns:
		A list of edges representing potential matches between database and query
		nodes. Each edge is a tuple (db_node, query_node, T_A2B, score).
	"""
	# Load descriptors from database and query nodes
	db_descriptors = np.array(
		[node.get_descriptor() for node in final_map.nodes.values()], dtype=np.float32)
	query_descriptors = np.array(
		[node.get_descriptor() for node in cur_submap.nodes.values()], dtype=np.float32)
	
	# Initialize the VPR model with database descriptors
	merger.vpr_match_model.initialize_model(db_descriptors)
	
	# Perform global localization with timing
	with Timer(name="Global Localization", 
			text=Fore.GREEN + "{name} costs: {milliseconds:.3f} ms"):
		
		# Perform initial VPR matching for all query nodes
		connected_indices = []
		for query_node in cur_submap.nodes.values():
			start_idx = max(0, query_node.id - merger.vpr_match_model.seqLen + 1)
			query_descs = query_descriptors[start_idx : query_node.id + 1]
			_, pred, score = merger.vpr_match_model.match(query_descs)
			connected_indices.append((pred, query_node.id, score))
		
		# Apply RANSAC-based outlier rejection if enabled
		best_indices = connected_indices
		lines_coeff = cluster_data = cluster_labels = None
		if getattr(merger.vpr_match_model, 'ENABLE_RANSAC', False):
			D_all = merger.vpr_match_model.compute_diff_matrix(query_descriptors)
			filtered_indices, lines_coeff, cluster_data, cluster_labels = \
				merger.vpr_match_model.ransac_check_match(
					D_all, 
					connected_indices[merger.vpr_match_model.seqLen:]
				)
			best_indices = connected_indices[:merger.vpr_match_model.seqLen] + filtered_indices
		
		# Perform Geomtric Verification
		edges = []
		for db_idx, query_idx, score in best_indices:
			db_node = final_map.get_node(db_idx)	
			query_node = cur_submap.get_node(query_idx)

			result = merger.pose_estimator.get_matched_kpts(merger.scene_root, db_node.rgb_image, query_node.rgb_image)
			num_inlier = result['num_inliers']
			print(Fore.GREEN + f"DB {db_node.id} - Query {query_node.id} - Number of matched kpts: {num_inlier}")

			if num_inlier > REFINE_GV_SCORE_THRESHOLD: 
				edges.append((db_node, query_node, np.eye(4), score))
	
	# Save visualization and debug data
	save_dir = f"{merger.log_dir}/preds"
	if getattr(merger.vpr_match_model, 'ENABLE_RANSAC', False):
		merger.vpr_match_model.save_diff_matrix_fitting(
			save_dir, connected_indices, best_indices, D_all, final_map,
			cur_submap, lines_coeff, cluster_data, cluster_labels)
	save_vis_pose_graph(
		save_dir, final_map, cur_submap, cur_submap_id, edges,
		suffix=f'{merger.args.vpr_match_model}_coarse')
	
	return edges

def perform_local_loc(
	edges_nodeA_to_nodeB_coarse: List[Tuple[ImageNode, ImageNode, np.ndarray, float]],
	merger: MergePipeline,
	final_map: ImageGraph,
	cur_submap: ImageGraph,
) -> List[Tuple[ImageNode, ImageNode, np.ndarray, float]]:
	"""Performs fine-grained localization using pose estimation on coarse matches.
	
	Args:
		edges_nodeA_to_nodeB_coarse: List of coarse matches (db_node, query_node, T_A2B, score)
		final_map: Reference map containing database nodes
		cur_submap: Query submap to localize
		merger: Merger object with pose estimator and configuration
		
	Returns:
		List of refined matches with relative pose estimates
	"""
	est_opts = dict(known_extrinsics=True, known_intrinsics=False, resize=512)
	refined_edges = []

	for edge in edges_nodeA_to_nodeB_coarse:
		db_node, query_node, _, score = edge
		if not db_node.edges: continue
		try:
			# Prepare database references
			db_node_pair = [db_node, db_node.edges[0][0]]
			db_names = [f"{final_map.map_root.split('/')[-1]}/{n.rgb_img_name}" for n in db_node_pair]
			db_poses = [torch.from_numpy(convert_vec_to_matrix(
				n.trans, n.quat, 'xyzw')) for n in db_node_pair]
			db_intrs = [{
				'K': torch.from_numpy(n.raw_K),
				'im_size': torch.from_numpy(n.raw_img_size)
			} for n in db_node_pair]

			# Prepare query data
			query_name = f"{cur_submap.map_root.split('/')[-1]}/{query_node.rgb_img_name}"
			query_intr = {
				'K': torch.from_numpy(query_node.raw_K),
				'im_size': torch.from_numpy(query_node.raw_img_size)
			}

			# Perform pose estimation with timing
			with Timer(name="Pose Estimation", 
			  text=Fore.GREEN + "{name} costs: {milliseconds:.3f} ms"):
				result = merger.pose_estimator(
					merger.scene_root,
					db_names,
					query_name,
					db_poses,
					db_intrs,
					query_intr,
					est_opts
				)
				
				T_db_est = convert_vec_to_matrix(
					db_node.trans, db_node.quat, 'xyzw')
				T_rel_est = np.linalg.inv(T_db_est) @ result['im_pose']

				##############################
				##### DEBUG(gogojjh):				
				T_nodeA_gt = convert_vec_to_matrix(db_node.trans_gt, db_node.quat_gt, 'xyzw')
				T_query_gt = convert_vec_to_matrix(query_node.trans_gt, query_node.quat_gt, 'xyzw')
				T_rel_gt = np.linalg.inv(T_nodeA_gt) @ T_query_gt
				print('EST Rel Pose: ', T_rel_est[:3, 3:4].T)
				print('GT Rel Pose: ', T_rel_gt[:3, 3:4].T)
				dis_tsl, dis_angle = compute_pose_error(T_rel_est, T_rel_gt, 'matrix')
				print(f"Error in translation: {dis_tsl:.3f} [m] and rotation {dis_angle:.3f} [deg]")
				print(Fore.GREEN + f"Reference: {', '.join(name for name in db_names)}")
				print(Fore.GREEN + f"Target: {query_name}")
				##############################

			# Add to refined matches if score exceeds threshold
			if score > REFINE_EDGE_SCORE_THRESHOLD:
				refined_edges.append((db_node, query_node, T_rel_est, score))
				
		except Exception as e:
			print(f"{Fore.RED} Pose estimation failed: {str(e)}")
			continue

	return refined_edges

def perform_submap_merging(merger: MergePipeline, args):
	"""Main loop for processing submap merging"""
	assert len(merger.submaps) > 0, "No submaps loaded."
	print(f"Processing {len(merger.submaps)} submaps.")
	
	# Initialize the final submap
	final_map = ImageGraph(merger.log_dir)
	merger.scene_root = pathlib.Path(final_map.map_root + '/../')

	# Incrementally merge each submap to the final map, only care about the first two submaps
	for cur_submap_id, cur_submap in merger.submaps:
		if final_map.get_num_node() > 0:
			edges_nodeA_to_nodeB_coarse = perform_global_loc(merger, final_map, cur_submap, cur_submap_id)
			exit()
			edges_nodeA_to_nodeB_refine = perform_local_loc(edges_nodeA_to_nodeB_coarse, merger, final_map, cur_submap)
			print()

			##################
			###### DEBUG(gogojjh):
			# print(Fore.GREEN + f"Fine Localization Results with the Submap {cur_submap_id} with Edge {len(edges_nodeA_to_nodeB_refine)}")
			# save_vis_pose_graph(merger.log_dir, final_map, cur_submap, cur_submap_id, edges_nodeA_to_nodeB_refine, suffix='refine')
			# save_query_result(merger.log_dir, query_result_info, cur_submap_id)
			# ##################

			##### Perform Pose Graph Optimization #####
			pose_graph = merger.create_pose_graph_from_submaps(final_map, cur_submap, edges_nodeA_to_nodeB_refine)
			g2o_file_path = os.path.join(merger.log_dir, "preds/initial_pose_graph.g2o")
			gtsam.writeG2o(pose_graph.get_factor_graph(), pose_graph.get_initial_estimate(), g2o_file_path)
			# NOTE(gogojjh): add optimization
			# pose_graph.perform_optimization()

			##### Merge the Pose Graph and Update Poses of Each Node #####
			edges_nodeA_nodeB_weight = []
			for edge in edges_nodeA_to_nodeB_refine:
				weight = np.linalg.norm(edge[2][:3, 3])
				edges_nodeA_nodeB_weight.append([edge[0], edge[1], weight])
			merger.merge_and_update_submaps(final_map, cur_submap, edges_nodeA_nodeB_weight)
		else:
			merger.merge_and_update_submaps(final_map, cur_submap, [])
	
	final_map.save_to_file()

if __name__ == '__main__':
	import warnings
	warnings.filterwarnings("ignore", category=FutureWarning)

	args = parse_arguments()
	log_dir = setup_log_environment(args.output_map_path, args)

	# Initialize the map merging pipeline
	merger = MergePipeline(args, log_dir)
	rospy.loginfo('Initialize Pose Estimator')
	merger.init_vpr_match_model()
	merger.init_pose_estimator()
	merger.read_map_from_file()

	rospy.init_node('map_merge_pipeline_node', anonymous=True)
	merger.initalize_ros()
	# loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	# loc_pipeline.child_frame_id = rospy.get_param('~child_frame_id', 'camera')

	perform_submap_merging(merger, args)
