#! /usr/bin/env python

"""
Usage: 
python loc_pipeline.py \
--dataset_path /Rocket_ssd/dataset/data_litevloc/matterport3d/out_17DRP5sb8fy/out_map \
--image_size 512 288 --device=cuda \
--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=256 --save_descriptors --num_preds_to_save 3 \
--img_matcher master --save_img_matcher \
--pose_solver pnp --config_pose_solver config/dataset/matterport3d.yaml \
--viz \
--global_pos_threshold 20.0 --min_inliers_threshold 300

Usage: 
rosbag record -O /Titan/dataset/data_litevloc/anymal_lab_upstair_20240722_0/vloc.bag \
/vloc/odometry /vloc/path /vloc/path_gt /vloc/image_map_obs
"""

import os
import sys
import torch
import cv2
import pathlib
import numpy as np
import time
import copy
import logging
import gtsam

import rospy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray
import tf2_ros
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# import rospkg
# rospkg = rospkg.RosPack()
# pack_path = rospkg.get_path('litevloc')
# sys.path.append(os.path.join(pack_path, '../image_matching_models'))
# sys.path.append(os.path.join(pack_path, '../image_matching_models'))

# from estimator.utils import to_tensor, to_numpy

from utils.utils_vpr_method import perform_knn_search
from utils.utils_map_merging import *
from utils.utils_image import load_rgb_image, load_depth_image
from image_graph import ImageGraphLoader as GraphLoader
from image_graph import ImageGraph
from image_node import ImageNode
from utils.vpr_topological_filter import PlaceRecognitionTopologicalFilter

import pycpptools.src.python.utils_math as pytool_math
import pycpptools.src.python.utils_ros as pytool_ros
import pycpptools.src.python.utils_sensor as pytool_sensor

from gtsam_pose_graph import PoseGraph

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):	matplotlib.use("Agg")

class MergePipeline:
	def __init__(self, args, log_dir):
		self.args = args
		self.log_dir = log_dir
		self.frame_id_map = 'map'

		self.submaps = []

	# def init_vpr_model(self):
	# 	self.vpr_model = initialize_vpr_model(self.args.vpr_method, self.args.vpr_backbone, self.args.vpr_descriptors_dimension, self.args.device)
	# 	logging.info(f"VPR Model: {self.args.vpr_method}")

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
		num_submap = self.args.num_submap
		for i in range(num_submap):
			submap_id = len(self.submaps)
			submap_path = os.path.join(self.args.dataset_path, f'out_map{submap_id}')
			image_graph = GraphLoader.load_data(
				submap_path,
				self.args.image_size,
				depth_scale=0.0,
				load_rgb=True,
				load_depth=False,
				normalized=False
			)
			self.submaps.append((submap_id, image_graph))
			logging.info(f"Loaded {image_graph} from {submap_path}")

		print(f"Loaded {len(self.submaps)} submaps.")

	def create_pose_graph_from_submaps(self, submapA, submapB, edges_nodeA_to_nodeB, std_rot_deg=1.0, std_tsl=0.01):
		# Convert the base graph to a gtsam pose graph
		pose_graph = PoseGraph()
		prior_sigma = np.array([np.deg2rad(std_rot_deg), np.deg2rad(std_rot_deg), np.deg2rad(std_rot_deg), std_tsl, std_tsl, std_tsl])
		odom_sigma = np.array([np.deg2rad(std_rot_deg), np.deg2rad(std_rot_deg), np.deg2rad(std_rot_deg), std_tsl, std_tsl, std_tsl])

		# Create a pose graph from submapA by adding internal edges of submapA
		for node in submapA.nodes.values():
			curr_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(node.trans, node.quat)
			pose_graph.add_init_estimate(node.id, curr_pose3)
			# Add prior factor
			if node.id == 0: pose_graph.add_prior_factor(node.id, curr_pose3, prior_sigma)
			# Add odometry factor
			for edge in node.edges:
				next_node = edge[0]
				next_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(next_node.trans, next_node.quat)
				pose_graph.add_odometry_factor(node.id, curr_pose3, next_node.id, next_pose3, odom_sigma)

		# Expand the pose graph from submapB by adding internal edges of submapA
		id_offset = max(submapA.get_all_id()) + 1
		for node in submapB.nodes.values():
			curr_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(node.trans, node.quat)
			pose_graph.add_init_estimate(node.id + id_offset, curr_pose3)
			# Add odometry factor
			for edge in node.edges:
				next_node = edge[0]
				next_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(next_node.trans, next_node.quat)
				pose_graph.add_odometry_factor(node.id + id_offset, curr_pose3, next_node.id + id_offset, next_pose3, odom_sigma)

		# Expand the pose graph from adding external edges from the submapA to the submapB
		for edge in edges_nodeA_to_nodeB:
			nodeA, nodeB = edge[0], edge[1]
			I_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(np.zeros(3), np.array([0, 0, 0, 1]))
			trans, quat = pytool_math.tools_eigen.convert_matrix_to_vec(edge[2], 'xyzw')
			next_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(trans, quat)
			pose_graph.add_odometry_factor(nodeA.id, I_pose3, nodeB.id + id_offset, next_pose3, odom_sigma)

		return pose_graph					

	def merge_and_update_submaps(self, submapA, submapB, edges_nodeA_nodeB_weight):
		"""
		Merges two submaps and updates the node IDs, image names, and adds edges.
		This operation will change values of submapB

		Args:
			submapA (Submap): The first submap to merge.
			submapB (Submap): The second submap to merge.
			edges_nodeA_nodeB_weight (list): A list of tuples representing the edges between nodes in submapA and submapB.
				Each tuple contains three elements: nodeA, nodeB, and weight.

		Returns:
			None

		Raises:
			None
		"""
		id_offset = 0 if not submapA.get_all_id() else max(submapA.get_all_id()) + 1
		for node in submapB.nodes.values():
			node.id += id_offset

			if os.path.exists(os.path.join(submapB.map_root, node.rgb_img_name)):
				rgb_img = Image.open(os.path.join(submapB.map_root, node.rgb_img_name))
				node.rgb_img_name = f"seq/{node.id:06d}.color.jpg"
				rgb_img.save(os.path.join(submapA.map_root, node.rgb_img_name))

			if os.path.exists(os.path.join(submapB.map_root, node.depth_img_name)):
				depth_img = Image.open(os.path.join(submapB.map_root, node.depth_img_name))
				node.depth_img_name = f"seq/{node.id:06d}.depth.png"
				depth_img.save(os.path.join(submapA.map_root, node.depth_img_name))

			submapA.add_node(node)
		for edge in edges_nodeA_nodeB_weight:
			nodeA, nodeB, weight = edge[0], edge[1], edge[2]
			submapA.add_edge_undirected(nodeA, nodeB, weight)

		print(f"Final Map Info: {submapA}")
		for node in submapA.nodes.values():
			print(f"Node: {node.id}, {node.rgb_img_name}")

def perform_submap_merging(merger: MergePipeline, args):
	"""Main loop for processing submap merging"""
	assert len(merger.submaps) > 0, "No submaps loaded."
	print(f"Processing {len(merger.submaps)} submaps.")

	# Initialize the final submap
	final_map = ImageGraph(merger.log_dir)
	# Incrementally merge each submap to the final map, only care about the first two submaps
	for cur_submap_id, cur_submap in merger.submaps:
		if final_map.get_num_node() > 0:
			##### Perform Coarse Localization #####
			# Load global descriptors and poses from the reference map 
			db_descriptors = np.array([node.get_descriptor() for _, node in final_map.nodes.items()], dtype="float32")
			db_poses = np.empty((final_map.get_num_node(), 7), dtype="float32")
			for indices, (_, node) in enumerate(final_map.nodes.items()):
				db_poses[indices, :3] = node.trans
				db_poses[indices, 3:] = node.quat

			######################################
			# NOTE(gogojjh): single matching
			print('Single Matching')
			from utils.vpr_single_matching import PlaceRecognitionSingleMatching
			single_matcher = PlaceRecognitionSingleMatching(db_descriptors, db_poses[:, :3], recall_values=5)
			single_matcher.initialize_model()
			# Load global descriptors of each node from current target submap, and incrementally update the belief
			preds = []
			succ = 0
			for node in cur_submap.nodes.values():
				query_desc = node.get_descriptor()
				recall_preds, pred, prob = single_matcher.match(query_desc.reshape(1, -1))
				preds.append(recall_preds)
				# print(recall_preds)
				dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
					final_map.get_node(pred).trans_gt, final_map.get_node(pred).quat_gt, node.trans_gt, node.quat_gt)
				if dis_tsl < 10.0 and dis_angle < 90.0:
					succ += 1				
			print(f"Accuracy: {succ / len(cur_submap.nodes):.3f}")

			# NOTE(gogojjh): topological filter
			print('Topological Filter')
			topo_filter = PlaceRecognitionTopologicalFilter(db_descriptors, db_poses[:, :3], recall_values=5)
			topo_filter.initialize_model()
			# Load global descriptors of each node from current target submap, and incrementally update the belief
			preds = []
			succ = 0
			for node in cur_submap.nodes.values():
				query_desc = node.get_descriptor()
				recall_preds, pred, prob = topo_filter.match(final_map, query_desc.reshape(1, -1))
				preds.append(recall_preds)
				print(recall_preds)
				dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
					final_map.get_node(pred).trans_gt, final_map.get_node(pred).quat_gt, node.trans_gt, node.quat_gt)
				if dis_tsl < 10.0 and dis_angle < 90.0:
					succ += 1		
			print(f"Accuracy: {succ / len(cur_submap.nodes):.3f}")
			exit()
			######################################
			
			# TODO(gogojjh): add local matching to filter out the false positives
			# if edge_score > edge_score_threshold:
				# Create connected edges
			
			# TODO(gogojjh): add the virtual edge if the submap has no connection with the final map

			##########################################
			# NOTE(gogojjh): old code
			# query_poses = np.empty((cur_submap.get_num_node(), 7), dtype="float32")
			# for indices, (_, node) in enumerate(cur_submap.nodes.items()):
			# 	query_poses[indices, :3] = node.trans
			# 	query_poses[indices, 3:] = node.quat
			# Perform kNN search
			# dist, preds = perform_knn_search(db_descriptors, query_descriptors, db_descriptors.shape[1], recall_values=[5])
			##########################################

			# Create connected edges
			edges_nodeA_to_nodeB_coarse = []
			for query_node_id in range(preds.shape[0]):
				query_node = cur_submap.get_node(query_node_id)
				db_node_id = preds[query_node_id][0]
				db_node = final_map.get_node(db_node_id)
				edges_nodeA_to_nodeB_coarse.append((db_node, query_node, np.eye(4)))
			###### DEBUG(gogojjh):
			print("Coarse Localization Results:")
			query_descriptors = np.array([node.get_descriptor() for _, node in cur_submap.nodes.items()], dtype="float32")
			print(f"Size of DB and Query Descriptions: {db_descriptors.shape}, {query_descriptors.shape}")
			print(f"Performing kNN search for submap {cur_submap_id} with {len(preds)} predictions.\n", preds)
			for edge in edges_nodeA_to_nodeB_coarse: print(f"DB: {edge[0].rgb_img_name} <-> Query: {edge[1].rgb_img_name}")
			save_vis_coarse_loc(merger.log_dir, final_map, cur_submap, cur_submap_id, preds)
			input()
			######

			##### Perform Fine Localization #####
			edges_nodeA_to_nodeB_refine = [] # [(nodeA, nodeB, T_A2B)]
			for edge_nodeA_to_nodeB in edges_nodeA_to_nodeB_coarse:
				nodeA, nodeB = edge_nodeA_to_nodeB[0], edge_nodeA_to_nodeB[1]
				if len(nodeA.edges) == 0: continue # Skip if the nodeA has no edges
				nodeA_list = [nodeA, nodeA.edges[0][0]]
				# Generate paths of images and intrinsics					
				list_img0_name = [f"{final_map.map_root.split('/')[-1]}/{node.rgb_img_name}" for node in nodeA_list]
				img1_name = f"{cur_submap.map_root.split('/')[-1]}/{nodeB.rgb_img_name}"
				list_img0_poses = [torch.from_numpy(pytool_math.tools_eigen.convert_vec_to_matrix(node.trans, node.quat, 'xyzw')) for node in nodeA_list]
				list_img0_intr = [{'K': torch.from_numpy(node.raw_K), 'im_size': torch.from_numpy(node.raw_img_size)} for node in nodeA_list]
				img1_intr = {'K': torch.from_numpy(nodeB.raw_K), 'im_size': torch.from_numpy(nodeB.raw_img_size)}
				scene_root = pathlib.Path(final_map.map_root + '/../')
				est_opts = {
					'known_extrinsics': True,
					'known_intrinsics': True,
					'resize': 512,
				}
				try:
					# start_time = time.time()
					result = merger.pose_estimator(scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts)
					edge_scores = merger.pose_estimator.get_edge_score()
					# print(f"Processing time: {time.time() - start_time:.2f}s")
					
					Twc0_est = pytool_math.tools_eigen.convert_vec_to_matrix(nodeA.trans, nodeA.quat, 'xyzw')
					Twc1_est = result['im_pose']
					T_c0_c1_est = np.linalg.inv(Twc0_est) @ Twc1_est
					print('Estimated pose: ', T_c0_c1_est[:3, 3:4].T)

					Twc0_gt = pytool_math.tools_eigen.convert_vec_to_matrix(nodeA.trans_gt, nodeA.quat_gt, 'xyzw')
					Twc1_gt = pytool_math.tools_eigen.convert_vec_to_matrix(nodeB.trans_gt, nodeB.quat_gt, 'xyzw')
					T_c0_c1_gt = np.linalg.inv(Twc0_gt) @ Twc1_gt
					print('GT pose: ', T_c0_c1_gt[:3, 3:4].T)

					dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis_TF(T_c0_c1_est, T_c0_c1_gt)
					print(f"Error in translation: {dis_tsl:.3f} [m] and rotation {dis_angle:.3f} [deg]")
					# print('Optimization Loss:', result['loss'])
					# TODO(gogojjh): use the information of the estimator to check
					if dis_tsl < 0.75 and dis_angle < 20: 
						print(f"Reference: {', '.join(name for name in list_img0_name)}")
						print(f"Target: {img1_name}")
						edges_nodeA_to_nodeB_refine.append((nodeA, nodeB, T_c0_c1_est))
					# TODO(gogojjh): fix this bug
					# merger.pose_estimator.show_reconstruction()
					# input()
				except Exception as e:
					print(f"Error in pose estimation: {e}")
					continue
				print()
			###### DEBUG(gogojjh):
			print("Fine Localization Results:")			
			for edge in edges_nodeA_to_nodeB_refine: 
				print(f"DB: {edge[0].rgb_img_name}, {edge[0].edges[0][0].rgb_img_name} <-> Query: {edge[1].rgb_img_name}")
			save_vis_pose_graph(merger.log_dir, final_map, cur_submap, cur_submap_id, edges_nodeA_to_nodeB_refine)
			######

			##### Perform Pose Graph Optimization #####
			pose_graph = merger.create_pose_graph_from_submaps(final_map, cur_submap, edges_nodeA_to_nodeB_refine)
			g2o_file_path = os.path.join(merger.log_dir, "preds/initial_pose_graph.g2o")
			gtsam.writeG2o(pose_graph.get_factor_graph(), pose_graph.get_initial_estimate(), g2o_file_path)
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
	str_suffix = '_'.join([f'{i}' for i in range(args.num_submap)])
	out_dir = os.path.join(args.dataset_path, 'out_map_' + str_suffix)
	log_dir = setup_log_environment(out_dir, args)

	# Initialize the map merging pipeline
	merger = MergePipeline(args, log_dir)
	# NOTE(gogojjh): no need for VPR
	# rospy.loginfo('Initialize VPR Method')
	# merger.init_vpr_model()
	rospy.loginfo('Initialize Pose Estimator')
	merger.init_pose_estimator()
	merger.read_map_from_file()

	rospy.init_node('map_merge_pipeline_node', anonymous=True)
	merger.initalize_ros()
	# loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	# loc_pipeline.child_frame_id = rospy.get_param('~child_frame_id', 'camera')

	perform_submap_merging(merger, args)
