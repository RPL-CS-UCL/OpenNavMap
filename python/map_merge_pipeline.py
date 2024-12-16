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
from utils.vpr_single_matching import PlaceRecognitionSingleMatching

import pycpptools.src.python.utils_math as pytool_math
import pycpptools.src.python.utils_ros as pytool_ros
import pycpptools.src.python.utils_sensor as pytool_sensor

from gtsam_pose_graph import PoseGraph

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

		if args.vpr_match_model == 'single_match':
			self.vpr_match_model = PlaceRecognitionSingleMatching()
		elif args.vpr_match_model == 'topo_filter':
			self.vpr_match_model = PlaceRecognitionTopologicalFilter()
		else:
			raise ValueError(f"Invalid VPR Match Model: {args.vpr_match_model}")

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
				self.args.image_size,
				depth_scale=0.0,
				load_rgb=True,
				load_depth=False,
				normalized=False
			)
			self.submaps.append((submap_id, image_graph))
			print(f"Loaded {submap_id}th {image_graph} from {submap_path}")

	# TODO(gogojjh): Adjust the noise model
	def create_pose_graph_from_submaps(self, submapA, submapB, edges_nodeA_to_nodeB, std_rot_deg=1.0, std_tsl=0.01):
		# Convert the base graph to a gtsam pose graph
		pose_graph = PoseGraph()
		prior_sigma = np.array([np.deg2rad(std_rot_deg), np.deg2rad(std_rot_deg), np.deg2rad(std_rot_deg), std_tsl, std_tsl, std_tsl])
		odom_sigma = np.array([np.deg2rad(std_rot_deg), np.deg2rad(std_rot_deg), np.deg2rad(std_rot_deg), std_tsl, std_tsl, std_tsl])
		loop_sigma = np.array([np.deg2rad(std_rot_deg), np.deg2rad(std_rot_deg), np.deg2rad(std_rot_deg), std_tsl, std_tsl, std_tsl])

		# Create a pose graph from submapA by adding internal edges of submapA
		for node in submapA.nodes.values():
			curr_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(node.trans, node.quat)
			pose_graph.add_init_estimate(node.id, curr_pose3)
			# Add prior factor
			if node.id == 0: pose_graph.add_prior_factor(node.id, curr_pose3, prior_sigma)
			# Add odometry factor
			for edge in node.edges:
				next_node = edge[0]
				# Avoid duplicate factors
				if node.id < next_node.id:
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
				# Avoid duplicate factors
				if node.id < next_node.id:				
					next_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(next_node.trans, next_node.quat)
					pose_graph.add_odometry_factor(node.id + id_offset, curr_pose3, next_node.id + id_offset, next_pose3, odom_sigma)

		# Add the loop factor
		for edge in edges_nodeA_to_nodeB:
			nodeA, nodeB = edge[0], edge[1]
			I_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(np.zeros(3), np.array([0, 0, 0, 1]))
			trans, quat = pytool_math.tools_eigen.convert_matrix_to_vec(edge[2], 'xyzw')
			next_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(trans, quat)
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
			merger.vpr_match_model.initialize_model(db_descriptors, db_poses[:, :3], recall_values=3)

			##############################
			###### DEBUG(gogojjh):
			query_result_info = np.zeros((cur_submap.get_num_node(), 3), dtype="float32")
			preds = []
			##############################
			edges_nodeA_to_nodeB_coarse = [] # [(db_node, query_node, np.eye(4), prob)]
			for query_node in cur_submap.nodes.values():
				# Incrementally update the belief of the full posterior
				query_desc = query_node.get_descriptor()
				recall_preds, pred, prob = merger.vpr_match_model.match(final_map, query_desc.reshape(1, -1))
				# Create connected edges for the coarse localization
				EDGE_PROB_THRE = 0.2
				if prob > EDGE_PROB_THRE:
					edges_nodeA_to_nodeB_coarse.append((final_map.get_node(pred), query_node, np.eye(4), prob))
				preds.append(recall_preds)
				query_result_info[query_node.id, 0] = prob
				print(query_node.id, recall_preds, prob)

			##############################
			###### DEBUG(gogojjh):
			succ_cnt = 0
			for edge in edges_nodeA_to_nodeB_coarse: 
				db_node, query_node = edge[0], edge[1]
			print(f"Coarse Loc Results with the Submap {cur_submap_id}")
			# save_vis_vpr(merger.log_dir, final_map, cur_submap, cur_submap_id, np.array(preds), suffix=f'{args.vpr_match_model}_coarse')
			save_vis_pose_graph(merger.log_dir, final_map, cur_submap, cur_submap_id, edges_nodeA_to_nodeB_coarse, suffix=f'{args.vpr_match_model}_coarse')
			##############################

			##### Perform Fine Localization #####
			est_opts = {
				'known_extrinsics': True,
				'known_intrinsics': True,
				'resize': 512,
			}
			edges_nodeA_to_nodeB_refine = [] # [(db_node, query_node, T_A2B)]
			for edge_nodeA_to_nodeB in edges_nodeA_to_nodeB_coarse:
				nodeA, nodeB = edge_nodeA_to_nodeB[0], edge_nodeA_to_nodeB[1]
				# Skip if the nodeA has no edges, which means it cannot recover the metric pose
				if len(nodeA.edges) == 0: continue 
				try:
					# Generate paths of images and intrinsics					
					nodeA_list = [nodeA, nodeA.edges[0][0]]
					list_img0_name = [f"{final_map.map_root.split('/')[-1]}/{node.rgb_img_name}" for node in nodeA_list]
					img1_name = f"{cur_submap.map_root.split('/')[-1]}/{nodeB.rgb_img_name}"
					list_img0_poses = [torch.from_numpy(pytool_math.tools_eigen.convert_vec_to_matrix(node.trans, node.quat, 'xyzw')) 
									   for node in nodeA_list]
					list_img0_intr = [{'K': torch.from_numpy(node.raw_K), 'im_size': torch.from_numpy(node.raw_img_size)} for node in nodeA_list]
					img1_intr = {'K': torch.from_numpy(nodeB.raw_K), 'im_size': torch.from_numpy(nodeB.raw_img_size)}
					scene_root = pathlib.Path(final_map.map_root + '/../')

					# Perform pose estimation
					start_time = time.time()
					result = merger.pose_estimator(scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts)
					edge_scores = merger.pose_estimator.get_edge_score(option='mean')
					max_edge_core_nodeA_nodeA_next = max(edge_scores['0_1'], edge_scores['1_0'])
					max_edge_score_nodeA_nodeB = max((value for key, value in edge_scores.items() if '2' in key), default=0)
					T_nodeA_est = pytool_math.tools_eigen.convert_vec_to_matrix(nodeA.trans, nodeA.quat, 'xyzw')
					T_query_est = result['im_pose']
					T_rel_est = np.linalg.inv(T_nodeA_est) @ T_query_est

					##############################
					##### DEBUG(gogojjh):
					query_result_info[nodeB.id, 1] = max_edge_score_nodeA_nodeB
					print(f"Processing time: {time.time() - start_time:.2f}s")
					print(edge_scores)
					print(Fore.GREEN + f"Max score for query: {max_edge_score_nodeA_nodeB:.3f}")
					T_nodeA_gt = pytool_math.tools_eigen.convert_vec_to_matrix(nodeA.trans_gt, nodeA.quat_gt, 'xyzw')
					T_query_gt = pytool_math.tools_eigen.convert_vec_to_matrix(nodeB.trans_gt, nodeB.quat_gt, 'xyzw')
					T_rel_gt = np.linalg.inv(T_nodeA_gt) @ T_query_gt
					print('EST Rel Pose: ', T_rel_est[:3, 3:4].T)
					print('GT Rel Pose: ', T_rel_gt[:3, 3:4].T)
					dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis_TF(T_rel_est, T_rel_gt)
					print(f"Error in translation: {dis_tsl:.3f} [m] and rotation {dis_angle:.3f} [deg]")
					print(Fore.GREEN + f"Reference: {', '.join(name for name in list_img0_name)}")
					print(Fore.GREEN + f"Target: {img1_name}")
					##############################

					EDGE_SCORE_THRE = 20.0 # threshold to select good refinement: out-of-range image, wrong coarse localization
					if max_edge_score_nodeA_nodeB > EDGE_SCORE_THRE:
						edges_nodeA_to_nodeB_refine.append((nodeA, nodeB, T_rel_est, max_edge_score_nodeA_nodeB))
						print(Fore.RED + f"Good Refinement")
						query_result_info[nodeB.id, 2] = 1.0
					# merger.pose_estimator.show_reconstruction()
					# input()
				except Exception as e:
					print(f"Error in pose estimation: {e}")
					continue
				print()

			##################
			###### DEBUG(gogojjh):
			print(Fore.GREEN + f"Fine Localization Results with the Submap {cur_submap_id} with Edge {len(edges_nodeA_to_nodeB_refine)}")
			save_vis_pose_graph(merger.log_dir, final_map, cur_submap, cur_submap_id, edges_nodeA_to_nodeB_refine, suffix='refine')
			save_query_result(merger.log_dir, query_result_info, cur_submap_id)
			##################

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
	log_dir = setup_log_environment(args.output_map_path, args)

	# Initialize the map merging pipeline
	merger = MergePipeline(args, log_dir)
	rospy.loginfo('Initialize Pose Estimator')
	merger.init_pose_estimator()
	merger.read_map_from_file()

	rospy.init_node('map_merge_pipeline_node', anonymous=True)
	merger.initalize_ros()
	# loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	# loc_pipeline.child_frame_id = rospy.get_param('~child_frame_id', 'camera')

	perform_submap_merging(merger, args)
