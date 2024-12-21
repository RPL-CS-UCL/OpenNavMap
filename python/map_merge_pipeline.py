#! /usr/bin/env python

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
import random

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
from tqdm import tqdm

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
from utils.vpr_sequence_matching import PlaceRecognitionSeqMatching

import pycpptools.src.python.utils_math as pytool_math
import pycpptools.src.python.utils_ros as pytool_ros
import pycpptools.src.python.utils_sensor as pytool_sensor

from gtsam_pose_graph import PoseGraph

from colorama import Fore, init
init(autoreset=True)

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):	matplotlib.use("Agg")

RMSE_THRESHOLD = 3.0
VPR_MATCH_THRESHOLD = 0.9

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
		elif args.vpr_match_model == 'sequence_match':
			self.vpr_match_model = PlaceRecognitionSeqMatching()
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
				normalized=False,
				edge_type='odometry'
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
			db_poses = np.array([np.hstack((node.trans, node.quat)) for node in final_map.nodes.values()])
			query_descriptors = np.array([node.get_descriptor() for _, node in cur_submap.nodes.items()], dtype="float32")
			query_poses = np.array([np.hstack((node.trans, node.quat)) for node in cur_submap.nodes.values()])
			
			# Initialize the VPR match model
			merger.vpr_match_model.initialize_model(db_descriptors, recall_values=5)
			
			##############################
			###### DEBUG(gogojjh):
			query_result_info = np.zeros((cur_submap.get_num_node(), 3), dtype="float32")
			preds = []
			##############################
			connected_indices = [] # [(pred_db_id, query_node_id, score)]
			start_time = time.time()
			for query_node in tqdm(cur_submap.nodes.values()):
				# NOTE(gogojjh): degrade to single-image-matching if not enough queries
				query_descs = query_descriptors[max(0, query_node.id-merger.vpr_match_model.seqLen+1) : query_node.id+1]
				recall_preds, pred, score = merger.vpr_match_model.match(query_descs, query_node.id)
				if score >= VPR_MATCH_THRESHOLD: continue
				connected_indices.append((pred, query_node.id, score))
				preds.append(recall_preds)
			print(f"Sequence Matching found {len(connected_indices)} edges")
			print(f"Sequence Matching Costs: {time.time() - start_time:.3f}s")
			
			# RANSAC-based reliable edges extraction
			best_min_rmse, best_indices, best_align_R_t_s = None, None, None
			for i in range(10):
				best_min_rmse, best_indices, best_align_R_t_s = \
					merger.vpr_match_model.ransac_check_match(db_poses, query_poses, connected_indices)
				print(f"Error: {best_min_rmse:.3f} - Candidates Size: {len(connected_indices)} - Best Indices Size: {len(best_indices)}")
				if best_min_rmse < RMSE_THRESHOLD: break
				best_min_rmse, best_indices, best_align_R_t_s = None, None, None

			if best_min_rmse is None:
				print(f"No Reliable Loops Found")
				exit()

			# Augment the edges for the subsequent fine localization
			ind_str = {f"{ind[0]}_{ind[1]}" for ind in best_indices}
			augment_indices = random.sample(connected_indices, max(1, len(connected_indices) // 2))
			R, t, s = best_align_R_t_s[0], best_align_R_t_s[1], best_align_R_t_s[2]
			for ind in augment_indices:
				if f"{ind[0]}_{ind[1]}" in ind_str: continue
				db_node, query_node = final_map.get_node(ind[0]), cur_submap.get_node(ind[1])
				dis = np.linalg.norm(R @ query_node.trans + t - db_node.trans)
				if dis >= best_min_rmse: continue
				best_indices.append(ind)
				ind_str.add(f"{ind[0]}_{ind[1]}")
			print(f"All inds: {len(connected_indices)} - Best inds: {len(best_indices)}")
			print(f"RMSE of traj alignment: {best_min_rmse:.3f}")

			T_init = np.block([[R, t.reshape(3, 1)], [0, 0, 0, 1]])
			edges_nodeA_to_nodeB_coarse = [
				(final_map.get_node(ind[0]), cur_submap.get_node(ind[1]), T_init, ind[2]) 
				for ind in best_indices
			]  # [(pred_db_node, query_node, T_A2B, score)]
			##############################
			###### DEBUG(gogojjh):
			succ = 0
			for edge in edges_nodeA_to_nodeB_coarse:
				db_node, query_node = edge[0], edge[1]
				dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
					query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt)
				if dis_tsl < 10.0:
					succ += 1
					print(f"Correct prediction: Query {query_node.id} - DB: {db_node.id}")
				else:
					print(f"Wrong prediction: Query {query_node.id} - DB: {db_node.id}")
			print(f"Success Rate: {succ / len(best_indices):.3f} with {len(best_indices)} Edges")
			# save_vis_vpr(merger.log_dir, final_map, cur_submap, cur_submap_id, np.array(preds), suffix=f'{args.vpr_match_model}_coarse')
			save_vis_pose_graph(merger.log_dir, final_map, cur_submap, cur_submap_id, 
								edges_nodeA_to_nodeB_coarse, suffix=f'{args.vpr_match_model}_coarse')
			exit()
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
