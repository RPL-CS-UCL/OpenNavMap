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
		print(f"id offset: {id_offset}")
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
			merger.vpr_match_model.initialize_model(db_descriptors)
			
			##############################
			###### DEBUG(gogojjh):
			query_result_info = np.zeros((cur_submap.get_num_node(), 4), dtype="float32")
			##############################
			connected_indices = [] # [(pred_db_id, query_node_id, score)]
			timer_global_loc = Timer(name="Global Localization", text=Fore.GREEN + "{name} costs: {milliseconds:.3f} ms")
			timer_global_loc.start()
			for query_node in tqdm(cur_submap.nodes.values()):
				query_descs = query_descriptors[max(0, query_node.id-merger.vpr_match_model.seqLen+1) : query_node.id+1]
				_, pred, score = merger.vpr_match_model.match(query_descs, backward=False)
				connected_indices.append((pred, query_node.id, score))
				query_result_info[query_node.id, 0] = score
			D_all = merger.vpr_match_model._compute_diff_matrix(query_descriptors)
			best_indices, lines_coeff, cluster_data, cluster_labels = \
				merger.vpr_match_model.ransac_check_match(D_all, connected_indices)
			timer_global_loc.stop()
			
			edges_nodeA_to_nodeB_coarse = [(
				final_map.get_node(ind[0]), cur_submap.get_node(ind[1]), np.eye(4), ind[2]) 
				for ind in best_indices
			]  # [(pred_db_node, query_node, T_A2B, score)]

			##############################
			###### DEBUG(gogojjh):
			print(f"Sequence Matching found {len(best_indices)} edges")
			tp, fp = 0, 0
			for edge in edges_nodeA_to_nodeB_coarse:
				db_node, query_node = edge[0], edge[1]
				dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(
					query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt
				)
				if dis_tsl < 20.0:
					tp += 1
					print(f"Correct prediction: Query {query_node.id} - DB: {db_node.id}")
				else:
					fp += 1
					print(f"Wrong prediction: Query {query_node.id} - DB: {db_node.id}")
			if tp + fp < 1:
				precision = 0
			else:
				precision = tp / (tp+fp)
			print(f"Precision: {precision:.3f} - {tp}/{tp+fp}")
			# save_vis_vpr(merger.log_dir, final_map, cur_submap, cur_submap_id, np.array(preds), suffix=f'{args.vpr_match_model}_coarse')
			save_vis_pose_graph(
				merger.log_dir, 
				final_map, 
				cur_submap, 
				cur_submap_id, 
				edges_nodeA_to_nodeB_coarse, 
				suffix=f'{args.vpr_match_model}_coarse'
			)

			if True:
				fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

				im1 = ax1.imshow(D_all, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
				for edge in connected_indices:
					db_node = final_map.get_node(edge[0])
					query_node = cur_submap.get_node(edge[1])
					dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(
						query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt
					)
					if dis_tsl < 20.0:
						ax1.plot(edge[1], edge[0], 'go', markersize=5)
					else:
						ax1.plot(edge[1], edge[0], 'ro', markersize=5)

				fig.colorbar(im1, ax=ax1, label='Difference')
				ax1.set_xlabel('Query Descriptor Index')
				ax1.set_ylabel('Database Descriptor Index')
				ax1.set_title("Difference Matrix [Before RANSAC]")

				im2 = ax2.imshow(D_all, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
				for edge in best_indices:
					db_node = final_map.get_node(edge[0])
					query_node = cur_submap.get_node(edge[1])
					dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(
						query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt
					)
					if dis_tsl < 10.0:
						ax2.plot(edge[1], edge[0], 'go', markersize=5)
					else:
						ax2.plot(edge[1], edge[0], 'ro', markersize=5)

				for line_coeff in lines_coeff:
					m, b = line_coeff
					x_vals = np.linspace(0, D_all.shape[1], 100)
					y_vals = m * x_vals + b
					ax2.plot(x_vals, y_vals, 'r-', linewidth=1)

				fig.colorbar(im2, ax=ax2, label='Difference')
				ax2.set_xlabel('Query Descriptor Index')
				ax2.set_ylabel('Database Descriptor Index')
				ax2.set_title(f"Difference Matrix [After RANSAC] Precision: {precision:.3f} - {tp}/{tp+fp}")
				ax2.set_xlim(0, D_all.shape[1])
				ax2.set_ylim(0, D_all.shape[0])
				ax2.invert_yaxis()

				im3 = ax3.imshow(D_all, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
				scatter = ax3.scatter(cluster_data[:, 0], cluster_data[:, 1], c=cluster_labels, cmap='rainbow', s=20)
				fig.colorbar(scatter, ax=ax3, label='Cluster Label')
				ax3.set_xlabel('Query Descriptor Index')
				ax3.set_ylabel('Database Descriptor Index')
				ax3.set_title(f"Dot Cluster")
				ax3.set_xlim(0, D_all.shape[1])
				ax3.set_ylim(0, D_all.shape[0])
				ax3.invert_yaxis()

				plt.savefig(f"{merger.log_dir}/preds/difference_matrix_fitting.jpg", dpi=300, bbox_inches='tight')
				plt.close()
			################################################
			# exit()
			################################################

			##### Perform Fine Localization #####
			est_opts = {
				'known_extrinsics': True,
				'known_intrinsics': True,
				'resize': 512,
			}
			edges_nodeA_to_nodeB_refine = [] # [(db_node, query_node, T_A2B)]
			for edge_nodeA_to_nodeB in edges_nodeA_to_nodeB_coarse:
				nodeA, nodeB = edge_nodeA_to_nodeB[0], edge_nodeA_to_nodeB[1]
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

					# Perform Geomtric Verification
					timer_gv = Timer(name="Geometric Verification", text=Fore.GREEN + "{name} costs: {milliseconds:.3f} ms")
					timer_gv.start()
					result = merger.pose_estimator.get_matched_kpts(scene_root, nodeA_list[0].rgb_image, nodeB.rgb_image)
					num_inlier0 = result['num_inliers']
					print(Fore.GREEN + f"DB {nodeA_list[0].id} - Query {nodeB.id} - Number of matched kpts: {num_inlier0}")
					result = merger.pose_estimator.get_matched_kpts(scene_root, nodeA_list[1].rgb_image, nodeB.rgb_image)
					num_inlier1 = result['num_inliers']
					print(Fore.GREEN + f"DB {nodeA_list[1].id} - Query {nodeB.id} - Number of matched kpts: {num_inlier1}")
					timer_gv.stop()
					max_num_inliers = max(num_inlier0, num_inlier1)
					if max_num_inliers < REFINE_GV_SCORE_THRESHOLD: continue

					# Perform pose estimation
					timer_pe = Timer(name="Pose Estimation", text=Fore.GREEN + "{name} costs: {milliseconds:.3f} ms")
					timer_pe.start()
					result = merger.pose_estimator(scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts)
					edge_scores = merger.pose_estimator.get_similarity(option='mean')
					max_edge_score_nodeA_nodeA_next = max(edge_scores['0_1'], edge_scores['1_0'])
					max_edge_score_nodeA_nodeB = max((value for key, value in edge_scores.items() if '2' in key), default=0)
					T_nodeA_est = pytool_math.tools_eigen.convert_vec_to_matrix(nodeA.trans, nodeA.quat, 'xyzw')
					T_query_est = result['im_pose']
					T_rel_est = np.linalg.inv(T_nodeA_est) @ T_query_est
					timer_pe.stop()

					##############################
					##### DEBUG(gogojjh):
					query_result_info[nodeB.id, 1] = max_edge_score_nodeA_nodeB
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

					# edges_nodeA_to_nodeB_refine.append((nodeA, nodeB, T_rel_est, max_edge_score_nodeA_nodeB))
					# print(Fore.RED + f"Good Refinement")
					# query_result_info[nodeB.id, 2] = 1.0

					# NOTE(gogojjh): not use edge score to determine good refinement
					if max_edge_score_nodeA_nodeB > REFINE_EDGE_SCORE_THRESHOLD:
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
	merger.init_pose_estimator()
	merger.read_map_from_file()

	rospy.init_node('map_merge_pipeline_node', anonymous=True)
	merger.initalize_ros()
	# loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	# loc_pipeline.child_frame_id = rospy.get_param('~child_frame_id', 'camera')

	perform_submap_merging(merger, args)
