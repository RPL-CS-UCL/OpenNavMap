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
from image_node import ImageNode

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
		submap_id = 0
		for i in range(num_submap):
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
			submap_id += 1

		print(f"Loaded {len(self.submaps)} submaps.")

	def process_submap(self):
		assert len(self.submaps) > 0, "No submaps loaded."
		print(f"Processing {len(self.submaps)} submaps.")
		# Using deep copy to avoid modifying the original submaps
		ref_submap_id, ref_submap = 0, copy.deepcopy(self.submaps[0][1])
		# Merge each submap to the final map
		for cur_submap_id, cur_submap in self.submaps[1:]:
			##### Perform Coarse Localization #####
			# Load global descriptors and poses from the reference map 
			db_descriptors = np.array([node.get_descriptor() for _, node in ref_submap.nodes.items()], dtype="float32")
			db_poses = np.empty((ref_submap.get_num_node(), 7), dtype="float32")
			for indices, (_, node) in enumerate(ref_submap.nodes.items()):
				db_poses[indices, :3] = node.trans
				db_poses[indices, 3:] = node.quat
			# Load global descriptors and poses from the current target submap
			query_descriptors = np.array([node.get_descriptor() for _, node in cur_submap.nodes.items()], dtype="float32")
			query_poses = np.empty((cur_submap.get_num_node(), 7), dtype="float32")
			for indices, (_, node) in enumerate(cur_submap.nodes.items()):
				query_poses[indices, :3] = node.trans
				query_poses[indices, 3:] = node.quat
			# Perform kNN search
			dist, preds = perform_knn_search(db_descriptors, query_descriptors, db_descriptors.shape[1], recall_values=[5])
			# Create connected edges
			edges_nodeA_to_nodeB = []
			for query_node_id in range(preds.shape[0]):
				query_node = cur_submap.get_node(query_node_id)
				db_node_id = preds[query_node_id][0]
				db_node = ref_submap.get_node(db_node_id)
				edges_nodeA_to_nodeB.append((db_node, query_node, np.eye(4)))
			###### DEBUG(gogojjh):
			for edge in edges_nodeA_to_nodeB: print(f"Graph0: {edge[0].id} <-> Graph1: {edge[1].id}")
			print(f"Size of DB and Query Descriptions: {db_descriptors.shape}, {query_descriptors.shape}")
			print(f"Performing kNN search for submap {cur_submap_id} with {len(preds)} predictions.\n", preds)
			save_vis_coarse_loc(self.log_dir, ref_submap, ref_submap_id, cur_submap, cur_submap_id, preds)
			save_pg_coarse_loc(self.log_dir, ref_submap, ref_submap_id, cur_submap, cur_submap_id, edges_nodeA_to_nodeB)
			######

			##### Perform Fine Localization #####
			for edge_nodeA_to_nodeB in edges_nodeA_to_nodeB:
				nodeA, nodeB = edge_nodeA_to_nodeB[0], edge_nodeA_to_nodeB[1]
				if len(nodeA.edges) == 0: continue # Skip if the nodeA has no edges
				nodeA_list = [nodeA, nodeA.edges[0][0]]
				# Generate paths of images and intrinsics					
				list_img0_name = [f'out_map0/{node.rgb_img_name}' for node in nodeA_list]
				img1_name = f'out_map{cur_submap_id}/{nodeB.rgb_img_name}'			
				list_img0_poses = [torch.from_numpy(pytool_math.tools_eigen.convert_vec_to_matrix(node.trans, node.quat, 'xyzw')) for node in nodeA_list]
				list_img0_intr = [{'K': torch.from_numpy(node.raw_K), 'im_size': torch.from_numpy(node.raw_img_size)} for node in nodeA_list]
				img1_intr = {'K': torch.from_numpy(nodeB.raw_K), 'im_size': torch.from_numpy(nodeB.raw_img_size)}
				scene_root = pathlib.Path(ref_submap.map_root + '/../')
				est_opts = {
					'known_extrinsics': True,
					'known_intrinsics': True,
					'resize': 512,
				}
				start_time = time.time()
				result = self.pose_estimator(scene_root, list_img0_name, img1_name, list_img0_poses, list_img0_intr, img1_intr, est_opts)
				print('Reference:\n', list_img0_name)
				print('Target:\n', img1_name)
				print(f"Processing time: {time.time() - start_time:.2f}s")
				print('Estimated pose: ', result['im_pose'][:3, 3:4].T) # Pose from world to camera
				# print('Loss:', result['loss'])
				
				Twc0 = pytool_math.tools_eigen.convert_vec_to_matrix(nodeA.trans_gt, nodeA.quat_gt, 'xyzw')
				Twc1 = pytool_math.tools_eigen.convert_vec_to_matrix(nodeB.trans_gt, nodeB.quat_gt, 'xyzw')
				T_c0_c1 = np.linalg.inv(Twc0) @ Twc1
				# TODO(gogojjh):
				print('GT pose: ', T_c0_c1[:3, 3].T)

				# self.pose_estimator.show_reconstruction()
				input()

			##### Perform Pose Graph Optimization #####
			pose_graph = self.create_pose_graph_from_submaps(ref_submap, cur_submap, edges_nodeA_to_nodeB)
			g2o_file_path = os.path.join(self.log_dir, "preds/pose_graph.g2o")
			gtsam.writeG2o(pose_graph.get_factor_graph(), pose_graph.get_initial_estimate(), g2o_file_path)
			# pose_graph.perform_optimization()

			##### Merge the Pose Graph #####
			ref_submap.merge(cur_submap, edges_nodeA_to_nodeB)
			print(ref_submap)

	def create_pose_graph_from_submaps(self, submapA, submapB, edges_nodeA_to_nodeB):
		# Convert the base graph to a gtsam pose graph
		pose_graph = PoseGraph()
		prior_sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
		odom_sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
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
			trans, qxyzw = pytool_math.tools_eigen.convert_matrix_to_vec(edge[2])
			next_pose3 = pytool_math.tools_eigen.convert_vec_gtsam_pose3(trans, qxyzw)
			pose_graph.add_odometry_factor(nodeA.id, I_pose3, nodeB.id + id_offset, next_pose3, odom_sigma)

		return pose_graph					

	# def perform_image_matching(self, matcher, map_node, obs_node):
	# 	try:
	# 		matcher_result = matcher(map_node.rgb_image, obs_node.rgb_image)
	# 		"""Save matching results"""
	# 		if self.args.save_img_matcher:
	# 			num_inliers, H, mkpts0, mkpts1 = (
	# 				matcher_result["num_inliers"],
	# 				matcher_result["H"],
	# 				matcher_result["inliers0"],
	# 				matcher_result["inliers1"],
	# 			)
	# 			save_img_matcher_visualization(
	# 				obs_node.rgb_image, map_node.rgb_image,
	# 				mkpts0, mkpts1, self.log_dir, obs_node.id, n_viz=100)		
	# 		return matcher_result
	# 	except Exception as e:
	# 		logging.error(f"Error in image matching: {e}")
	# 		null_kpts = np.zeros((0, 2), dtype=np.float32)
	# 		return {"num_inliers_kpts": 0, "num_inliers": 0, "inliers0": null_kpts, "inliers1": null_kpts}

	# def perform_global_loc(self, save=False):
	# 	_, vpr_pred = self.perform_vpr(self.DB_DESCRIPTORS, self.curr_obs_node.get_descriptor())
	# 	if len(vpr_pred[0]) == 0: return {'succ': False, 'map_id': None}
	# 	if save:
	# 		list_of_images_paths = [self.curr_obs_node.rgb_img_path]
	# 		for i in range(len(vpr_pred[0, :self.args.num_preds_to_save])):
	# 			map_node = self.image_graph.get_node(vpr_pred[0, i])
	# 			list_of_images_paths.append(map_node.rgb_img_path)
	# 		preds_correct = [None] * len(list_of_images_paths)
	# 		save_vpr_visualization(self.log_dir, 0, list_of_images_paths, preds_correct)
	# 	return {'succ': True, 'map_id': vpr_pred[0, 0]}
	
	# def perform_local_loc(self):
	# 	matching_start_time = time.time()
	# 	ref_node = self.search_keyframe_from_graph(self.curr_obs_node)
	# 	if ref_node is None: return {'succ': False, 'T_w_obs': None, 'solver_inliers': 0}
	# 	self.ref_map_node = ref_node

	# 	matcher_result = self.perform_image_matching(self.img_matcher, self.ref_map_node, self.curr_obs_node)
	# 	num_kpts_inliers, num_H_inliers = matcher_result["num_inliers_kpts"], matcher_result["num_inliers"]
	# 	mkpts0, mkpts1 = (matcher_result["inliers0"], matcher_result["inliers1"])
	# 	mkpts0_raw = mkpts0 * [self.ref_map_node.raw_img_size[0] / self.ref_map_node.img_size[0], 
	# 							self.ref_map_node.raw_img_size[1] / self.ref_map_node.img_size[1]]
	# 	mkpts1_raw = mkpts1 * [self.curr_obs_node.raw_img_size[0] / self.curr_obs_node.img_size[0], 
	# 							self.curr_obs_node.raw_img_size[1] / self.curr_obs_node.img_size[1]]
	# 	self.ref_map_node.set_matched_kpts(mkpts0, num_kpts_inliers)
	# 	self.curr_obs_node.set_matched_kpts(mkpts1, num_kpts_inliers)
	# 	rospy.loginfo(f'Number of kpts inliers: {num_kpts_inliers}, H inliers: {num_H_inliers}')
	# 	rospy.loginfo(f"Image matching costs: {time.time() - matching_start_time: .3f}s")

	# 	if num_kpts_inliers < self.args.min_kpts_inliers_thre:
	# 		rospy.logwarn(f'[Fail] No sufficient matching kpts')
	# 		return {'succ': False, 'T_w_obs': None, 'solver_inliers': 0}
	# 	try:
	# 		if self.args.img_matcher == "mickey": # Not used in this project
	# 			R, t = self.img_matcher.scene["R"].squeeze(0), self.img_matcher.scene["t"].squeeze(0)
	# 			R, t = to_numpy(R), to_numpy(t)
	# 			num_solver_inliers = self.img_matcher.scene["inliers"]
	# 		else:
	# 			depth_img1 = to_numpy(self.curr_obs_node.depth_image.squeeze(0))
	# 			R, t, num_solver_inliers = self.pose_solver.estimate_pose(
	# 				mkpts1_raw, mkpts0_raw,
	# 				self.curr_obs_node.raw_K, self.ref_map_node.raw_K,
	# 				depth_img1, None)
	# 		if num_solver_inliers < self.args.min_solver_inliers_thre:
	# 			rospy.logwarn(f'[Fail] No sufficient number {num_solver_inliers} solver inliers')
	# 			return {'succ': False, 'T_w_obs': None, 'num_solver_inliers': 0}
	# 		else:
	# 			T_mapnode_obs = np.eye(4)
	# 			T_mapnode_obs[:3, :3], T_mapnode_obs[:3, 3] = R, t.reshape(3)
	# 			T_w_mapnode = pytool_math.tools_eigen.convert_vec_to_matrix(self.ref_map_node.trans_gt, self.ref_map_node.quat_gt, 'xyzw')
	# 			T_w_obs = T_w_mapnode @ T_mapnode_obs
	# 			rospy.logwarn(f'[Succ] sufficient number {num_solver_inliers} solver inliers')
	# 			return {'succ': True, 'T_w_obs': T_w_obs, 'solver_inliers': num_solver_inliers}
	# 	except Exception as e:
	# 		rospy.logwarn(f'[Fail] to estimate pose with error:', e)
	# 		return {'succ': False, 'T_w_obs': None, 'solver_inliers': num_solver_inliers}

	# def publish_message(self):
	# 	header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id_map)
	# 	tf_msg = pytool_ros.ros_msg.convert_vec_to_rostf(np.array([0, 0, -2.0]), np.array([0, 0, 0, 1]), header, f"{self.frame_id_map}_graph")
	# 	self.br.sendTransform(tf_msg)
	# 	header = Header(stamp=rospy.Time.now(), frame_id=f"{self.frame_id_map}_graph")
	# 	pytool_ros.ros_vis.publish_graph(self.image_graph, header, self.pub_graph, self.pub_graph_poses)

	# 	if self.curr_obs_node is not None:
	# 		header = Header(stamp=rospy.Time.from_sec(self.curr_obs_node.time), frame_id=self.frame_id_map)
			
	# 		# Publish odometry and path if the local position is available
	# 		if self.has_local_pos:
	# 			odom = pytool_ros.ros_msg.convert_vec_to_rosodom(self.curr_obs_node.trans, self.curr_obs_node.quat, header, self.child_frame_id)
	# 			self.pub_odom.publish(odom)
	# 			pose_msg = pytool_ros.ros_msg.convert_odom_to_rospose(odom)
				
	# 			self.path_msg.header = header
	# 			self.path_msg.poses.append(pose_msg)
	# 			self.pub_path.publish(self.path_msg)

	# 		if self.curr_obs_node.has_pose_gt:
	# 			pose_msg = pytool_ros.ros_msg.convert_vec_to_rospose(self.curr_obs_node.trans_gt, self.curr_obs_node.quat_gt, header)
	# 			self.path_gt_msg.header = header
	# 			self.path_gt_msg.poses.append(pose_msg)
	# 			self.pub_path_gt.publish(self.path_gt_msg)

	# 		if self.ref_map_node is not None and self.args.viz:
	# 			n_viz = 10 # visualize n_viz matched keypoints
	# 			rgb_img_ref = (np.transpose(to_numpy(self.ref_map_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
	# 			rgb_img_obs = (np.transpose(to_numpy(self.curr_obs_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
	# 			mkpts_map, num_inliers = self.ref_map_node.get_matched_kpts()
	# 			mkpts_obs, _ = self.curr_obs_node.get_matched_kpts()
	# 			if mkpts_map is not None and mkpts_obs is not None:
	# 				step_size = max(1, len(mkpts_map) // n_viz)
	# 				rgb_img_ref_bgr = cv2.cvtColor(rgb_img_ref, cv2.COLOR_RGB2BGR)
	# 				rgb_img_obs_bgr = cv2.cvtColor(rgb_img_obs, cv2.COLOR_RGB2BGR)
	# 				merged_img = np.hstack((rgb_img_ref_bgr, rgb_img_obs_bgr))
	# 				for i in range(0, len(mkpts_map), step_size):
	# 					x0, y0 = mkpts_map[i]
	# 					x1, y1 = mkpts_obs[i]
	# 					cv2.circle(rgb_img_ref_bgr, (int(x0), int(y0)), 3, (0, 255, 0), -1)
	# 					cv2.circle(rgb_img_obs_bgr, (int(x1), int(y1)), 3, (0, 255, 0), -1)
	# 					cv2.line(merged_img, (int(x0), int(y0)), (int(x1) + rgb_img_ref.shape[1], int(y1)), (0, 255, 0), 2)	
	# 				text = f'Matched inliers kpts: {num_inliers}'
	# 				text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
	# 				text_x = (merged_img.shape[1] - text_size[0])
	# 				text_y = (merged_img.shape[0] - text_size[1])
	# 				cv2.putText(merged_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
	# 				img_msg = pytool_ros.ros_msg.convert_cvimg_to_rosimg(merged_img, "bgr8", header, compressed=False)
	# 				self.pub_map_obs.publish(img_msg)

def perform_map_merging(merger: MergePipeline, args):
	"""Main loop for processing observations"""
	pass
	# obs_poses_gt = np.loadtxt(os.path.join(args.dataset_path, '../out_general', 'poses.txt'))
	# obs_cam_intrinsics = np.loadtxt(os.path.join(args.dataset_path, '../out_general', 'intrinsics.txt'))
	# resize = args.image_size
	# loc.last_obs_node = None

	# for obs_id in range(0, len(obs_poses_gt), 20):
	# 	if rospy.is_shutdown(): break
	# 	print(f"Loading observation with id {obs_id}")

	# 	rgb_img_path = os.path.join(args.dataset_path, '../out_general/seq', f'{obs_id:06d}.color.jpg')
	# 	rgb_img = load_rgb_image(rgb_img_path, resize, normalized=False)

	# 	depth_img_path = os.path.join(args.dataset_path, '../out_general/seq', f'{obs_id:06d}.depth.png')
	# 	depth_img = load_depth_image(depth_img_path, depth_scale=0.001)

	# 	raw_K = np.array([obs_cam_intrinsics[obs_id, 0], 0, obs_cam_intrinsics[obs_id, 2], 0, 
	# 					  obs_cam_intrinsics[obs_id, 1], obs_cam_intrinsics[obs_id, 3], 
	# 					  0, 0, 1], dtype=np.float32).reshape(3, 3)
	# 	raw_img_size = (int(obs_cam_intrinsics[obs_id, 4]), int(obs_cam_intrinsics[obs_id, 5])) # width, height
	# 	K = pytool_sensor.utils.correct_intrinsic_scale(raw_K, resize[0] / raw_img_size[0], resize[1] / raw_img_size[1]) if resize is not None else raw_K
	# 	img_size = (int(resize[0]), int(resize[1])) if resize is not None else raw_img_size
		
	# 	with torch.no_grad():
	# 		desc = loc.vpr_model(rgb_img.unsqueeze(0).to(args.device)).cpu().numpy()

	# 	# Create observation node
	# 	obs_node = ImageNode(obs_id, rgb_img, depth_img, desc,
	# 						 rospy.Time.now().to_sec(),
	# 						 np.zeros(3), np.array([0, 0, 0, 1]),
	# 						 K, img_size,
	# 						 rgb_img_path, depth_img_path)
	# 	obs_node.set_raw_intrinsics(raw_K, raw_img_size)
	# 	obs_node.set_pose_gt(obs_poses_gt[obs_id, 1:4], obs_poses_gt[obs_id, 4:])
	# 	loc.curr_obs_node = obs_node

	# 	"""Perform global localization via. visual place recognition"""
	# 	if not loc.has_global_pos:
	# 		loc_start_time = time.time()
	# 		result = loc.perform_global_loc(save=(args.num_preds_to_save!=0))
	# 		rospy.loginfo(f"Global localization costs: {time.time() - loc_start_time:.3f}s")
	# 		if result['succ']:
	# 			matched_map_id = result['map_id']
	# 			loc.has_global_pos = True
	# 			loc.ref_map_node = loc.image_graph.get_node(matched_map_id)
	# 			loc.curr_obs_node.set_pose(loc.ref_map_node.trans, loc.ref_map_node.quat)
	# 			rospy.logwarn(f'Found VPR Node in global position: {matched_map_id}')
	# 		else:
	# 			rospy.logwarn('[Fail] to determine the global position since no VPR results.')
	# 			continue
	# 	else:
	# 		if loc.last_obs_node is not None:
	# 			init_trans, init_quat = loc.last_obs_node.trans, loc.last_obs_node.quat
	# 			loc.curr_obs_node.set_pose(init_trans, init_quat)

	# 			dis_trans, _ = pytool_math.tools_eigen.compute_relative_dis(init_trans, init_quat, loc.ref_map_node.trans, loc.ref_map_node.quat)
	# 			if dis_trans > loc.args.global_pos_threshold:
	# 				rospy.logwarn('Too far distance from the ref_map_node. Losing Visual Tracking. Reset the global position.')
	# 				loc.has_global_pos = False
	# 				loc.ref_map_node = None
	# 		else:
	# 			rospy.logwarn('[Fail] to determine the global position since not correct VPR.')
	# 			continue				

	# 	"""Perform local localization via. image matching"""
	# 	if loc.has_global_pos:
	# 		loc_start_time = time.time()
	# 		result = loc.perform_local_loc()
	# 		rospy.loginfo(f"Local localization costs: {time.time() - loc_start_time:.3f}s")
	# 		if result['succ']:
	# 			T_w_obs = result['T_w_obs']
	# 			trans, quat = pytool_math.tools_eigen.convert_matrix_to_vec(T_w_obs, 'xyzw')
	# 			loc.curr_obs_node.set_pose(trans, quat)
	# 			loc.has_local_pos = True
	# 			rospy.loginfo(f'Groundtruth Poses: {loc.curr_obs_node.trans_gt.T}')
	# 			rospy.loginfo(f'Estimated Poses: {trans.T}\n')
	# 		else:
	# 			loc.has_local_pos = False
	# 			rospy.logwarn('[Fail] to determine the local position.')

	# 	loc.publish_message()
	# 	# Set as the initial guess of the next observation
	# 	loc.last_obs_node = loc.curr_obs_node
	# 	time.sleep(0.01)
	# 	input()

if __name__ == '__main__':
	args = parse_arguments()
	str_suffix = '_'.join([f'{i}' for i in range(args.num_submap)])
	out_dir = pathlib.Path(os.path.join(args.dataset_path, 'output_map_' + str_suffix))
	out_dir.mkdir(exist_ok=True, parents=True)
	log_dir = setup_log_environment(out_dir, args)

	# Initialize the map merging pipeline
	merger = MergePipeline(args, log_dir)
	# NOTE(gogojjh): no need for VPR
	# rospy.loginfo('Initialize VPR Method')
	# merger.init_vpr_model()
	rospy.loginfo('Initialize Pose Estimator')
	merger.init_pose_estimator()
	merger.read_map_from_file()

	merger.process_submap()

	# rospy.init_node('loc_pipeline_node', anonymous=True)
	# loc_pipeline.initalize_ros()
	# loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	# loc_pipeline.child_frame_id = rospy.get_param('~child_frame_id', 'camera')

	perform_map_merging(merger, args)
