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
import pathlib
import numpy as np
import torch
import time
import cv2
import rospy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray
import tf2_ros
import matplotlib
import logging

import rospkg
rospkg = rospkg.RosPack()
pack_path = rospkg.get_path('litevloc')
# sys.path.append(os.path.join(pack_path, '../image_matching_models'))
# sys.path.append(os.path.join(pack_path, '../image_matching_models'))

from utils.utils_pose_estimation_method import *
from utils.utils_image import load_rgb_image, load_depth_image
from image_graph import ImageGraphLoader as GraphLoader
from image_node import ImageNode

import pycpptools.src.python.utils_math as pytool_math
import pycpptools.src.python.utils_ros as pytool_ros
import pycpptools.src.python.utils_sensor as pytool_sensor

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
		for submap_id in range(num_submap):
			submap_path = os.path.join(self.args.dataset_path, f'out_map{submap_id}')
			image_graph = GraphLoader.load_data(
				submap_path,
				self.args.image_size,
				depth_scale=0.0,
				load_rgb=False,
				load_depth=False,
				normalized=False
			)
			# Extract VPR descriptors for all nodes in the map
			db_descriptors = np.array([map_node.get_descriptor() for _, map_node in image_graph.nodes.items()], dtype="float32")
			print(f"Extracted {db_descriptors.shape} VPR descriptors from the {submap_id} submap.")
			db_poses = np.empty((image_graph.get_num_node(), 7), dtype="float32")
			for indices, (_, map_node) in enumerate(image_graph.nodes.items()):
				db_poses[indices, :3] = map_node.trans
				db_poses[indices, 3:] = map_node.quat
			submap_dict = {'id': submap_id, 'graph': image_graph, 'db_descriptors': db_descriptors, 'db_poses': db_poses}
			logging.info(f"Loaded {image_graph} from {submap_path}")

			self.submaps.append(submap_dict)
		print(f"Loaded {len(self.submaps)} submaps.")

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
	out_dir = pathlib.Path(os.path.join(args.dataset_path, 'output_map_merging'))
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

	# rospy.init_node('loc_pipeline_node', anonymous=True)
	# loc_pipeline.initalize_ros()
	# loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	# loc_pipeline.child_frame_id = rospy.get_param('~child_frame_id', 'camera')

	perform_map_merging(merger, args)
