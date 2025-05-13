#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import numpy as np
import argparse
import torch
import itertools
from pathlib import Path
import open3d as o3d
from PIL import Image
import pyiqa
from joblib import Parallel, delayed
from torchvision import transforms

from utils.utils_geom import read_intrinsics, read_poses, read_timestamps, read_descriptors, convert_vec_to_matrix
from utils.utils_vpr_method import *
from utils.utils_viz2d_camera import plot_camera_poses_pair

from metric.full_kf_selector import FullKFSelector
from metric.landmark_selector import LandmarkSelector
from metric.pose_density_selector import PoseDensitySelector
from metric.feature_selector import FeatureSelector

from estimator import THIRD_PARTY_DIR, get_estimator, add_to_path
add_to_path(THIRD_PARTY_DIR.joinpath("mast3r/dust3r"))
from dust3r.utils.geometry import inv, geotrf

from matching import available_models, get_matcher

DB_Ratio = 0.75
Time_Threshold = 60.0

class SubmapManager:
	def __init__(self, time_threshold):
		self.submaps = []
		self.current_submap = None
		self.time_threshold = time_threshold
		
	def add_frame(self, img_name, timestamp):
		if not self.submaps:
			self._create_new_submap(img_name, timestamp)
			return
			
		last_time = self.current_submap['end_time']
		if timestamp - last_time >= self.time_threshold:
			self._finalize_current_submap()
			self._create_new_submap(img_name, timestamp)
		else:
			self.current_submap['end_time'] = timestamp
			self.current_submap['frames'].append(img_name)
			
	def _create_new_submap(self, img_name, timestamp):
		self.current_submap = {
			'start_time': timestamp,
			'end_time': timestamp,
			'frames': [img_name]
		}
		self.submaps.append(self.current_submap)
		
	def _finalize_current_submap(self):
		if self.current_submap:
			self.current_submap['duration'] = \
				self.current_submap['end_time'] - self.current_submap['start_time']

def save_point_cloud(pts3d, save_path, save_flag=False):
	"""
	Save a point cloud to a file using Open3D.
	
	Args:
		pts3d (np.ndarray): Point cloud of shape [H, W, 3].
		save_path (str): Path to save the point cloud.
	"""
	pts3d_flat = pts3d.reshape(-1, 3)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pts3d_flat)
	if save_flag:
		o3d.io.write_point_cloud(save_path, pcd)
	return pcd

def visualize_proj_depth(output_dir, depthmap, proj_depthmap, i, j):
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(1, 2, figsize=(16, 12))
	im0 = axs[0].imshow(depthmap, cmap='turbo')
	axs[0].set_title(f'Original Depth Camera {j} onto Camera {j}')
	plt.colorbar(im0, ax=axs[0], label='Depth')
	
	im1 = axs[1].imshow(proj_depthmap, cmap='turbo')
	axs[1].set_title(f'Projected Depth of Camera {i} onto Camera {j})')
	plt.colorbar(im1, ax=axs[1], label='Depth')

	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f'depth_maps_{i}_to_{j}.jpg'))
	plt.close()

def pre_compute(scene_path, str_matcher, str_estimator):
	"""Enhanced pre-computation with structured submap handling"""
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	resize = (512, 288)
	
	##### Step 1: Load raw data
	original_scene_path = scene_path.replace('keyframe_selection_eval', 'map_free_eval')
	intrinsics = read_intrinsics(os.path.join(original_scene_path, 'intrinsics.txt'))
	poses = read_poses(os.path.join(original_scene_path, 'poses.txt'))
	timestamps = read_timestamps(os.path.join(original_scene_path, 'timestamps.txt'))

	# Generate Fake timestamp
	if timestamps is None:
		timestamps = dict()
		num_img = len(poses)
		num_database = int(num_img * DB_Ratio)
		for id, key in enumerate(poses.keys()):
			timestamps[key] = np.array([int(id / (num_img / 10)) * Time_Threshold])
			
	# Validate data consistency
	for key in poses.keys(): 
		if key not in timestamps:
			raise KeyError(f"{key} not found in timestamps")

	##### Step 2:  Create submap split
	submap_mgr = SubmapManager(Time_Threshold)
	for img_name, timestamp in timestamps.items():
		submap_mgr.add_frame(img_name, timestamp[0])
	submap_mgr._finalize_current_submap()

	# Convert to structured numpy array
	submap_data = []
	for submap in submap_mgr.submaps:
		submap_data.append((
			submap['start_time'],
			submap['end_time'],
			np.array(submap['frames'])
		))
	
	dtype = np.dtype([
		('start_time', 'f8'), 
		('end_time', 'f8'),
		('frames', 'O')
	])
	np.save(os.path.join(scene_path, 'submap_split.npy'), np.array(submap_data, dtype=dtype))
	print(f"{len(submap_data)} submaps are split")

	##### Step 3:  Create iqa.txt
	# print('Processing IQA')
	# IQA_METRIC = 'musiq'
	# iqa_metric = pyiqa.create_metric(IQA_METRIC, device=device)
	# iqa_scores = np.empty((len(poses), 2), dtype=object)
	# for indice, (img_name, _) in enumerate(poses.items()):
	# 	img_path = os.path.join(original_scene_path, img_name)
	# 	score = iqa_metric(img_path).detach().squeeze(0).cpu().numpy()[0]
	# 	iqa_scores[indice, 0], iqa_scores[indice, 1] = img_name, score
	# np.savetxt(os.path.join(scene_path, 'iqa.txt'), iqa_scores, fmt="%s %.4f")

	#### Step 4:  Create descriptor.txt
	# print('Processing Desc')
	# desc_dimenson = 256
	# vpr_model = initialize_vpr_model('cosplace', 'ResNet18', desc_dimenson, device)    
	# transformations = [
	# 	transforms.ToTensor(),
	# 	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	# 	transforms.Resize(size=resize, antialias=True)
	# ]
	# transform = transforms.Compose(transformations)
	# all_descriptors = np.empty((len(poses), desc_dimenson + 1), dtype=object)
	# for indice, (img_name, _) in enumerate(poses.items()):
	# 	img_path = os.path.join(original_scene_path, img_name)
	# 	pil_img = Image.open(img_path).convert("RGB")
	# 	normalized_img = transform(pil_img)
	# 	descriptors = vpr_model(normalized_img.unsqueeze(0).to(device))
	# 	descriptors = descriptors.detach().cpu().numpy()
	# 	all_descriptors[indice, 0], all_descriptors[indice, 1:] = img_name, descriptors
	# np.savetxt(os.path.join(scene_path, 'descriptors.txt'), all_descriptors, fmt="%s" + " %.9f" * desc_dimenson)

	##### Step 5:  Create information reduction and information gain
	print('Processing Overlap')
	if 'mapfree' in original_scene_path:
		lm_redu, lm_gain = {}, {}
		# To support pre-computed overlap of mapfree dataset
		overlap_path = os.path.join(original_scene_path, 'overlaps.npz')
		if os.path.exists(overlap_path):
			overlap_score = np.load(overlap_path)
			idxs, overlaps = overlap_score['idxs'], overlap_score['overlaps']
			for img_name_0, img_name_1 in itertools.combinations(poses.keys(), 2):
				if 'seq0' in img_name_0 and 'seq1' in img_name_1:
					id0 = int(img_name_0.split('/')[1].split['.'](0))
					id1 = int(img_name_1.split('/')[1].split['.'](0))
					filter_idx = (idxs == np.array((0, id0, 1, id1))).all(axis=1)
					if filter_idx is None:
						overlap = 0
					else:
						overlap = overlaps[filter_idx]

				lm_redu[(img_name_0, img_name_1)] = overlap          # how much information is redundant of img_0
				lm_gain[(img_name_0, img_name_1)] = 1.0 - overlap    # how much information is gained of img_0 
				lm_redu[(img_name_1, img_name_0)] = overlap          # how much information is redundant of img_1
				lm_gain[(img_name_1, img_name_0)] = 1.0 - overlap    # how much information is gained of img_1                
		else:
			T_cams = {k: convert_vec_to_matrix(p[4:], p[:4], 'wxyz') for k, p in poses.items()}
			inv_T_cams = {k: np.linalg.inv(T) for k, T in T_cams.items()}
			for n0 in poses.keys():
				fx, fy, cx, cy, W, H = intrinsics[n0][:6]
				W, H = int(W), int(H)
				# Convert depth image to point cloud
				depth_img_0 = np.array(Image.open(os.path.join(original_scene_path, n0.replace('.jpg', '.mickey.png'))), 
									   dtype=np.float32) / 1000.0
				u, v = np.meshgrid(np.arange(W), np.arange(H))
				u, v, z = u.flatten(), v.flatten(), depth_img_0.flatten()
				valid = z > 0
				x = (u[valid] - cx) * z[valid] / fx
				y = (v[valid] - cy) * z[valid] / fy
				pts0 = np.column_stack((x, y, z[valid]))  # Shape: (N, 3)

				# Compute overlap ratio in parallel
				n1_candidates = [n1 for n1 in poses.keys() if n1 != n0]
				def compute_ratio(n0, n1, pts0, inv_T_cam_n0, T_cam_n1, fx, fy, cx, cy, W, H):
					T_rel = T_cam_n1 @ inv_T_cam_n0
					R, t = T_rel[:3, :3], T_rel[:3, 3]
					proj = (R @ pts0.T).T + t
					u = np.round(fx * proj[:, 0] / proj[:, 2] + cx)
					v = np.round(fy * proj[:, 1] / proj[:, 2] + cy)
					valid = (proj[:, 2] > 0) & (0 <= u) & (u < W) & (0 <= v) & (v < H)
					return valid.sum() / len(pts0)
				
				ratios = Parallel(n_jobs=-1)(
					delayed(compute_ratio)(
						n0, n1, pts0, inv_T_cams[n0], T_cams[n1], fx, fy, cx, cy, W, H
					) for n1 in n1_candidates
				)
				for n1, ratio in zip(n1_candidates, ratios):
					lm_redu[(n0, n1)] = ratio
					lm_gain[(n0, n1)] = 1.0 - ratio
					# print(f"{n0} -> {n1}: {ratio:.3f}")
	else:
		estimator = get_estimator(str_estimator, device)
		estimator.verbose = True
		output_dir = os.path.join(scene_path, 'preds')
		os.makedirs(output_dir, exist_ok=True)

		# Compute the number of overlapping landmarks
		lm_redu, lm_gain = {}, {}
		for n0, n1 in itertools.combinations(poses.keys(), 2):
			print(f"{n0} - {n1}")
			img_path_0 = os.path.join(original_scene_path, n0)
			img_path_1 = os.path.join(original_scene_path, n1)
			estimator(Path(scene_path), [img_path_0], img_path_1, None, None, None, dict())

			ratio_A2B = dict()
			K = estimator.scene.get_intrinsics()
			cams = inv(estimator.scene.get_im_poses())
			depthmaps = estimator.scene.get_depthmaps()
			all_pts3d = estimator.scene.get_pts3d() # all pts3d in the world frame
			# msk_conf = estimator.scene.get_masks()
			H, W = depthmaps[0].shape

			assert len(all_pts3d) == 2
			for i in range(len(all_pts3d)):          
				j = 1 - i
				print(f"{i} -> {j}")

				# Project depth of camera i into camera j
				pts3d_flat = all_pts3d[i].reshape(-1, 3)
				proj = geotrf(cams[j], pts3d_flat)
				proj_depth = proj[:, 2]
				u, v = geotrf(K[j], proj, norm=1, ncol=2).round().long().unbind(-1)

				# Mask for overlapping points
				valid_mask = (proj_depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
				proj_depth_map = torch.zeros(H, W, device=device)
				proj_depth_map[v[valid_mask], u[valid_mask]] = proj_depth[valid_mask]

				u, v = u[valid_mask], v[valid_mask]
				proj_depth = proj_depth[valid_mask]
				msk = torch.abs(proj_depth - depthmaps[j][v, u].reshape(1, -1)) < 0.5 * depthmaps[j][v, u].reshape(1, -1)

				# Overlap score
				ratio_A2B[(i, j)] = np.sum(msk.detach().cpu().numpy()) / (len(pts3d_flat))

				viz_flag = False
				if viz_flag:
					visualize_proj_depth(
						output_dir, 
						depthmaps[j].detach().cpu().numpy(), 
						proj_depth_map.detach().cpu().numpy(), 
						i, j
					)
					save_point_cloud(all_pts3d[i].detach().cpu().numpy(), os.path.join(output_dir, f'pts3d_{i}.pcd'), True)

			lm_redu[(n0, n1)] = ratio_A2B[(0, 1)]          # how much information is redundant of img_0
			lm_gain[(n0, n1)] = 1.0 - ratio_A2B[(0, 1)]    # how much information is gained of img_0 
			lm_redu[(n1, n0)] = ratio_A2B[(1, 0)]          # how much information is redundant of img_1
			lm_gain[(n1, n0)] = 1.0 - ratio_A2B[(1, 0)]    # how much information is gained of img_1

			print(f'Info Redu: larger, the more of A is observed by B')
			print(f'(A to B) Info Redu: {ratio_A2B[(0, 1)]:.3f}, Info Gain: {1.0-ratio_A2B[(0, 1)]:.3f}')        
			print(f'(B to A) Info Redu: {ratio_A2B[(1, 0)]:.3f}, Info Gain: {1.0-ratio_A2B[(1, 0)]:.3f}')

	np.save(os.path.join(scene_path, 'landmark_redundancy.npy'), lm_redu)
	np.save(os.path.join(scene_path, 'landmark_gain.npy'), lm_gain)

	##### Step 5:  Create image matcher and relax the threshold for considering more mkpts 
	# matcher = get_matcher(str_matcher, device=device)
	# imgs = dict()
	# for key in poses.keys():
	# 	imgs[key] = matcher.load_image(os.path.join(original_scene_path, key))
	# DEBUG(gogojjh):
	# print('Processing Matching')
	# kpts_match = {}
	# key_list = list(poses.keys())
	# for id, n0 in enumerate(key_list):
	# 	if id == len(key_list) - 1: continue
	# 	for n1 in key_list[id+1:]:
	# 		result = matcher(imgs[n0], imgs[n1])
	# 		num_matches = len(result['inlier_kpts0'])
	# 		kpts_match[(n0, n1)] = num_matches
	# 		kpts_match[(n1, n0)] = num_matches
	# 		# print(n0, n1, num_matches)

	# np.save(os.path.join(scene_path, f'keypoint_matched_{str_matcher}.npy'), kpts_match)    

	# Copy timestamps
	import shutil
	timestamp_file = os.path.join(original_scene_path, 'timestamps.txt')
	destination_file = os.path.join(scene_path, 'timestamps.txt')
	if os.path.exists(timestamp_file):
		shutil.copy2(timestamp_file, destination_file)
		print(f"Copied {timestamp_file} to {destination_file}")    
	else:
		timestamps_np = np.empty((len(timestamps), 2), dtype=object)
		for id, (key, timestamp) in enumerate(timestamps.items()):
			timestamps_np[id][0], timestamps_np[id][1] = key, timestamp
		np.savetxt(destination_file, timestamps_np, fmt='%s %6f')

	# Copy poses
	import shutil
	pose_file = os.path.join(original_scene_path, 'poses.txt')
	destination_file = os.path.join(scene_path, 'poses.txt')
	shutil.copy2(pose_file, destination_file)
	print(f"Copied {pose_file} to {destination_file}")
	
def load_scene_data(scene_path, str_matcher, str_estimator):
	"""Improved data loading with submap structure conversion"""
	required_files = ['iqa.txt', 'landmark_redundancy.npy', 'landmark_gain.npy', 
					  'submap_split.npy', 'timestamps.txt', 'descriptors.txt', 'poses.txt']
	Path(scene_path).mkdir(parents=True, exist_ok=True)

	if not all(os.path.exists(os.path.join(scene_path, f)) for f in required_files):
		print("Pre-computed files not found. Computing...")
		pre_compute(scene_path, str_matcher, str_estimator)
	
	# Load and convert submap structure
	submap_array = np.load(os.path.join(scene_path, 'submap_split.npy'), allow_pickle=True)
	submap_splits = [{
		'start': item['start_time'],
		'end': item['end_time'],
		'frames': item['frames'].tolist()
	} for item in submap_array]
	
	return {
		'timestamps': read_timestamps(os.path.join(scene_path, 'timestamps.txt')),
		'poses': read_poses(os.path.join(scene_path, 'poses.txt')),
		'descriptors': read_descriptors(os.path.join(scene_path, 'descriptors.txt')),
		'iqa_scores': read_timestamps(os.path.join(scene_path, 'iqa.txt')),
		'lm_redu': np.load(os.path.join(scene_path, 'landmark_redundancy.npy'), allow_pickle=True),
		'lm_gain': np.load(os.path.join(scene_path, 'landmark_gain.npy'), allow_pickle=True),
		'submap_splits': submap_splits,
	}

def select_keyframes(scene_path, scene_data, args):
	###### Definition
	###### timestamps[img_name] = timestamp
	###### iqa_scores[img_name] = iqa_score 
	###### lm_redu[img_name0, img_name1] = lm_redu
	###### lm_gain[img_name0, img_name1] = lm_gain
	###### submap_splits[i] = {'start': start_time, 'end': end_time, 'frames': [img_name0, img_name1, ...]}
	timestamps = scene_data['timestamps']
	poses = scene_data['poses']
	descriptors = scene_data['descriptors']
	iqa_scores = scene_data['iqa_scores']       
	lm_redu = scene_data['lm_redu'].item()  
	lm_gain = scene_data['lm_gain'].item()  
	submap_splits = scene_data['submap_splits']

	# NOTE(gogojjh): the criteria to split submap
	# database: [0, DB_Ratio]
	# query:    [DB_Ratio, 1]
	num_query = max(1, int(len(submap_splits) * (1 - DB_Ratio)))
	num_database = len(submap_splits) - num_query
	submap_database = submap_splits[:num_database]
	submap_query = submap_splits[num_database:]
	print(f"Split database and query map number: {num_database} - {num_query}")

	ori_path = scene_path.replace('keyframe_selection_eval', 'map_free_eval')

	all_db_frames = [img_name for submap in submap_database for img_name in submap['frames']]
	all_query_frames = [img_name for submap in submap_query for img_name in submap['frames']]
	db_keyframes = []
	if args.method == 'full_kf':         # not select keyframes
		kf_selector = FullKFSelector()
		db_keyframes = kf_selector.select_keyframes(submap_database)
	elif args.method == 'pose_density':  # pose density
		kf_selector = PoseDensitySelector()
		db_keyframes = kf_selector.select_keyframes(poses, submap_database)
	elif args.method == 'feature':       # 2D feature
		kf_selector = FeatureSelector()
		db_keyframes = kf_selector.select_keyframes(ori_path, descriptors, submap_database)
	elif args.method == 'landmark':      # 3D landmark
		kf_selector = LandmarkSelector()
		db_keyframes = kf_selector.select_keyframes(ori_path, timestamps, descriptors, iqa_scores, lm_redu, lm_gain, submap_database)
		
	return all_db_frames, all_query_frames, db_keyframes

def parse_arguments():
	parser = argparse.ArgumentParser(description='Keyframe Selection Algorithm')
	parser.add_argument('--keyframe_path', type=str, required=True, 
					   help='Path to the dataset')
	parser.add_argument('--scenes', type=str, required=True, nargs='+',
					   help='Scenes name to process')
	parser.add_argument('--matcher', type=str, required=True, default="master", 
					   help=f'{available_models}')
	parser.add_argument('--estimator', type=str, required=True, default="master", 
					   help=f'master, duster')
	parser.add_argument('--method', type=str, default=None, 
					   help='full_kf, pose_density, feature, landmark')
	return parser.parse_args()

def main():
	args = parse_arguments()
	for scene in args.scenes:
		print(f"Processing scene {scene}")

		# Pre-compute data
		scene_path = os.path.join(args.keyframe_path, scene)
		scene_data = load_scene_data(scene_path, args.matcher, args.estimator)
		print(f"Loaded {scene} data with {len(scene_data['submap_splits'])} submaps")

		# Run keyframe selection with different methods
		if args.method is not None:
			(Path(scene_path) / 'preds').mkdir(parents=True, exist_ok=True)
			all_db_frames, all_query_frames, db_keyframes = select_keyframes(scene_path, scene_data, args)
			np.savetxt(os.path.join(scene_path, f"keyframes_{args.method}.txt"), np.array(db_keyframes, dtype=object), fmt='%s')
			print(f"Saved keyframes to {os.path.join(scene_path, f'keyframes_{args.method}.txt')}")

			# Visualize selected keyframes
			viz_kf_flag = True
			if viz_kf_flag:
				poses_kf, poses_rm, poses_query = [], [], []
				for img_name in all_db_frames:
					trans, quat = scene_data['poses'][img_name][4:], np.roll(scene_data['poses'][img_name][:4], -1)
					if img_name in db_keyframes:
						poses_kf.append(np.concatenate((trans, quat)))
					else:
						poses_rm.append(np.concatenate((trans, quat)))               
						
				for img_name in all_query_frames:
					trans, quat = scene_data['poses'][img_name][4:], np.roll(scene_data['poses'][img_name][:4], -1)
					poses_query.append(np.concatenate((trans, quat)))

				poses = np.array(poses_kf + poses_rm + poses_query)
				fig = plot_camera_poses_pair(poses, [0, len(poses_kf), len(poses_kf)+len(poses_rm)], 
											 1, title=f"Selected Keyframes in {scene}")
				fig.savefig(os.path.join(scene_path, f"preds/poses_{args.method}.pdf"))
				plt.close()

if __name__ == "__main__":
	main()
