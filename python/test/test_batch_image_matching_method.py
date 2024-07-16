'''
Usage: python test_batch_image_matching_method.py --matcher duster \
--dataset_path /Titan/dataset/data_topo_loc/anymal_ops_mos \
--image_size 288 512 --device cuda --sample_map 1 --sample_obs 1000
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import time
import argparse
import matplotlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as pl
pl.ion()

from pycpptools.python.utils_math.tools_eigen import compute_relative_dis, compute_relative_dis_TF, convert_vec_to_matrix
from matching.utils import to_numpy

from utils.utils_image_matching_method import *
from image_graph import ImageGraphLoader

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
	matplotlib.use("Agg")

def main(args):
	"""Main function to run the image matching process."""
	image_size = args.image_size
	log_dir = setup_log_environment(os.path.join(args.dataset_path, 'output_batch_image_matching'), args)

	"""Initialize image matcher"""
	matcher = initialize_matcher(args.matcher, args.device, args.n_kpts)

	"""Load image data"""
	path_map = os.path.join(args.dataset_path, 'map')
	image_graph = ImageGraphLoader.load_data(path_map, image_size, args.depth_scale, normalized=False, num_sample=args.sample_map)
	path_obs = os.path.join(args.dataset_path, 'obs')
	image_obs = ImageGraphLoader.load_data(path_obs, image_size, args.depth_scale, normalized=False, num_sample=args.sample_obs)

	rot_e, trans_e = [], []

	"""Perform image matcher"""
	start_time = time.time()
	for obs_id, obs_node in image_obs.nodes.items():

		# Find the closest map node to the observation node.
		all_map_id, all_dis_trans, all_dis_angle = [], [], []
		for map_id, map_node in image_graph.nodes.items():
			dis_trans, dis_angle = compute_relative_dis(map_node.t_w_cam, map_node.quat_w_cam, obs_node.t_w_cam, obs_node.quat_w_cam)
			if dis_angle > 90.0: continue
			all_map_id.append(map_id)
			all_dis_trans.append(dis_trans)
			all_dis_angle.append(dis_angle)

		map_id = all_map_id[all_dis_trans.index(min(all_dis_trans))]
		map_node = image_graph.get_node(map_id)

		# Matching image pairs
		try:
			out_str = f"Paths: map_id ({map_id}), obs_id ({obs_id}). "
			result = matcher(map_node.rgb_image, obs_node.rgb_image)
			num_inliers, H, mkpts0, mkpts1 = (
					result["num_inliers"],
					result["H"],
					result["inliers0"],
					result["inliers1"],
			)
			assert num_inliers > 100
			
			"""Save matching results"""
			out_str += f"Found {num_inliers} inliers after RANSAC. "
			viz_path = save_visualization(map_node.rgb_image, obs_node.rgb_image, mkpts0, mkpts1, log_dir, obs_id, n_viz=100)
			out_str += f"Viz saved in {viz_path}. "
			dict_path = save_output(result, None, None, args.matcher, args.n_kpts, image_size, log_dir, obs_id)
			out_str += f"Output saved in {dict_path}"       
			print(out_str)
		except Exception as e:
			print(f"Error in Matching: {e}, May occur due to no overlapping regions or insufficient matching.")
			print(out_str)
			continue

		"""Visualize matching results"""
		if args.matcher == 'duster':
			# Groundtruth poses
			T_w_map = convert_vec_to_matrix(map_node.t_w_cam, map_node.quat_w_cam, 'xyzw')
			T_w_obs = convert_vec_to_matrix(obs_node.t_w_cam, obs_node.quat_w_cam, 'xyzw')
			T_map_obs = np.linalg.inv(T_w_map) @ T_w_obs
			print('Groundtruth Poses:\n', T_map_obs)
			# print('Estimated H:\n', H)
			
			scene = matcher.scene
			
			# DEBUG(gogojjh): analyze the generate rgb and depth image
			rgb_image = to_numpy(map_node.rgb_image)
			depth_image = to_numpy(map_node.depth_image)
			save_input_images(rgb_image * 255, depth_image / args.depth_scale,
												os.path.join(log_dir, f'map_gt_rgb_{map_id}.png'), 
 											  os.path.join(log_dir, f'map_gt_depth_{map_id}.png'))
			rgb_image = scene.imgs[0]
			depth_image = to_numpy(scene.get_depthmaps())[0]
			save_duster_images(rgb_image * 255, depth_image / args.depth_scale, 
												 os.path.join(log_dir, f'map_duster_rgb_{map_id}.png'), 
												 os.path.join(log_dir, f'map_duster_depth_{map_id}.png'))

			rgb_image = to_numpy(obs_node.rgb_image)
			depth_image = to_numpy(obs_node.depth_image)
			save_input_images(rgb_image * 255, depth_image / args.depth_scale,
												os.path.join(log_dir, f'obs_gt_rgb_{obs_id}.png'), 
 											  os.path.join(log_dir, f'obs_gt_depth_{obs_id}.png'))
			rgb_image = scene.imgs[1]
			depth_image = to_numpy(scene.get_depthmaps())[1]
			save_duster_images(rgb_image * 255, depth_image / args.depth_scale, 
												 os.path.join(log_dir, f'obs_duster_rgb_{obs_id}.png'), 
												 os.path.join(log_dir, f'obs_duster_depth_{obs_id}.png'))
			######################################

			im_poses = scene.get_im_poses()
			im_poses = to_numpy(scene.get_im_poses())
			est_T = im_poses[1]
			# Change the definition of est_T since it is originally defined as T_obs_map
			if abs(np.sum(np.diag(est_T)) - 4.0) < 1e-5:
				est_T = np.linalg.inv(im_poses[0])
			print('Estimated Poses:\n', est_T)

			# Normalized poses
			est_T_normalized = np.copy(est_T)
			if abs(est_T[2, 3]) < 1e-9:
				scale = 1.0
			else:
				scale = T_map_obs[2, 3] / est_T[2, 3]
			est_T_normalized[:3, 3] *= scale
			print(f'Normalized Poses with scale {scale}:\n', est_T_normalized)

			# Compute error
			dis_trans, dis_angle = compute_relative_dis_TF(est_T_normalized, T_map_obs)
			rot_e.append(dis_angle)
			trans_e.append(dis_trans)

			if not args.no_viz:			
				scene.show(cam_size=0.05)

	print(f'Matching costs {(time.time() - start_time) / image_obs.get_num_node()}s')
	
	# Save rotation and translation error
	save_error(np.array(rot_e), np.array(trans_e), log_dir)

if __name__ == "__main__":
		args = parse_arguments()
		main(args)
