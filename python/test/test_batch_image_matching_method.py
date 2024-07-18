'''
Usage1: python test_batch_image_matching_method.py --matcher duster \
--dataset_path /Titan/dataset/data_topo_loc/anymal_ops_mos \
--image_size 288 512 --device cuda --sample_map 1 --sample_obs 1000 \
--min_depth_pro 0.1 --max_depth_pro 5.5 --depth_scale 0.001

Usage2: python test_batch_image_matching_method.py --matcher duster \
--dataset_path /Titan/dataset/data_topo_loc/cmu_navigation_matterport3d_17DRP5sb8fy \
--image_size 288 512 --device cuda --sample_map 1 --sample_obs 1000 \
--min_depth_pro 0.1 --max_depth_pro 5.5 --depth_scale 0.039
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

from pycpptools.src.python.utils_math.tools_eigen import compute_relative_dis, compute_relative_dis_TF, convert_vec_to_matrix
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
	matcher = initialize_img_matcher(args.matcher, args.device, args.n_kpts)

	"""Load image data"""
	path_map = os.path.join(args.dataset_path, 'map')
	image_graph = ImageGraphLoader.load_data(path_map, image_size, depth_scale=args.depth_scale, normalized=False, num_sample=args.sample_map)
	path_obs = os.path.join(args.dataset_path, 'obs')
	image_obs = ImageGraphLoader.load_data(path_obs, image_size, depth_scale=args.depth_scale, normalized=False, num_sample=args.sample_obs)

	"""Perform image matcher"""
	start_time = time.time()
	rot_e, trans_e = [], []
	for obs_id, obs_node in image_obs.nodes.items():

		# Find the closest map node to the observation node.
		all_map_id, all_dis_trans, all_dis_angle = [], [], []
		for map_id, map_node in image_graph.nodes.items():
			dis_trans, dis_angle = compute_relative_dis(
				map_node.trans_gt, map_node.quat_gt, 
				obs_node.trans_gt, obs_node.quat_gt
			)
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
			T_w_map = convert_vec_to_matrix(map_node.trans_gt, map_node.quat_gt, 'xyzw')
			T_w_obs = convert_vec_to_matrix(obs_node.trans_gt, obs_node.quat_gt, 'xyzw')
			T_map_obs = np.linalg.inv(T_w_map) @ T_w_obs
			print('Groundtruth Poses:\n', T_map_obs)
			# print('Estimated H:\n', H)
			
			scene = matcher.scene
			
			###################################### Analyze duster's results
			# Retrieve rgb and predicted confidence and depth map from duster
			rgb_image_map = np.transpose(to_numpy(map_node.rgb_image), (1, 2, 0)) # 3xHXW -> HxWx3
			rgb_image_obs = np.transpose(to_numpy(obs_node.rgb_image), (1, 2, 0)) # 3xHXW -> HxWx3
			save_image(np.hstack((rgb_image_map * 255, rgb_image_obs * 255)), 
								 save_path=os.path.join(log_dir, 'preds_depthmap', f'{obs_id}_rgb.png'))

			conf_img_map = to_numpy(scene.im_conf[0])
			conf_img_obs = to_numpy(scene.im_conf[1])
			plot_images(conf_img_map, conf_img_obs, title1=f"Conf (Map-{map_id})", title2=f"Conf (Obs-{obs_id})", 
									save_path=os.path.join(log_dir, 'preds_depthmap', f'{obs_id}_conf.png'))
			
			# Compute the scale by aligning two depth images
			depth_image_ref = np.squeeze(np.transpose(to_numpy(obs_node.depth_image), (1, 2, 0)), axis=2) # 1xHXW -> HxWx1
			depth_image_target = to_numpy(scene.get_depthmaps())[1]
			mask = (depth_image_ref < args.min_depth_pro) | (depth_image_ref > args.max_depth_pro)
			depth_image_ref[mask] = 0.0
			depth_image_target[mask] = 0.0
			meas_scale = compute_scale_factor(depth_image_ref, depth_image_target)
			print(f'Scale Factor: {meas_scale:.3f}')			

			total_dis_before_scaling = np.sum(compute_residual_matrix(depth_image_ref, depth_image_target, 1.0))
			mean_dis_before_scaling = total_dis_before_scaling / np.size(depth_image_ref)
			total_dis_after_scaling = np.sum(compute_residual_matrix(depth_image_ref, depth_image_target, meas_scale))
			mean_dis_after_scaling = total_dis_after_scaling / np.size(depth_image_ref)
			print(f'Total Disp before Scaling: {total_dis_before_scaling:.5f}, ', 
						f'Mean Disp before Scaling: {mean_dis_before_scaling:.5f}')
			print(f'Total Disp after Scaling: {total_dis_after_scaling:.5f}, ', 
						f'Mean Disp after Scaling: {mean_dis_after_scaling:.5f}')
			print(f'Reduce Ratio: {mean_dis_before_scaling / mean_dis_after_scaling:.5f}')

			depth_image_target_scale = meas_scale * depth_image_target
			plot_images(depth_image_ref, depth_image_target, 
									title1=f"Depth-GT (Obs-{obs_id})", title2=f"Depth-Ori (Obs-{obs_id})", 
									save_path=os.path.join(log_dir, 'preds_depthmap', f'{obs_id}_depth.png'))
			plot_images(depth_image_ref, depth_image_target_scale, 
									title1=f"Depth-GT (Obs-{obs_id})", title2=f"Depth-Scaled (Obs-{obs_id})", 
									save_path=os.path.join(log_dir, 'preds_depthmap', f'{obs_id}_depth_scaling.png'))		
			plot_images(depth_image_ref, compute_residual_matrix(depth_image_ref, depth_image_target, 1.0), 
									title1=f"Depth-GT (Obs-{obs_id})", title2="Error Map",
									save_path=os.path.join(log_dir, 'preds_depthmap', f'{obs_id}_error_map.png'))
			plot_images(depth_image_ref, compute_residual_matrix(depth_image_ref, depth_image_target, meas_scale), 
									title1=f"Depth-GT (Obs-{obs_id})", title2="Error Map", 
									save_path=os.path.join(log_dir, 'preds_depthmap', f'{obs_id}_error_map_scaling.png'))		
			######################################

			im_poses = scene.get_im_poses()
			im_poses = to_numpy(scene.get_im_poses())
			est_T = im_poses[1]

			# Change the definition of est_T since it is originally defined as T_obs_map
			if abs(np.sum(np.diag(est_T)) - 4.0) < 1e-5:
				est_T = np.linalg.inv(im_poses[0])

			# Normalized poses with pose scale
			if abs(est_T[2, 3]) < 1e-9:
				pose_scale = 1.0
			else:
				pose_scale = T_map_obs[2, 3] / est_T[2, 3]
			est_T_normalized = np.copy(est_T)
			est_T_normalized[:3, 3] *= pose_scale
			print(f'Normalized Poses with Pose scale {pose_scale}:\n', est_T_normalized)

			# Normalized poses with measurement scale
			est_T_normalized = np.copy(est_T)
			est_T_normalized[:3, 3] *= meas_scale
			print(f'Normalized Poses with Meas scale {meas_scale}:\n', est_T_normalized)
			print('\n')

			# Compute error
			dis_trans, dis_angle = compute_relative_dis_TF(est_T_normalized, T_map_obs)
			rot_e.append(dis_angle)
			trans_e.append(dis_trans)

			if not args.no_viz:
				scene.show(cam_size=0.05)

	print(f'Matching costs {(time.time() - start_time) / image_obs.get_num_node():.3f}s\n')
	
	# Save rotation and translation error
	save_error(np.array(rot_e), np.array(trans_e), log_dir)

if __name__ == "__main__":
		args = parse_arguments()
		main(args)
