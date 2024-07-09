'''
Usage: python test_batch_image_matching_method.py \
	--matcher duster \
	--input /Titan/code/robohike_ws/src/topo_loc/python/test/logs/anymal_ops_mos/2024-07-08_13-32-18/match_pairs.txt \
	--im_width 288 \
	--im_height 512
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import time
import argparse
import matplotlib
from pathlib import Path

from pycpptools.python.utils_math.tools_eigen import compute_relative_dis

from matching.utils import get_image_pairs_paths, to_numpy
from matching import available_models

from utils.utils_image_matching_method import initialize_matcher, matching_image_pair, save_visualization, save_output
from image_graph import ImageGraphLoader, ImageGraph
from image_obs import ImageObsLoader, ImageObs

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
	matplotlib.use("Agg")

def setup_args():
	"""Setup command-line arguments."""
	parser = argparse.ArgumentParser(description="Batch Image Matching Test",
																		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--dataset_path", type=str, default="matterport3d", help="path to dataset_path")
	parser.add_argument("--matcher", type=str, default="sift-lg", choices=available_models, help="choose your matcher")
	parser.add_argument("--im_width", type=int, default=288, help="resize img to im_width")
	parser.add_argument("--im_height", type=int, default=512, help="resize img to im_height")
	parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
	parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
	parser.add_argument("--no_viz", action="store_true", help="pass --no_viz to avoid saving visualizations")
	parser.add_argument("--sample_obs", type=int, default=1, help="sample of observation")
	return parser.parse_args()

def main(args):
	"""Main function to run the image matching process."""
	image_size = [args.im_height, args.im_width]
	out_dir = Path(os.path.join(args.dataset_path, 'output_batch_image_matching'))
	out_dir.mkdir(exist_ok=True, parents=True)

	"""Initialize image matcher"""
	matcher = initialize_matcher(args.matcher, args.device, args.n_kpts)

	start_time = time.time()
	image_graph = ImageGraphLoader.load_data(os.path.join(args.dataset_path, 'map'), image_size)
	print(f'Load {len(image_graph.nodes)} image nodes of the graph costs {(time.time() - start_time):3f}s')

	start_time = time.time()
	image_obs = ImageObsLoader.load_data(os.path.join(args.dataset_path, 'obs'), image_size, args.sample_obs)
	print(f'Load {len(image_obs.nodes)} image obs costs {(time.time() - start_time):3f}s')	

	for obs_id, obs_node in image_obs.nodes.items():
		all_dis_trans, all_dis_angle, all_map_id = [], [], []
		for map_id, map_node in image_graph.nodes.items():
			dis_trans, dis_angle = compute_relative_dis(map_node.t_w_cam, map_node.quat_w_cam, obs_node.t_w_cam, obs_node.quat_w_cam)
			all_dis_trans.append(dis_trans)
			all_dis_angle.append(dis_angle)
			all_map_id.append(map_id)

		if all_dis_trans.index(min(all_dis_trans)) == all_dis_angle.index(min(all_dis_angle)):
			map_id = all_map_id[all_dis_trans.index(min(all_dis_trans))]
			map_node = image_graph.get_node(map_id)

			start_time = time.time()
			result = matching_image_pair(matcher, map_node.image, obs_node.image)
			print('Matching costs time: {:3f}s'.format(time.time() - start_time))

			num_inliers, mkpts0, mkpts1 = result["num_inliers"], result["inliers0"], result["inliers1"]
			print('Found {} matched keypoints'.format(num_inliers))
			
			out_str = f"Paths: map_id ({map_id}), obs_id ({obs_id}). Found {num_inliers} inliers after RANSAC. "
			if not args.no_viz:
					viz_path = save_visualization(map_node.image, obs_node.image, mkpts0, mkpts1, out_dir, obs_id, n_viz=100)
					out_str += f"Viz saved in {viz_path}. "
			
			dict_path = save_output(result, None, None, args.matcher, args.n_kpts, image_size, out_dir, obs_id)
			out_str += f"Output saved in {dict_path}"       
			print(out_str)

			if args.matcher == 'duster':
				scene = result["duster_scene"]
				im_poses = scene.get_im_poses()
				im_poses = to_numpy(scene.get_im_poses())
				est_R, est_t = im_poses[1][:3, :3], im_poses[1][:3, 3]
				print('Estimated R:\n', est_R)
				print('Estimated t:\n', est_t.T)
				scene.show()

if __name__ == "__main__":
		args = setup_args()
		main(args)
