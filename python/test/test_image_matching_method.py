'''
Usage: python test_image_matching_method.py \
	--matcher duster \
	--input /Titan/code/robohike_ws/src/topo_loc/python/test/logs/anymal_ops_mos/2024-07-08_13-32-18/match_pairs.txt \
	--im_width 288 \
	--im_height 512
'''

import torch
import argparse
import matplotlib
from pathlib import Path

from matching.utils import get_image_pairs_paths, to_numpy  # Import utility for getting image pairs
from matching import available_models  # Import necessary modules from matching package

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import utils.utils_image_matching_method as uimm

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
		matplotlib.use("Agg")

def setup_args():
		"""Setup command-line arguments."""
		parser = argparse.ArgumentParser(
				description="Image Matching Models",
				formatter_class=argparse.ArgumentDefaultsHelpFormatter,
		)
		parser.add_argument(
				"--matcher",
				type=str,
				default="sift-lg",
				choices=available_models,
				help="choose your matcher",
		)
		parser.add_argument("--image_size", type=int, default=512, nargs="+",
												help="Resizing shape for images (HxW). If a single int is passed, set the"
												"smallest edge of all images to this value, while keeping aspect ratio")
		parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
		parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
		parser.add_argument(
				"--no_viz",
				action="store_true",
				help="pass --no_viz to avoid saving visualizations",
		)
		parser.add_argument(
				"--input",
				type=str,
				default="assets/example_pairs",
				help="path to either (1) dir with dirs with pairs or (2) txt file with two img paths per line",
		)
		parser.add_argument(
				"--out_dir", type=str, default=None, help="path where outputs are saved"
		)

		if args.image_size and len(args.image_size) > 2:
				raise ValueError(f"The --image_size parameter can only take up to 2 values, but has received {len(args.image_size)}.")

		return parser.parse_args()

def main(args):
		"""Main function to run the image matching process."""
		image_size = [args.im_width, args.im_height]
		args.out_dir.mkdir(exist_ok=True, parents=True)

		matcher = uimm.initialize_matcher(args.matcher, args.device, args.n_kpts)
		pairs_of_paths = get_image_pairs_paths(args.input)
		for i, (img0_path, img1_path) in enumerate(pairs_of_paths):
				image0 = matcher.load_image(img0_path, resize=image_size)
				image1 = matcher.load_image(img1_path, resize=image_size)

				import time
				start_time = time.time()
				result = uimm.matching_image_pair(matcher, image0, image1)
				print('Matching costs time: {:3f}s'.format(time.time() - start_time))

				num_inliers, mkpts0, mkpts1 = result["num_inliers"], result["inliers0"], result["inliers1"]
				print('Found {} matched keypoints'.format(num_inliers))
				out_str = f"Paths: {str(img0_path), str(img1_path)}. Found {num_inliers} inliers after RANSAC. "
				if not args.no_viz:
						viz_path = uimm.save_visualization(image0, image1, mkpts0, mkpts1, args.out_dir, i, n_viz=100)
						out_str += f"Viz saved in {viz_path}. "
				
				dict_path = uimm.save_output(result, img0_path, img1_path, 
																 		 args.matcher, args.n_kpts, 
																		 image_size, args.out_dir, i)
				out_str += f"Output saved in {dict_path}"       
				print(out_str)

				if args.matcher == 'duster':
					scene = result["duster_scene"]
					im_poses = to_numpy(scene.get_im_poses())
					est_R, est_t = im_poses[1][:3, :3], im_poses[1][:3, 3]
					print('Estimated R:\n', est_R)
					print('Estimated t:\n', est_t.T)
					scene.show()

if __name__ == "__main__":
		args = setup_args()
		if args.out_dir is None:
				args.out_dir = Path(f"outputs_{args.matcher}")
		args.out_dir = Path(args.out_dir)
		main(args)
