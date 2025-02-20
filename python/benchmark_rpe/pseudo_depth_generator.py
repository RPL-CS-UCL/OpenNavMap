import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch

from datamodules import DataModule
from estimator import get_estimator
from rpe_default import cfg  # Import configuration settings

def process_data(loader, estimator, args):
	"""Process data batches to estimate and save depth maps."""
	existing_keys = set()	
	for data in tqdm(loader):
		try:
			scene_root = Path(data['scene_root'][0])

			list_img0_name = [name[0] for name in data['list_image0_path']]
			list_img0_poses = [pose.squeeze(0) for pose in data['list_image0_pose']]
			list_img0_intr = [{'K': K.squeeze(0), 'im_size': im_size.squeeze(0)} \
								for K, im_size in zip(data['list_K_color0'], data['list_im_size0'])]

			img1_name = data['image1_path'][0]
			img1_intr = {'K': data['K_color1'].squeeze(0), 'im_size': data['im_size1'].squeeze(0)} # K, WxH

			# Check whether these images are already generated depth
			k = sum(1 for img_name in list_img0_name if str(scene_root / img_name) in existing_keys)
			if k == len(list_img0_name): 
				continue
			existing_keys.update(str(scene_root / img_name) for img_name in list_img0_name)
			print(list_img0_name, img1_name)

			# Configure estimation options
			est_opts = {
				'known_extrinsics': True,
				'known_intrinsics': False,
				'resize': 512,
			}

			# Run depth estimation
			estimator(
				scene_root,
				list_img0_name,
				img1_name,
				list_img0_poses,
				list_img0_intr,
				img1_intr,
				est_opts
			)
			# estimator.show_reconstruction()

		except Exception as e:
			print(f"Error processing: {e}")

		# Retrieve and process depth map
		depth_maps = estimator.scene.get_depthmaps()
		weight_i = estimator.scene.weight_i
		weight_j = estimator.scene.weight_j

		# Check consistency
		list_img_name = list_img0_name + [img1_name]
		assert len(list_img_name) == len(depth_maps)

		# Create output path and save
		for idx, (depth_map, img_name) in enumerate(zip(depth_maps, list_img_name)):
			if idx == 0:
				if weight_i['0_1'].mean() > weight_j['1_0'].mean():
					weight_map = weight_i['0_1'].detach().cpu().numpy()
				else:
					weight_map = weight_j['1_0'].detach().cpu().numpy()
			else:
				key1, key2 = f"{0}_{idx}", f"{idx}_{0}"
				if weight_i[key2].mean() > weight_j[key1].mean():
					weight_map = weight_i[key2].detach().cpu().numpy()
				else:
					weight_map = weight_j[key1].detach().cpu().numpy()

			depth_map_np = depth_map.detach().cpu().numpy() if torch.is_tensor(depth_map) else depth_map
			depth_map_np = (depth_map_np * 1000).astype(np.uint16)

			# Remove depth values with low confidence
			mask_depth_low_conf = weight_map < estimator.calib_params['pseudo_gt_thre']
			depth_map_np[mask_depth_low_conf] = 0
			
			rel_path = Path(img_name)
			output_dir = Path(args.out_dir) / scene_root.name / rel_path.parent
			output_dir.mkdir(parents=True, exist_ok=True)
			output_path = output_dir / f"{rel_path.stem}.pdepth.png"
			if not os.path.exists(output_path):
				cv2.imwrite(str(output_path), depth_map_np)


def main(args):
	"""Main pipeline setup and execution."""
	# Load configuration file
	cfg.merge_from_file(args.config)

	# Configure and initialize data loader
	cfg.TRAINING.BATCH_SIZE = 1
	cfg.TRAINING.NUM_WORKERS = 1
	cfg.DATASET.TOP_K = args.top_k
	cfg.DATASET.N_QUERY = args.n_query
	dataloader = DataModule(cfg).test_dataloader()

	# Initialize depth estimation model
	estimator = get_estimator(args.model, device=args.device)
	assert (estimator.calib_params is not None), "Should use duster_calib or master_calib"

	# Process all images
	process_data(dataloader, estimator, args)

if __name__ == "__main__":
	# Configure command-line arguments
	parser = argparse.ArgumentParser(description="Pseudo Depth Map Generator")
	parser.add_argument("--config", required=True, help="Path to config.yaml")
	parser.add_argument("--model", choices=['duster_calib', 'master_calib'], required=True, help="Model selection")
	parser.add_argument("--out_dir", required=True, help="Output directory")
	parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
	parser.add_argument("--top_k", type=int, default=2, help="Number of reference images")
	parser.add_argument("--n_query", type=int, default=1, help="Number of query images")
	
	args = parser.parse_args()
	main(args)