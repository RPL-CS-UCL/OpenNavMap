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
	data_pairs = []
	scene_data_cnt = dict()
	est_opts = {
		'known_extrinsics': True,
		'known_intrinsics': False,
		'resize': 512,
	}
	for data in tqdm(loader):
		try:
			scene_root = Path(data['scene_root'][0])
			# Generate a fixed number of data for each scene
			if scene_root.name in scene_data_cnt and scene_data_cnt[scene_root.name] * 2 >= args.n_query:
				continue

			list_img0_name = [name[0] for name in data['list_image0_path']]
			list_img0_poses = [pose.squeeze(0) for pose in data['list_image0_pose']]
			list_img0_intr = [{'K': K.squeeze(0), 'im_size': im_size.squeeze(0)} \
							  for K, im_size in zip(data['list_K_color0'], data['list_im_size0'])]

			img1_name = data['image1_path'][0]
			img1_intr = {'K': data['K_color1'].squeeze(0), 'im_size': data['im_size1'].squeeze(0)} # K, WxH
		
			# Run depth estimation
			print(f"Running test {list_img0_name} {img1_name}")
			estimator(
				scene_root,
				list_img0_name, img1_name,
				list_img0_poses, list_img0_intr,
				img1_intr,
				est_opts
			)
			# estimator.show_reconstruction()
		except Exception as e:
			print(f"Error processing: {e}")
			continue			

		# Retrieve and process depth map
		depth_maps = estimator.scene.get_depthmaps()
		weight_i = estimator.scene.weight_i
		weight_j = estimator.scene.weight_j

		# Store depth_map0 and depth_map1 for simplicity
		list_img_name = list_img0_name[:2]
		list_depth_name = [name.replace('.jpg', '.pdepth.png') for name in list_img_name]
		list_intr = list_img0_intr[:2]

		depths = [(d.detach().cpu().numpy() * 1000.0).astype(np.uint16) for d in depth_maps[:2]]
		weights = [weight_i['0_1'].detach().cpu().numpy(), weight_j['0_1'].detach().cpu().numpy()]
		masks = [w < estimator.calib_params['pseudo_gt_thre'] for w in weights[:2]]

		# Only add new paris with reliable match
		if all(np.sum(m) < d.size * 0.35 for m, d in zip(masks, depths)):
			# Avoid duplicate update on the depth map
			key1 = f"{scene_root.name}/{list_img0_name[0]}"
			key2 = f"{scene_root.name}/{list_img0_name[1]}"
			if key1 not in existing_keys and key2 not in existing_keys:
				existing_keys.add(key1)
				existing_keys.add(key2)
			else:
				continue			

			for idx in range(len(list_img_name)):
				# Filter out unreliable depth
				depth = depths[idx]; depth[masks[idx]] = 0
				# Resize the depth image to the original size
				new_size = tuple(list_intr[idx]['im_size'].cpu().numpy().astype(int)) # WxH
				re_depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)
				output_path = Path(args.out_dir) / scene_root.name / list_depth_name[idx]
				output_path.parent.mkdir(parents=True, exist_ok=True)
				cv2.imwrite(str(output_path), re_depth)
				print(f'Saving pdepth to {str(output_path)}')

			data_pairs.append((scene_root.name, list_img0_name[0], list_img0_name[1], list_depth_name[0], list_depth_name[1]))
			if scene_root.name in scene_data_cnt:
				scene_data_cnt[scene_root.name] += 1
			else:
				scene_data_cnt[scene_root.name] = 1

	dtype = [
		('scene_name', 'U20'),   # Unicode string up to 20 chars
		('img0', 'U50'),         # Image path field
		('img1', 'U50'),
		('depth0', 'U50'),
		('depth1', 'U50')
	]
	print(f"Total pairs are generated: {len(data_pairs)}")
	np.save(os.path.join(args.out_dir, f"mapfree_pairs_{len(data_pairs)*2}pdepth.npy"), np.array(data_pairs, dtype=dtype))

	data_pairs_gtdepth = []
	for pair in data_pairs:
		new_pair = (
			pair[0], pair[1], pair[2],
			pair[3].replace('.pdepth.png', '.zed.png'),
			pair[4].replace('.pdepth.png', '.zed.png')
		)
		data_pairs_gtdepth.append(new_pair)
		np.save(os.path.join(args.out_dir, f"mapfree_pairs_{len(data_pairs)*2}gtdepth.npy"), np.array(data_pairs_gtdepth, dtype=dtype))

def main(args):
	"""Main pipeline setup and execution."""
	# Load configuration file
	cfg.merge_from_file(args.config)

	# Configure and initialize data loader
	cfg.TRAINING.BATCH_SIZE = 1
	cfg.TRAINING.NUM_WORKERS = 1
	cfg.DATASET.TOP_K = args.top_k
	cfg.DATASET.N_QUERY = args.n_query * 5
	dataloader = DataModule(cfg).train_dataloader()

	# Initialize depth estimation model
	estimator = get_estimator(args.model, device=args.device)
	estimator.verbose = False
	assert (estimator.calib_params is not None), "Should use duster_calib_pretrain or master_calib_pretrain"

	# Process all images
	process_data(dataloader, estimator, args)

if __name__ == "__main__":
	# Configure command-line arguments
	parser = argparse.ArgumentParser(description="Pseudo Depth Map Generator")
	parser.add_argument("--config", required=True, help="Path to config.yaml")
	parser.add_argument("--model", required=True, help="Model selection")
	parser.add_argument("--out_dir", required=True, help="Output directory")
	parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
	parser.add_argument("--top_k", type=int, default=2, help="Number of reference images")
	parser.add_argument("--n_query", type=int, default=1, help="Number of query images")
	
	args = parser.parse_args()
	main(args)