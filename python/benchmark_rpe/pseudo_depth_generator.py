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
	data_pairs = dict()     # data_pairs[scene_name] store the list of pairs
	existing_pairs = dict() # existing_pairs[scene_name] store the pair name
	est_opts = {
		'known_extrinsics': True,
		'known_intrinsics': False,
		'resize': 512,
	}
	for data in tqdm(loader):
		try:
			scene_root = Path(data['scene_root'][0])
			if scene_root.name in data_pairs:
				if len(data_pairs[scene_root.name]) >= args.n_query:
					continue
			else:
				data_pairs[scene_root.name] = []
				existing_pairs[scene_root.name] = set()
				existing_pairs[scene_root.name].add(f"{scene_root.name}/seq0/frame_00000.jpg")

			list_img0_name = [name[0] for name in data['list_image0_path']]
			list_img0_poses = [pose.squeeze(0) for pose in data['list_image0_pose']]
			list_img0_intr = [{'K': K.squeeze(0), 'im_size': im_size.squeeze(0)} \
							  for K, im_size in zip(data['list_K_color0'], data['list_im_size0'])]

			# Use seq0/frame_00000.jpg
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
		
		# conf_i['i_j']: the confidence of ith image with i_j pair
		# conf_j['i_j']: the confidence of jth image with i_j pair
		# conf_i, conf_j = estimator.scene.conf_i, estimator.scene.conf_j

		# weight_i['i_j']: the calibrated confidence of ith image with i_j pair
		# weight_j['i_j']: the calibrated confidence of jth image with i_j pair
		# weight map for pair (0, 1) -> weight_i['0_1'], weight_j['0_1']
		weight_i, weight_j = estimator.scene.weight_i, estimator.scene.weight_j

		# Retrieve and process depth map
		depth_maps = estimator.scene.get_depthmaps()

		# The connectivity graph for computation
		msp_edges = estimator.get_minimum_spanning_tree()
		print('MST edges: ', msp_edges)

		# Generate and Store depth_map
		list_img_name = list_img0_name + [img1_name]
		list_depth_name = [name.replace('.jpg', '.pdepth.png') for name in list_img_name]
		list_intr = list_img0_intr + [img1_intr]
		depths = [(d.detach().cpu().numpy() * 1000.0).astype(np.uint16) for d in depth_maps]

		for edge in msp_edges:
			edge_str = estimator.get_edge_str(edge[0], edge[1])
			weights = [weight_i[edge_str].detach().cpu().numpy(), weight_j[edge_str].detach().cpu().numpy()]
			valid_masks = [w >= estimator.calib_params['pseudo_gt_thre'] for w in weights]

			print(f'Edges: {edge_str}')
			SIZE_THRE = 0.3 # reliable match threshold - outdoor setting: 0.3; indoor setting: 0.65
			for m, d in zip(valid_masks, depths):
				print(f"{np.sum(m):.3f}, {d.size}, {d.size * SIZE_THRE:.3f}")

			if all(np.sum(m) >= d.size * SIZE_THRE for m, d in zip(valid_masks, depths)):
				# Avoid duplicate update on the depth map
				key1 = f"{scene_root.name}/{list_img_name[edge[0]]}"
				key2 = f"{scene_root.name}/{list_img_name[edge[1]]}"
				if (key1 not in existing_pairs[scene_root.name]) and (key2 not in existing_pairs[scene_root.name]):
					existing_pairs[scene_root.name].add(key1)
					existing_pairs[scene_root.name].add(key2)
				else:
					continue			

				for idx in range(len(edge)):
					# Filter out unreliable depth
					depth = depths[edge[idx]]; depth[~valid_masks[idx]] = 0
					# Resize the depth image to the original size
					new_size = tuple(list_intr[edge[idx]]['im_size'].cpu().numpy().astype(int)) # WxH
					re_depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)
					output_path = Path(args.out_dir) / scene_root.name / list_depth_name[edge[idx]]
					output_path.parent.mkdir(parents=True, exist_ok=True)
					cv2.imwrite(str(output_path), re_depth)
					print(f'Saving pdepth to {str(output_path)}')

				data_pairs[scene_root.name].append((
					scene_root.name, 
					list_img_name[edge[0]], 
					list_img_name[edge[1]], 
					list_depth_name[edge[0]], 
					list_depth_name[edge[1]])
				)
		
	dtype = [
		('scene_name', 'U20'),   # Unicode string up to 20 chars
		('img0', 'U50'),         # Image path field
		('img1', 'U50'),
		('depth0', 'U50'),
		('depth1', 'U50')
	]

	for scene_name, pairs in data_pairs.items():
		print(f"Total pairs for scene {scene_name} are generated: {len(pairs)}")
		np.save(os.path.join(args.out_dir, f"mapfree_pairs_{scene_name}_{len(pairs)*2}pdepth.npy"), np.array(pairs, dtype=dtype))

		if args.save_gtpair:
			pairs_gtdepth = []
			for pair in pairs:
				new_pair = (
					pair[0], pair[1], pair[2],
					pair[3].replace('.pdepth.png', f'.{cfg.DATASET.ESTIMATED_DEPTH}.png'),
					pair[4].replace('.pdepth.png', f'.{cfg.DATASET.ESTIMATED_DEPTH}.png')
				)
				pairs_gtdepth.append(new_pair)
				np.save(os.path.join(args.out_dir, f"mapfree_pairs_{scene_name}_{len(pairs)*2}gtdepth.npy"), np.array(pairs_gtdepth, dtype=dtype))

def main(args):
	"""Main pipeline setup and execution."""
	# Load configuration file
	cfg.merge_from_file(args.config)

	# Configure and initialize data loader
	cfg.TRAINING.BATCH_SIZE = 1
	cfg.TRAINING.NUM_WORKERS = 1
	cfg.DATASET.TOP_K = args.top_k
	cfg.DATASET.N_QUERY = args.n_query * 10
	dataloader = DataModule(cfg).test_dataloader()

	# Initialize depth estimation model
	estimator = get_estimator(args.model, device=args.device)
	estimator.verbose = False
	estimator.niter = 600
	estimator.set_calib_params(dict(mu=1.0, conf_thre=0.5, pseudo_gt_thre=args.pseudo_gt_thre))
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
	parser.add_argument("--save_gtpair", action="store_true")
	parser.add_argument("--pseudo_gt_thre", type=float, default=1.5, help="Pseudo gt threshold for specific datasets")
	
	args = parser.parse_args()
	main(args)