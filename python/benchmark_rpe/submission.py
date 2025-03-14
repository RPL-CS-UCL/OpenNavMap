# [Computation]:
# python submission.py --config ../config/dataset/matterport3d.yaml --split test --out_dir xx --models master --debug
# [Evaluation] (in mickey folder):
# python -m benchmark.mapfree --submission_path /Titan/dataset/data_mapfree/results/master_essentialmatrixmetricmean/submission.zip --dataset_path /Titan/dataset/data_mapfree --split val --log error

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile

import time
import numpy as np
from tqdm import tqdm
from colorama import Fore, Back, Style

from transforms3d.quaternions import mat2quat

from estimator import available_models, get_estimator

from pycpptools.src.python.utils_sensor.utils import correct_intrinsic_scale

from rpe_default import cfg
from datamodules import DataModule

@dataclass
class Pose:
	top_k: int
	list_img0_name: list
	img1_name: str
	q: np.ndarray
	t: np.ndarray
	loss: float

	def __str__(self) -> str:
		formatter = {"float": lambda v: f"{v:.6f}"}
		max_line_width = 1000
		q_str = np.array2string(
			self.q, formatter=formatter, max_line_width=max_line_width
		)[1:-1]
		t_str = np.array2string(
			self.t, formatter=formatter, max_line_width=max_line_width
		)[1:-1]
		str_img0_names = " ".join(img0_name for img0_name in self.list_img0_name)
		return f"{self.top_k} {str_img0_names} {self.img1_name} {q_str} {t_str} {self.loss:.3f}"

def predict(loader, estimator, str_estimator, cfg):
	results_dict = defaultdict(list)
	results_debug_dict = defaultdict(list)
	running_time = []
	save_indice = 0
	for data in tqdm(loader):
		try:
			scene_root = Path(data['scene_root'][0])

			list_img0_name = [name[0] for name in data['list_image0_path']]
			list_img0_poses = [pose.squeeze(0) for pose in data['list_image0_pose']]
			list_img0_intr = [{'K': K.squeeze(0), 'im_size': im_size.squeeze(0)} \
								for K, im_size in zip(data['list_K_color0'], data['list_im_size0'])]
			
			img1_name = data['image1_path'][0]
			img1_intr = {'K': data['K_color1'].squeeze(0), 'im_size': data['im_size1'].squeeze(0)} # K, WxH

			print(Fore.GREEN + f'Scene Root: {scene_root}' + Style.RESET_ALL)
			print(Fore.GREEN + f'Loading Reference Image:', ', '.join(list_img0_name) + Style.RESET_ALL)
			print(Fore.GREEN + f'Loading Target Image: {img1_name}' + Style.RESET_ALL)

			"""Absolute Pose Estimation"""
			# TODO(gogojjh): Images and intrinsics are resized inside the estimator
			# TODO(gogojjh): Joint optimization of intrinsics is better
			est_opts = {
				'known_extrinsics': True,
				'known_intrinsics': False,
				'resize': 512,
			}

			start_time = time.time()
			est_result = estimator(
				scene_root,
				list_img0_name, img1_name, 
				list_img0_poses, 
				list_img0_intr, img1_intr,
				est_opts
			)
			est_time = time.time() - start_time                       
			running_time.append(est_time)

			"""Definition of solver output"""
			# Rwc (numpy.ndarray): Estimated rotation matrix from world (reference frame) to camera
			# twc (numpy.ndarray): Estimated translation vector. Shape: [3, 1] that translate depth_img1 to depth_img0.
			im_pose, loss = est_result["im_pose"], est_result["loss"]
			if im_pose is None: 
				raise ValueError(f"{str_estimator} - Estimated pose is None.")
			elif np.isnan(im_pose).any():
				raise ValueError("Estimated pose is NaN or infinite.")
			
			"""Save Results"""
			scene_id = data['scene_id'][0]
			Twc = np.eye(4); Twc[:3, :3] = im_pose[:3, :3]; Twc[:3, 3] = im_pose[:3, 3]
			Tcw = np.linalg.inv(Twc); Rcw = Tcw[:3, :3]; tcw = Tcw[:3,  3].reshape(3, 1)

			# populate results_dict
			top_k = len(list_img0_name)
			estimated_pose = Pose(top_k=top_k,
								list_img0_name=list_img0_name, 
								img1_name=img1_name,
								q=mat2quat(Rcw).reshape(-1),
								t=tcw.reshape(-1),
								loss=loss)
			results_dict[scene_id].append(estimated_pose)

			print(Fore.GREEN + f'Estimated Pose: {tcw.T}' + Style.RESET_ALL)
			if args.viz: estimator.show_reconstruction(cam_size=cfg.DATASET.VIZ_CAM_SIZE)
			if args.debug:
				out_est_dir = Path(os.path.join(args.out_dir, f"{str_estimator}"))
				out_est_dir.mkdir(parents=True, exist_ok=True)
				Path(out_est_dir / "preds").mkdir(parents=True, exist_ok=True)
		
				list_depth_img_name = \
					[name.replace('.jpg', '.zed.png') for name in list_img0_name] + \
					[img1_name.replace('.jpg', '.zed.png')]
				save_log = Path(os.path.join(out_est_dir, 'preds', scene_id))
				save_log.mkdir(exist_ok=True, parents=True)
				avg_depth_error, corr_score = estimator.save_results(save_log, scene_root, list_depth_img_name, save_indice)
				results_debug_dict[scene_id].append([avg_depth_error, corr_score])	
				save_indice += 1
		
		except Exception as e:
			scene = data['scene_id'][0]
			img1_name = data['image1_path'][0]
			tqdm.write(Fore.RED + f"Error with {str_estimator}: {e}" + Style.RESET_ALL)
			tqdm.write(Fore.RED + f"May occur due to no overlapping regions or insufficient matching at {scene}/{img1_name}." + Style.RESET_ALL)

	avg_runtime = running_time[0] if len(running_time) == 1 else np.mean(running_time)
	return results_dict, results_debug_dict, avg_runtime

def save_submission(results_dict: dict, output_path: Path):
	with ZipFile(output_path, "w") as zip:
		for scene, poses in results_dict.items():
			poses_str = "#N img0_name1 img0_name2 ... img0_nameN img1_name qw qx qy qz tx ty tz loss\n"
			poses_str += "\n".join((str(pose) for pose in poses))
			zip.writestr(f"pose_{scene}.txt", poses_str.encode("utf-8"))

def eval(args):
	# Load configs
	cfg.merge_from_file(args.config)

	# Create dataloader for different datasets
	if args.split == 'test':
		cfg.TRAINING.BATCH_SIZE = 1
		cfg.TRAINING.NUM_WORKERS = 1
		cfg.DATASET.TOP_K = args.top_k
		cfg.DATASET.N_QUERY = args.n_query
		dataloader = DataModule(cfg).test_dataloader()
	elif args.split == 'val':
		cfg.TRAINING.BATCH_SIZE = 1
		cfg.TRAINING.NUM_WORKERS = 1
		cfg.DATASET.TOP_K = args.top_k
		cfg.DATASET.N_QUERY = args.n_query        
		dataloader = DataModule(cfg).val_dataloader()
	elif args.split == 'train':
		cfg.TRAINING.BATCH_SIZE = 1
		cfg.TRAINING.NUM_WORKERS = 1
		cfg.DATASET.TOP_K = args.top_k
		cfg.DATASET.N_QUERY = args.n_query        
		dataloader = DataModule(cfg).train_dataloader()
	else:
		raise NotImplemented(f'Invalid split: {args.split}')

	output_root = Path(args.out_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	for model in args.models:
		estimator = get_estimator(model, 
									device=args.device, 
									out_dir=os.path.join(args.out_dir, f'{model}/preds'),
									lora_path=args.lora_path)
		results_dict, results_debug_dict, avg_runtime = predict(dataloader, estimator, model, cfg)

		if args.debug:
			for scene, values in results_debug_dict.items():
				np.savetxt(os.path.join(args.out_dir, f'{model}/preds', f'debug_{scene}.txt'), np.array(values), fmt='%.5f %.5f')

		print(Fore.GREEN + f"Running APE Method: {model}" + Style.RESET_ALL)

		log_dir = Path(output_root / f"{model}")
		log_dir.mkdir(parents=True, exist_ok=True)

		# Save runtimes to txt
		runtime_str = f"{model}: {avg_runtime:.3f}s"
		with open(log_dir / "runtime_results.txt", "w") as f:
			f.write(runtime_str + "\n")
		tqdm.write(runtime_str)

		# Save predictions to txt per scene within zip
		save_submission(results_dict, log_dir / f"submission_{args.top_k}.zip")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", help="path to config file")
	parser.add_argument(
		"--models",
		type=str,
		nargs="+",
		default="all",
		help=f"Available models: {str(available_models)}"
	)
	parser.add_argument(
		"--device", type=str, default="cuda", choices=["cpu", "cuda"]
	)
	parser.add_argument(
		"--viz",
		action="store_true",
		help="pass --viz to avoid saving visualizations",
	)
	parser.add_argument(
		"--debug",
		action="store_true",
		help="pass --debug to visualize intermediate results",
	)    
	parser.add_argument(
		"--out_dir", type=str, default=None, help="path where outputs are saved"
	)
	parser.add_argument(
		"--num_iters",
		type=int,
		default=1,
		help="number of interations to run benchmark and average over",
	)    
	parser.add_argument(
		"--split",
		choices=("train", "val", "test"),
		default="test",
		help="Dataset split to use for evaluation. Choose from test or val. Default: test",
	)
	parser.add_argument(
		"--lora_path",
		default="lora.pt",
		help="Path to the finetuned LoRA weight",
	)	
	parser.add_argument(
		'--top_k', 
		type=int, 
		default=2, 
		help='Number of randomly selected reference images for localization'
	)
	parser.add_argument(
		'--n_query', 
		type=int, 
		default=1, 
		help='Number of query images for localization'
	)
	args = parser.parse_args()
	if args.models == "all":
		args.models = available_models
	eval(args)
