#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import time
import copy
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from colorama import Fore, Back, Style

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import parser
from dataloader import TestDataset
from utils.utils_vpr_method import *
from utils.utils_image_matching_method import initialize_img_matcher

def extract_descriptors(model, test_ds, args):
	"""Extract and return all descriptors from the test dataset."""
	with torch.inference_mode():
		# Extract database descriptors
		logging.debug(
			"Extracting database descriptors for evaluation/testing"
		)
		database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
		database_dataloader = DataLoader(
			dataset=database_subset_ds,
			num_workers=args.num_workers,
			batch_size=args.batch_size,
		)
		all_descriptors = np.empty(
			(len(test_ds), args.descriptors_dimension), dtype="float32"
		)
		for images, indices, image_name in tqdm(database_dataloader):
			descriptors = model(images.to(args.device))
			descriptors = descriptors.cpu().numpy()
			all_descriptors[indices.numpy(), :] = descriptors

		# Extract query descriptors
		logging.debug(
			"Extracting queries descriptors for evaluation/testing using batch size 1"
		)
		queries_subset_ds = Subset(
			test_ds,
			list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries)),
		)
		queries_dataloader = DataLoader(
			dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1
		)
		for images, indices, image_name in tqdm(queries_dataloader):
			descriptors = model(images.to(args.device))
			descriptors = descriptors.cpu().numpy()
			all_descriptors[indices.numpy(), :] = descriptors

	queries_descriptors = all_descriptors[test_ds.num_database :]
	database_descriptors = all_descriptors[: test_ds.num_database]
	return queries_descriptors, database_descriptors

def predict(test_ds, vpr_model, match_model, image_matcher_model, args):
	queries_descriptors, database_descriptors = extract_descriptors(vpr_model, test_ds, args)
	queries_image_names = test_ds.queries_image_names
	database_image_names = test_ds.database_image_names
	assert len(queries_descriptors) == len(queries_image_names)
	print("Shape of database_descriptors: ", database_descriptors.shape)
	print("Shape of queries_descriptors: ", queries_descriptors.shape)

	match_model.initialize_model(database_descriptors, recall_values=3)

	running_time = []

	# Initial Sequence Matching
	start_time = time.time()	
	init_results_dict, init_db_query_indices = defaultdict(list), []
	for query_indice in range(len(queries_descriptors)):
		query_image_name = queries_image_names[query_indice]
		query_descs = queries_descriptors[max(0, query_indice-match_model.seqLen+1) : query_indice+1]
		recall_preds, pred, score = match_model.match(query_descs)
		init_results_dict[query_image_name] = (recall_preds, pred, score)
		init_db_query_indices.append((pred, query_indice, score))

	# RANSAC-based fitting for outlier rejection
	if match_model.ENABLE_RANSAC:
		D_all = match_model._compute_diff_matrix(queries_descriptors)
		best_db_query_indices, lines_coeff, cluster_data, cluster_labels = \
			match_model.ransac_check_match(D_all, init_db_query_indices[match_model.seqLen:])
		best_db_query_indices = init_db_query_indices[:match_model.seqLen] + best_db_query_indices

		# Add reliable results after filtering and set high score
		best_results_dict = defaultdict(list)
		for db_query_indice in best_db_query_indices:
			# pred0, pred1, ..., predN-1, score of pred0
			query_image_name = queries_image_names[db_query_indice[1]]
			best_results_dict[query_image_name] =  [database_image_names[i] for i in init_results_dict[query_image_name][0]]
			best_results_dict[query_image_name] += [1.0]
			# print(f"Fitting score: {db_query_indice[2]:.3f}")
		
		# Add unreliable results after filtering and set low score
		for k, v in init_results_dict.items():
			if not (k in best_results_dict):
				best_results_dict[k] =  [database_image_names[i] for i in v[0]]
				best_results_dict[k] += [0.0]

		avg_vpr_time = (time.time() - start_time) / len(queries_descriptors)
		
		match_model.save_diff_matrix_fitting(\
			f"{args.out_dir}/{args.method}_{args.match_model}/preds", 
			init_db_query_indices, best_db_query_indices, 
			D_all, None, None, 
			lines_coeff, cluster_data, cluster_labels)
	else:
		best_results_dict = init_results_dict
		avg_vpr_time = (time.time() - start_time) / len(queries_descriptors)

	# Geometric Verification
	if image_matcher_model is not None:
		avg_gv_time = 0.0
		for db_query_indice in best_db_query_indices:
			query_image_name = queries_image_names[db_query_indice[1]]
			if best_reslts_dict[query_image_name] >= 1e-3:
				db_img_path = test_ds.database_image_paths[db_query_indice[0]]
				img0 =  image_matcher_model.load_image(db_img_path, resize=512)
				query_img_path = test_ds.queries_image_paths[db_query_indice[1]]
				img1 =  image_matcher_model.load_image(query_img_path, resize=512)
				
				start_time = time.time()
				result = image_matcher_model(img0, img1)
				num_inliers, H, mkpts0, mkpts1 = result['num_inliers'], result['H'], result['inlier_kpts0'], result['inlier_kpts1']
				print(f"num_inliers: {num_inliers}")
				avg_gv_time += (time.time() - start_time) / len(queries_descriptors)
				if num_inliers < 50: best_results_dict[query_image_name][2] = 0.0
	else:
		avg_gv_time = 0.0

	avg_runtime = avg_vpr_time + avg_gv_time

	return best_results_dict, avg_runtime

def save_submission(results_dict: dict, output_path: Path):
	results = np.empty((0, 3), dtype=object)
	for query_image_name, database_image_names in results_dict.items():
		vec = np.empty((1, 3), dtype=object)
		vec[0, 0], vec[0, 1], vec[0, 2] = \
			query_image_name, database_image_names[0], database_image_names[-1]
		results = np.vstack((results, vec))
		
	np.savetxt(output_path, results, fmt='%s %s %f')

def save_predictions(results_dict: dict, test_ds, log_dir):
	"""Save visualizations of predictions."""    
	for query_idx, (query_image_name, database_image_names) in enumerate(results_dict.items()):
		query_path = [os.path.join(test_ds.queries_folder, query_image_name)]
		database_paths = [os.path.join(test_ds.database_folder, name) 
						  for name in database_image_names[:-2]]
		image_paths = query_path + database_paths
		save_visualization(log_dir, query_idx, image_paths, [1] * len(image_paths))

def eval(args):
	##### Dataloader
	test_ds = TestDataset(
		database_folder=args.database_folder,
		queries_folder=args.queries_folder,
		image_size=args.image_size
	)

	##### Prediction
	output_root = Path(args.out_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	with open(output_root / "runtime_results.txt", "w") as f:
		vpr_model = initialize_vpr_model(args.method, args.backbone, args.descriptors_dimension, args.device)
		match_model = initialize_match_model(args.match_model)
		if args.image_match_model == "none":
			image_matcher_model = None
		else:
			image_matcher_model = initialize_img_matcher(args.image_match_model, args.device, max_num_keypoints=2048)
		results_dict, avg_runtime = predict(test_ds, vpr_model, match_model, image_matcher_model, args)
		print(Fore.GREEN + f"Running VPR Method {args.method} with Match Method {args.match_model}" + Style.RESET_ALL)

		# Save runtimes to txt
		print(args.method, args.match_model, avg_runtime)
		runtime_str = f"{args.method}_{args.match_model}: {avg_runtime:.3f}s"
		f.write(runtime_str + "\n")
		tqdm.write(runtime_str)

		# Save predictions to txt per scene
		log_dir = Path(output_root / f"{args.method}_{args.match_model}")
		log_dir.mkdir(parents=True, exist_ok=True)
		query_name = args.queries_folder.split('out_map_')[-1]
		database_name = args.database_folder.split('out_map_')[-1]
		save_submission(results_dict, log_dir / f"submission-{query_name}-{database_name}.txt")
		if args.debug:
			Path(log_dir / f"preds").mkdir(parents=True, exist_ok=True)
			save_predictions(results_dict, test_ds, log_dir)

def main():
	args = parser.parse_arguments()
	eval(args)

if __name__ == "__main__":
	main()
