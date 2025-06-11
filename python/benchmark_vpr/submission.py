#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import time
import json
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
from utils.utils_vpr_method import initialize_vpr_model, initialize_match_model, save_visualization
from utils.utils_image_matching_method import initialize_img_matcher
from utils.vpr_single_matching import PlaceRecognitionSingleMatching
from utils.vpr_graph_search import PlaceRecognitionGraphSearch

def extract_descriptors(model, test_ds, args):
	global descriptors_dimension
	total_db_desc_time, total_query_desc_time = 0.0, 0.0
	
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
			(len(test_ds), descriptors_dimension), dtype="float32"
		)
		for images, indices, _ in tqdm(database_dataloader):
			start_time = time.time()
			descriptors = model(images.to(args.device))
			total_db_desc_time += time.time() - start_time
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
			dataset=queries_subset_ds, 
			num_workers=args.num_workers, 
			batch_size=args.batch_size
		)
		for images, indices, _ in tqdm(queries_dataloader):
			start_time = time.time()
			descriptors = model(images.to(args.device))
			total_query_desc_time += time.time() - start_time
			descriptors = descriptors.cpu().numpy()
			all_descriptors[indices.numpy(), :] = descriptors

	queries_descriptors = all_descriptors[test_ds.num_database :]
	database_descriptors = all_descriptors[: test_ds.num_database]

	return queries_descriptors, database_descriptors, total_query_desc_time

def predict(test_ds, vpr_model, vpr_match_model, image_matcher_model, setting, args):
	# Extract VPR Descriptors
	queries_descriptors, database_descriptors, total_query_desc_time = extract_descriptors(vpr_model, test_ds, args)

	queries_image_names = test_ds.queries_image_names
	database_image_names = test_ds.database_image_names
	assert len(queries_descriptors) == len(queries_image_names)
	print("Shape of database_descriptors: ", database_descriptors.shape)
	print("Shape of queries_descriptors: ", queries_descriptors.shape)

	##### Initial and Run VPR Match Model
	vpr_match_model.initialize_model(database_descriptors)
	init_results_dict, init_db_query_indices = defaultdict(list), []
	total_vpr_time = 0.0

	##### VPR Matching
	if type(vpr_match_model).__name__ == 'PlaceRecognitionSingleMatching':
		for query_indice in range(len(queries_descriptors)):
			start_time = time.time()	
			query_desc = queries_descriptors[query_indice, :].reshape(1, -1)
			_, pred, score = vpr_match_model.match(query_desc)
			total_vpr_time += time.time() - start_time
		
			init_db_query_indices.append((pred, query_indice))
			query_image_name = queries_image_names[query_indice]
			init_results_dict[query_image_name] = (database_image_names[pred], score, 1)
	elif type(vpr_match_model).__name__ == 'PlaceRecognitionGraphSearch':
		start_time = time.time()
		init_db_query_indices, score = vpr_match_model.match(queries_descriptors)
		total_vpr_time += time.time() - start_time
		
		for pred, query_indice in init_db_query_indices:
			query_image_name = queries_image_names[query_indice]
			init_results_dict[query_image_name] = (database_image_names[pred], score, 1)
	else:
		for query_indice in range(len(queries_descriptors)):
			start_time = time.time()	
			query_descs = queries_descriptors[max(0, query_indice-vpr_match_model.seqLen+1) : query_indice+1]
			_, pred, score = vpr_match_model.match(query_descs)
			total_vpr_time += time.time() - start_time
		
			init_db_query_indices.append((pred, query_indice))
			query_image_name = queries_image_names[query_indice]
			init_results_dict[query_image_name] = (database_image_names[pred], score, 1)

	D_all = vpr_match_model.compute_diff_matrix(queries_descriptors)
	vpr_match_model.viz_diff_matrix(f"{args.out_dir}/{setting}/preds", D_all, init_db_query_indices)
	best_results_dict = init_results_dict

	##### Geometric Verification
	total_gv_time = 0.0
	if image_matcher_model is not None:
		for db_query_indice in init_db_query_indices:
			query_image_name = queries_image_names[db_query_indice[1]]
			if best_results_dict[query_image_name][2] > 0:
				db_img_path = test_ds.database_image_paths[db_query_indice[0]]
				img0 = image_matcher_model.load_image(db_img_path, resize=512)
				query_img_path = test_ds.queries_image_paths[db_query_indice[1]]
				img1 = image_matcher_model.load_image(query_img_path, resize=512)
				
				start_time = time.time()
				try:
					result = image_matcher_model(img0, img1)
				except Exception as e:
					print(f"Error in Image Matching: {e}")
					result = {'num_inliers': 0.0}
				total_gv_time += time.time() - start_time

				same_place = int(result['num_inliers'] > 50)
				best_results_dict[query_image_name] = \
					(best_results_dict[query_image_name][0], result['num_inliers'], same_place)
			else:
				best_results_dict[query_image_name] = \
					(best_results_dict[query_image_name][0], 0, 0)
		
	total_runtime = total_query_desc_time + total_vpr_time + total_gv_time

	return best_results_dict, total_runtime

def save_submission(results_dict: dict, output_path: Path):
	results = np.empty((0, 4), dtype=object)
	for query_image_name, db_image_name_score in results_dict.items():
		vec = np.empty((1, 4), dtype=object)
		vec[0, 0] = query_image_name 
		vec[0, 1] = db_image_name_score[0]
		vec[0, 2] = db_image_name_score[1]
		vec[0, 3] = db_image_name_score[2]
		results = np.vstack((results, vec))
		
	np.savetxt(output_path, results, fmt='%s %s %f %s')

def save_predictions(results_dict: dict, test_ds, log_dir):
	"""Save visualizations of predictions."""    
	for query_idx, (query_image_name, db_image_name_score) in enumerate(results_dict.items()):
		query_path = [os.path.join(test_ds.queries_folder, query_image_name)]
		database_paths = [os.path.join(test_ds.database_folder, db_image_name_score[0])]
		image_paths = query_path + database_paths
		save_visualization(log_dir, query_idx, image_paths, [None] * len(image_paths))

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

	query_name = args.queries_folder.split('out_map_')[-1]
	database_name = args.database_folder.split('out_map_')[-1]
	output_time = dict()

	for str_vpr_match_model in args.vpr_match_models:
		for str_image_match_model in args.image_match_models:
			for str_backbone, str_desc_dimension, str_vpr_model in zip(args.str_backbones, args.str_descriptors_dimensions, args.str_vpr_models):		
				if 'sequence_match' in str_vpr_match_model:
					seq_lens = args.vpr_match_seq_lens
				else:
					seq_lens = [1] # single_match only has one sequence length
				
				for vpr_match_seq_len in seq_lens:
					setting  = f"{str_vpr_model}_{str_backbone}_{str_desc_dimension}_"
					setting += f"{str_vpr_match_model}_{vpr_match_seq_len}_{str_image_match_model}"
					print(f"Evaluating VPR Setting: {setting}")
					
					log_dir = Path(output_root / f"{setting}")
					log_dir.mkdir(parents=True, exist_ok=True)
					Path(log_dir / f"preds").mkdir(parents=True, exist_ok=True)
					
					global descriptors_dimension
					str_vpr_model, backbone, descriptors_dimension = \
						parser.check_vpr_params(str_vpr_model, str_backbone, int(str_desc_dimension), args.image_size)

					vpr_model = initialize_vpr_model(str_vpr_model, backbone, descriptors_dimension, args.device)
					vpr_match_model = initialize_match_model(str_vpr_match_model, vpr_match_seq_len)
					image_matcher_model = initialize_img_matcher(str_image_match_model, args.device, max_num_keypoints=2048)
					results_dict, total_runtime = \
						predict(test_ds, vpr_model, vpr_match_model, image_matcher_model, setting, args)
			
					print(Fore.GREEN + 
						  f"Running {str_vpr_model} [VPR Model] " + 
						  f"{str_vpr_match_model} [VPR Match Model] with {vpr_match_seq_len} [Seq] " +
						  f"{str_image_match_model} [Image Match Model]" + 
						  Style.RESET_ALL)

					# Save runtimes to txt
					output_time[setting] = dict()
					output_time[setting]['Total Runtime [s]'] = total_runtime
					output_time[setting]['Query Number'] = test_ds.num_queries

					# Save predictions to txt per scene
					save_submission(results_dict, log_dir / f"submission-{query_name}-{database_name}.txt")
					if args.debug:
						save_predictions(results_dict, test_ds, log_dir)

	output_json = json.dumps(output_time, indent=2)
	with open(os.path.join(args.out_dir, f"runtime_results-{query_name}-{database_name}.json"), 'w') as f:
		f.write(output_json)

def main():
	args = parser.parse_arguments()
	eval(args)

if __name__ == "__main__":
	import warnings
	warnings.filterwarnings("ignore", category=FutureWarning)
	main()
