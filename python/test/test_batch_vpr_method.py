'''
Usage: python test_batch_vpr_method.py \
	--method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
	--num_preds_to_save=3 \
	--image_size 200 200
	--device=cuda \
	--sample_map=3 --sample_obs=1000
	--dataset_path=/Titan/dataset/data_topo_loc/anymal_ops_mos
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../VPR-methods-evaluation'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../VPR-methods-evaluation/third_party/deep-image-retrieval'))

import time
import argparse
import matplotlib
from pathlib import Path
import numpy as np
import faiss
import torch

from utils.utils_vpr_method import *
from image_graph import ImageGraphLoader, ImageGraph

import visualizations

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
	matplotlib.use("Agg")

def setup_args():
	"""Setup command-line arguments."""
	args = parse_arguments()
	return args

def extract_descriptors(model, image_list, descriptors_dimension, device):
	with torch.inference_mode():
		logging.info("Extracting descriptors for evaluation/testing")
		all_descriptors = np.empty((image_list.get_num_node(), descriptors_dimension), dtype="float32")
		for indices, (id, node) in enumerate(image_list.nodes.items()):
			descriptor = model(node.image.unsqueeze(0).to(device))
			node.set_descriptor(descriptor)
			all_descriptors[indices, :] = descriptor.cpu().numpy()
		return all_descriptors

def main(args):
	"""Main function to run the image matching process."""
	out_dir = Path(os.path.join(args.dataset_path, 'output_batch_vpr_method'))
	out_dir.mkdir(exist_ok=True, parents=True)
	log_dir = setup_log_environment(out_dir, args)
	image_size = args.image_size

	"""Initialize VPR model"""
	model = initialize_vpr_model(args.method, args.backbone, args.descriptors_dimension, args.device)

	"""Load images"""
	image_graph = ImageGraphLoader.load_data(os.path.join(args.dataset_path, 'map'), 
																					 image_size=image_size, 
																					 normalized=True, 
																					 num_sample=args.sample_map)
	image_obs = ImageGraphLoader.load_data(os.path.join(args.dataset_path, 'obs'), 
																				 image_size=image_size, 
																				 normalized=True, 
																				 num_sample=args.sample_obs)

	"""Extract image descriptors"""
	start_time = time.time()
	database_descriptors = extract_descriptors(model, image_graph, args.descriptors_dimension, args.device)
	all_map_id = image_graph.get_all_id()
	queries_descriptors = extract_descriptors(model, image_obs, args.descriptors_dimension, args.device)
	all_obs_id = image_obs.get_all_id()
	print('Extracting desc costs: {:3f}s'.format((time.time() - start_time) / (image_graph.get_num_node() + image_obs.get_num_node())))

	"""Perform KNN search"""
	start_time = time.time()
	predictions = perform_knn_search(database_descriptors, queries_descriptors, 
																	 args.descriptors_dimension, args.recall_values)
	print('Matching each desc costs: {:3f}s'.format((time.time() - start_time) / len(predictions)))

	"""Save image descriptors"""
	if args.save_descriptors:
		save_descriptors(log_dir, queries_descriptors, database_descriptors)	

	"""Save visualizations of predictions."""
	if args.num_preds_to_save != 0:
			logging.info("Saving final predictions")
			for i in range(len(predictions)):
				obs_id = all_obs_id[i]
				obs_node = image_obs.get_node(obs_id)
				if obs_node is not None:
					list_of_images_paths = [obs_node.img_path]

					for j in range(len(predictions[i][:args.num_preds_to_save])):
						if predictions[i][j] < 0: continue
						map_node = image_graph.get_node(all_map_id[predictions[i][j]])
						if map_node is not None:
							list_of_images_paths.append(map_node.img_path)

					preds_correct = [None] * len(list_of_images_paths)
					save_visualization(log_dir, obs_id, list_of_images_paths, preds_correct)

if __name__ == "__main__":
		args = setup_args()
		main(args)
