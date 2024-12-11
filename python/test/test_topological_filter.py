#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.topological_filter import PlaceRecognitionTopologicalFilter
import numpy as np
import torch
import argparse

from image_graph import ImageGraphLoader as GraphLoader
from image_graph import ImageGraph
from utils.utils_map_merging import *

def test_pr_topological_filter(args):
    submaps = []
    for i in range(args.num_submap):
        submap_id = len(submaps)
        submap_path = os.path.join(args.dataset_path, f'out_map{submap_id}')
        image_graph = GraphLoader.load_data(
            submap_path,
            [512, 288],
            depth_scale=0.0,
            load_rgb=True,
            load_depth=False,
            normalized=False
        )
        submaps.append((submap_id, image_graph))
        print(f"Loaded {image_graph} from {submap_path}")
    print(f"Loaded {len(submaps)} submaps.")

    # Initialize the Bayesian filter
    db_descriptors = np.array([node.get_descriptor() for _, node in submaps[0][1].nodes.items()], dtype="float32")
    topo_filter = PlaceRecognitionTopologicalFilter(db_descriptors)
    
    results = []
    for node in submaps[2][1].nodes.values():
        query_desc = node.get_descriptor()
        if topo_filter.belief is None:
            recall_preds, pred, score = topo_filter.initialize_model(query_desc)
        else:
            recall_preds, pred, score = topo_filter.match(query_desc)
        results.append(recall_preds)
        print(f"Node id {node.id}: Map node with highest posterior = {recall_preds[0]}, Probability = {score}")
    
    results = np.array(results)
    save_vis_coarse_loc('/Rocket_ssd/dataset/tmp', submaps[0][1], submaps[2][1], 2, results)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to the map file")
    parser.add_argument("--num_submap", type=int, help="Number of submaps in the map file")
    args = parser.parse_args()
    results = test_pr_topological_filter(args)
