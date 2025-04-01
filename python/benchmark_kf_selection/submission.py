#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transforms3d.quaternions import mat2quat
from matching import available_models, get_matcher
from matching.utils import to_numpy

from utils.utils_geom import read_intrinsics, read_poses, read_descriptors, read_img_names
from image_node import ImageNode
from image_graph import ImageGraph
from utils.utils_vpr_method import perform_knn_search
from keyframe_selection import DB_Raio

def parse_arguments():
    # Define argparse for user inputs
    parser = argparse.ArgumentParser(description="Keyframe Selection and Pose Estimation")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--scene", type=str, required=True, help="Scene identifier")
    parser.add_argument("--keyframe_dir", type=str, required=True, help="Path to the keyframe directory")
    args = parser.parse_args()

    return args
def load_data(dataset_dir, scene, keyframe_dir):
    poses = read_poses(os.path.join(dataset_dir, scene, "poses.txt"))   
    intrinsics = read_intrinsics(os.path.join(dataset_dir, scene, "intrinsics.txt"))
    
    descs = read_descriptors(os.path.join(keyframe_dir, scene, "descriptors.txt"))
    keyframe_names = read_img_names(os.path.join(keyframe_dir, scene, "keyframes.txt"))

    submap_array = np.load(os.path.join(keyframe_dir, scene, 'submap_split.npy'), allow_pickle=True)
    submap_splits = [{
        'start': item['start_time'],
        'end': item['end_time'],
        'frames': item['frames'].tolist()
    } for item in submap_array]

    return poses, intrinsics, descs, keyframe_names, submap_splits

def main():
    args = parse_arguments()

    # Load raw data
    poses, intrinsics, descs, keyframes, submap_splits = \
        load_data(args.dataset_dir, args.scene, args.keyframe_dir)
    
    submap_database = submap_splits[:int(len(submap_splits) * DB_Raio)]
    submap_query = submap_splits[int(len(submap_splits) * DB_Raio):]

    # Load database 
    map_root = os.path.join(args.dataset_dir, args.scene)
    graph = ImageGraph(map_root=map_root)
    for submap in submap_database:
        for img_name in submap['frames']:
            if img_name in keyframes:
                curr_node = ImageNode(img_name, None, None, descs[img_name], None, None, None, None, None, None, None, None)
                graph.add_node(curr_node)

    db_descs = np.array([node.get_descriptor() for node in graph.nodes.values()], dtype=np.float32)
    print('Map info:\n' + '\n'.join(node.id for node in graph.nodes.values()))

    # Load query
    for submap in submap_query:
        for img_name in submap['frames']:
            print(img_name)
            # curr_node = ImageNode(img_name, None, None, descs[img_name], None, None, None, None, None, None, None, None)
            
            # query_desc = curr_node.get_descriptor().reshape(1, -1)
            # dis, pred = perform_knn_search(db_descs, query_desc, query_desc.shape[1], [1])
            # for idx, node in enumerate(graph.nodes.values()):
            #     if idx == pred[0][0]:
            #         closest_node = node
            #         break

            # Compute matched keypoints and relative pose between curr_node and closest_node

            # Store results

    # Save results

if __name__ == "__main__":
    main()