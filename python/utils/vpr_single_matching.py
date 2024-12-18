#! /usr/bin/env python

import faiss
import torch
import numpy as np
from typing import Union

class PlaceRecognitionSingleMatching:
    def __init__(self):
        pass
    
    def initialize_model(self, db_descriptors, recall_values=5):
        # get map descriptors
        self.db_descriptors = db_descriptors
        self.recall_values = recall_values  

        self.db_faiss_index = faiss.IndexFlatL2(db_descriptors.shape[1])
        self.db_faiss_index.add(db_descriptors)

    def match(self, db_map, query_desc: np.ndarray):
        _, recall_preds = self.db_faiss_index.search(query_desc, self.recall_values)
        return recall_preds[0], recall_preds[0][0], 1.0

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
    import argparse
    from image_graph import ImageGraphLoader as GraphLoader
    import pycpptools.src.python.utils_math as pytool_math
    from tqdm import tqdm

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_map_path", type=str, help="Path to the database map file")
    parser.add_argument("--query_map_path", type=str, help="Path to the query map file")
    args = parser.parse_args()
    # Load database and query
    db_map = GraphLoader.load_data(
        args.db_map_path,
        [512, 288],
        depth_scale=0.0,
        load_rgb=True,
        load_depth=False,
        normalized=False
    )
    query_map = GraphLoader.load_data(
        args.query_map_path,
        [512, 288],
        depth_scale=0.0,
        load_rgb=True,
        load_depth=False,
        normalized=False
    )
    # Performance test
    db_descriptors = np.array([node.get_descriptor() for _, node in db_map.nodes.items()], dtype="float32")
    model = PlaceRecognitionSingleMatching()
    model.initialize_model(db_descriptors, recall_values=5)
    preds = []
    for node in tqdm(query_map.nodes.values()):
        query_desc = node.get_descriptor()
        recall_preds, pred, score = model.match(db_map, query_desc.reshape(1, -1))
        preds.append(recall_preds)

    succ = 0
    for i, node in enumerate(query_map.nodes.values()):
        ref_map_node = db_map.nodes[preds[i][0]]
        dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
            node.trans_gt, node.quat_gt, ref_map_node.trans_gt, ref_map_node.quat_gt)
        if dis_tsl < 10.0 and dis_angle < 90.0:
            succ += 1
            print(f"Correct prediction: Query {node.id} - DB: {preds[i][0]}")
        else:
            print(f"Wrong prediction: Query {node.id} - DB: {preds[i][0]}")
    print(f"Success rate: {succ / len(query_map.nodes)}")