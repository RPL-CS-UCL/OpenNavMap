#! /usr/bin/env python

import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class PlaceRecognitionSeqMatching:
    def __init__(self):
        self.wContrast = 10
        self.numVel = 10
        self.vMin = 1
        self.vMax = 1.5 # vMax * seq_length < db_descriptors.shape[0]
        self.matchWindow = 20 # window size for selecting the best score < number of template
        self.enhance = False # False for learning-based VPR methods
    
    def initialize_model(self, db_descriptors, recall_values=5):
        self.db_descriptors = db_descriptors
        self.recall_values = recall_values

    def match(self, query_descriptors, id):
        D = self._compute_diff_matrix(query_descriptors)
 
        # ################################
        # plt.figure(figsize=(8, 8))
        # plt.imshow(D, cmap='viridis', aspect='auto')
        # plt.colorbar(label='Difference')
        # plt.xlabel('Query Descriptor Index')
        # plt.ylabel('Database Descriptor Index')
        # plt.title('Difference Matrix')
        # diff_matrix_path = f"/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map0/preds/diff_matrix_euc_{id}.png"
        # plt.savefig(diff_matrix_path)
        # ################################

        if self.enhance:
            D = self._enhance_contrast(D)

        template_scores = self._score_ref_templates(D)
        recall_preds, pred, score = self._locate_best_match(template_scores)
        return recall_preds, pred, score

    def _compute_diff_matrix(self, query_descriptors) -> np.ndarray:
        """
            Return:
                D: np.ndarray, db_descriptors.shape[0] x query_descriptors.shape[0]
        """
        ##### Option 1: cosine similarity
        # D = np.sqrt(2 - 2 * np.dot(self.db_descriptors, query_descriptors.transpose()))
        ##### Option 2: euclidean distance
        D = np.linalg.norm(query_descriptors[None, :, :] - self.db_descriptors[:, None, :], axis=2)
        return D

    def _enhance_contrast(self, D):
        nref = D.shape[0]
        Denhanced = np.empty_like(D)
        for i in range(nref):
            # reference indices of window around each reference image
            idx_lower = max(i - int(self.wContrast / 2), 0)
            idx_upper = min(i + int(self.wContrast / 2) + 1, nref - 1)
            # local normalization of window given by indices above
            Denhanced[i, :] = (
                D[i, :] - np.mean(D[idx_lower:idx_upper, :], axis=0)
            ) / np.std(D[idx_lower:idx_upper, :], axis=0)
        return Denhanced

    def _score_ref_templates(self, D):
        # N: number of database descriptors
        # L: number of query descriptors with sequence length
        N, L = D.shape

        # v = vMin, vMin+vStep, ..., vMax
        velocities = np.linspace(self.vMin, self.vMax, self.numVel + 1)
        # t = 0, ..., L
        times = np.arange(L)
        # i = 0, ..., max_ind <- truncated so line search not cut off
        max_ind = int(N - 1 - self.vMax * L)
        # last template image to begin sequence matching on: 0, 1, ..., max_ind - 1
        refs = np.arange(max_ind) 
        # D score for best velocity for each starting point (template image)
        # optD[i]: best score for sequence starting at template i
        optD = np.empty(max_ind); optD[:] = np.inf

        for vel in velocities:
            # indices in D for line search given a particular velocity
            # include all template number
            row_indices = (
                np.floor(refs[:, np.newaxis] + vel * times[np.newaxis, :])
                .astype(int)
                .reshape(-1)
            )
            col_indices = np.tile(times, max_ind)
            # evaluate D at indices and sum to get aggregate difference
            Dsum = np.sum(D[row_indices, col_indices].reshape(max_ind, L), axis=1)
            # for sequence matching scores better than
            # prior scores (under different velocities), update
            ind_better = Dsum < optD
            optD[ind_better] = Dsum[ind_better]
        return optD

    def _locate_best_match(self, template_scores):
        # indices of best match and window around it
        iOpt = np.argmin(template_scores)
        iWinL = np.maximum(iOpt - int(self.matchWindow / 2), 0)
        iWinU = np.minimum(iOpt + int(self.matchWindow / 2), len(template_scores))
        # check best match outside window
        outside_scores = np.concatenate((template_scores[:iWinL], template_scores[iWinU:]))
        optOutside = min(outside_scores)
        # exit()
        # for negative scores, u \in [0, 1]
        # increases the score... adjust
        if optOutside > 0:
            mu = template_scores[iOpt] / optOutside
        else:
            mu = optOutside / template_scores[iOpt]

        recall_preds = np.argsort(template_scores)[:self.recall_values]

        return recall_preds, iOpt, mu

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
    model = PlaceRecognitionSeqMatching()
    model.initialize_model(db_descriptors)

    query_descriptors = np.array([node.get_descriptor() for _, node in query_map.nodes.items()], dtype="float32")
    preds = []
    for node in tqdm(query_map.nodes.values()):
        if node.id + 10 < query_descriptors.shape[0]:
            query_descs = query_descriptors[node.id:node.id + 10, :]
            recall_preds, pred, score = model.match(query_descs, node.id)
            preds.append(recall_preds)

    succ = 0
    for i, node in enumerate(query_map.nodes.values()):
        if i >= len(preds): break
        ref_map_node = db_map.nodes[preds[i][0]]
        dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
            node.trans_gt, node.quat_gt, ref_map_node.trans_gt, ref_map_node.quat_gt)
        if dis_tsl < 10.0:
            succ += 1
            print(f"Correct prediction: Query {node.id} - DB: {preds[i][0]}")
        else:
            print(f"Wrong prediction: Query {node.id} - DB: {preds[i][0]}")
    print(f"Success rate: {succ / len(query_map.nodes)}")