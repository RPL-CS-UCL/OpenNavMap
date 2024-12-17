'''
More sequential matching models are available: https://github.com/mingu6/ProbFiltersVPR/tree/master/src/models
'''

import numpy as np
import torch
from typing import Tuple
from typing import Union
import faiss

class PlaceRecognitionTopologicalFilter:
    '''
    Adapted from https://github.com/mingu6/ProbFiltersVPR/blob/master/src/models/TopologicalFilter.py
    '''
    def __init__(self):
        pass

    def initialize_model(self, db_descriptors, db_poses, delta=5, recall_values=5):
        """
        Initialize the VPRTopologicalFilter object.
        Initialize the belief distribution - uniform distribution

        Args:
            db_descriptors (numpy.ndarray): The map descriptors.
            db_poses (numpy.ndarray): The database poses (translation)
            delta (int, optional): The delta value. Defaults to 5.
            prop_radius (float, optional): The propagation radius. Defaults to 10.0.
            recall_values (int, optional): The number of recall values. Defaults to 5.
        """
        # get map descriptors
        self.db_descriptors = db_descriptors
        self.db_poses = db_poses
        self.recall_values = recall_values  

        # initialize hidden states and obs likelihood parameters
        self.delta = delta
        self.lambda1 = None
        self.belief = None

        self.belief = np.ones(self.db_descriptors.shape[0]) / self.db_descriptors.shape[0]

    def get_back_prop_node(self, node) -> list:
        preds = {node.id}
        for edge in node.edges:
            if node.id > edge[0].id:
                preds.add(edge[0].id)
            for sub_edge in edge[0].edges:
                if edge[0].id > sub_edge[0].id:
                    preds.add(sub_edge[0].id)
        preds = list(set(preds))
        return preds

    def comp_dist_descriptor(self, descriptor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        ##### Option 1: cosine similarity
        # dists = np.sqrt(2 - 2 * np.dot(self.db_descriptors, descriptor.reshape(-1)))
        ##### Option 2: euclidean distance
        dists = np.linalg.norm(self.db_descriptors - descriptor, axis=1)
        return dists

    def obs_lhood(self, descriptor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        '''Observation likelihood of the query descriptor'''
        dists = self.comp_dist_descriptor(descriptor)
        vsim = np.exp(-self.lambda1 * dists)
        return vsim

    def match(self, db_map, query_desc: Union[np.ndarray, torch.Tensor]):
        '''
        Match the query image to the topological map.

        Runs a prediction step followed by a measurement step:
        - Prediction: Propagate belief mass using the transition model
        - Measurement: Update belief mass using the observation likelihood

        After the process, the map node with the highest probability is
        returned as the subgoal.

        Returns:
        - recall_preds: the top recall indices of the matched map nodes
        - pred: the index of the matched map node
        - prob: the probability of the matching
        '''
        # Initialize the lambda
        if self.lambda1 is None:
            dists = self.comp_dist_descriptor(query_desc)
            descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
            self.lambda1 = np.log(self.delta) / (descriptor_quantiles[1] - descriptor_quantiles[0])

        # Prediction step
        belief_pred = np.zeros_like(self.belief)
        for i in range(len(belief_pred)):
            node = db_map.get_node(i)
            back_prop_node_id = self.get_back_prop_node(node)
            belief_pred[i] = np.sum(self.belief[back_prop_node_id])
        obs_lhood = self.obs_lhood(query_desc)
        # Measurement step
        self.belief = obs_lhood * belief_pred
        self.belief /= self.belief.sum()

        # print('Prediction: Belief')
        # str = ' '.join([f'{x:.2f}' for x in belief_pred])
        # print(str)
        # print('Measurement Update: Belief')
        # str = ' '.join([f'{x:.2f}' for x in self.belief])
        # print(str)

        # Get the top recall values
        recall_preds = np.argsort(self.belief)[-self.recall_values:][::-1]
        pred = np.argmax(self.belief)
        prob = self.belief[pred]
    
        return recall_preds, pred, prob
