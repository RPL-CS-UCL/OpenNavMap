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
    # NOTE(gogojjh): the windonw_lower and window_upper change the performance significantly
    def __init__(self, db_descriptors, db_poses, delta=5, prop_radius=10.0, recall_values=5):
        # get map descriptors
        self.db_descriptors = db_descriptors

        # initialize hidden states and obs likelihood parameters
        self.delta = delta
        self.lambda1 = 0.0
        self.belief = None

        self.prop_radius = prop_radius
        self.recall_values = recall_values  

        self.pose_faiss_index = faiss.IndexFlatL2(3)
        self.pose_faiss_index.add(db_poses)

    def get_back_prop_node(self, query_pose) -> list:
        """
        Retrieves the nearest node in the graph that can be propagated from the given query pose.

        Parameters:
            query_pose (numpy.ndarray): The query pose for which to find the nearest node.

        Returns:
            list: A list containing the distance to the nearest node and the index of the nearest node.

        """
        dis, preds = self.pose_faiss_index.range_search(1, query_pose, radius=self.prop_radius)
        return preds[0]

    def initialize_model(self):
        '''
        Initialize the belief distribution - uniform distribution
        '''
        self.belief = np.ones(self.db_descriptors.shape[0]) / self.db_descriptors.shape[0]

    def comp_dist_descriptor(self, descriptor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        ##### Option 1: cosine similarity
        # dists = np.sqrt(2 - 2 * np.dot(self.db_descriptors, descriptor))
        ##### Option 2: euclidean distance
        dists = np.linalg.norm(self.db_descriptors - descriptor, axis=1)
        return dists

    def obs_lhood(self, descriptor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        '''Observation likelihood of the query descriptor'''
        dists = self.comp_dist_descriptor(descriptor)
        vsim = np.exp(-self.lambda1 * dists)
        return vsim

    def match(self, query_desc: Union[np.ndarray, torch.Tensor])
        '''
        Match the query image to the topological map.

        Runs a prediction step followed by a measurement step:
        - Prediction: Propagate belief mass using the transition model
        - Measurement: Update belief mass using the observation likelihood

        After the process, the map node with the highest probability is
        returned as the subgoal.

        Returns:
        - pred: the index of the matched map node
        - prob: the probability of the matching
        '''
        if self.belief is None:
            dists = self.comp_dist_descriptor(query_desc)
            # Init for lambda1
            descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
            self.lambda1 = np.log(self.delta) / (descriptor_quantiles[1] - descriptor_quantiles[0])
            # Init for belief distribution
            self.belief = np.exp(-self.lambda1 * dists)
            belief_pred = np.copy(self.belief)
        else:
            belief_pred = np.zeros_like(self.belief)
            for i in range(len(self.belief)):
                back_prop_node_id = self.get_back_prop_node(self.db_poses[i].reshape(1, -1))
                print(back_prop_node_id)
                belief_pred[i] = np.sum(self.belief[back_prop_node_id])
            obs_lhood = self.obs_lhood(query_desc)
            self.belief = obs_lhood * belief_pred
            self.belief /= self.belief.sum()

        print('Prediction: Belief')
        str = ' '.join([f'{x:.2f}' for x in belief_pred])
        print(str)
        print('Measurement Update: Belief')
        str = ' '.join([f'{x:.2f}' for x in self.belief])
        print(str)

        # Get the top recall values
        recall_preds = np.argsort(self.belief)[-self.recall_values:][::-1]
        pred = np.argmax(self.belief)
        prob = self.belief[pred]
    
        return recall_preds, pred, prob
