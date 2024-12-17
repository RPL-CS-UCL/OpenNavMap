'''
More sequential matching models are available: https://github.com/mingu6/ProbFiltersVPR/tree/master/src/models
'''

import numpy as np
import torch
from typing import Tuple
from typing import Union
import faiss

class PlaceRecognitionSingleMatching:
    def __init__(self):
        pass
    
    def initialize_model(self, db_descriptors, db_poses, recall_values=5):
        # get map descriptors
        self.db_descriptors = db_descriptors
        self.db_poses = db_poses
        self.recall_values = recall_values  

        self.db_faiss_index = faiss.IndexFlatL2(db_descriptors.shape[1])
        self.db_faiss_index.add(db_descriptors)

    def match(self, db_map, query_desc: np.ndarray):
        _, recall_preds = self.db_faiss_index.search(query_desc, self.recall_values)
        return recall_preds[0], recall_preds[0][0], 1.0
