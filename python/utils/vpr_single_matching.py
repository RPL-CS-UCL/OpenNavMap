'''
More sequential matching models are available: https://github.com/mingu6/ProbFiltersVPR/tree/master/src/models
'''

import numpy as np
import torch
from typing import Tuple
from typing import Union
import faiss

class PlaceRecognitionSingleMatching:
    def __init__(self, db_descriptors, db_poses, recall_values=5):
        # get map descriptors
        self.db_descriptors = db_descriptors
        self.recall_values = recall_values  

        self.db_faiss_index = faiss.IndexFlatL2(db_descriptors.shape[1])
        self.db_faiss_index.add(db_descriptors)

        self.pose_faiss_index = faiss.IndexFlatL2(3)
        self.pose_faiss_index.add(db_poses)

    def match(self, query_desc: Union[np.ndarray, torch.Tensor]):
        dis, preds = self.db_faiss_index.search(1, query_desc, self.recall_values)
        return preds[0]
