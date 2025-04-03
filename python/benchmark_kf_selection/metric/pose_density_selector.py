import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../utils"))

import math
import numpy as np
from utils.utils_geom import compute_pose_error, convert_vec_to_matrix, convert_matrix_to_vec

class PoseDensitySelector:
    def __init__(self):
        self.Trans_Threshold = 5.0 # Translation threshold in meters: the smaller, the closer
        self.Rot_Threshold = 45.0  # Rotation threshold in degrees: the smaller, the closer
    def update_keyframes(self, submap, graph, poses):
        """Update keyframe graph with pose density criteria"""
        nodes_to_add = []
        for img_name in submap['frames']:
            # map-free pose format
            pose = poses[img_name]
            T_c2w = convert_vec_to_matrix(pose[4:], pose[:4], 'wxyz')
            trans, quat = convert_matrix_to_vec(np.linalg.inv(T_c2w), 'xyzw')
            curr_node = dict(id=img_name, trans=trans, quat=quat)
            nodes_to_add.append(curr_node)
        
        nodes_to_remove = set()
        for new_node in nodes_to_add:
            for idx, map_node in enumerate(graph):
                t_error, r_error = compute_pose_error(
                    (map_node['trans'], map_node['quat']),
                    (new_node['trans'], new_node['quat']), 
                    mode='vector'
                )
                if t_error < self.Trans_Threshold and r_error < self.Rot_Threshold:
                    nodes_to_remove.add(idx)
                print(f"{map_node['id']} -> {new_node['id']} with error {t_error:.3f}, {r_error:.3f}")

        print('Remove keyframs: ' + ', '.join([graph[idx]['id'] for idx in nodes_to_remove]))

        # update keyframes
        for idx in sorted(nodes_to_remove, reverse=True):
            del graph[idx]        

        graph += nodes_to_add

    def select_keyframes(self, poses, submap_database):
        """Main keyframe selection pipeline"""
        graph = []
        
        # Process all submaps sequentially
        for submap in submap_database:
            self.update_keyframes(submap, graph, poses)

        keyframes = [node['id'] for node in graph]
        print(f'Selected {len(keyframes)} keyframes')
        print(', '.join(key for key in keyframes))

        return keyframes
