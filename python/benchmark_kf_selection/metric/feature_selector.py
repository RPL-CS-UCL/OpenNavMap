import os
import numpy as np
from image_node import ImageNode
from utils.utils_vpr_method import perform_knn_search
from pathlib import Path

from matching import available_models, get_matcher

matcher = get_matcher('loftr', device='cuda')

class FeatureSelector:
    def __init__(self):
        # Keypoint matching threshold
        self.Mkpts_Threshold = 100  # Minimum matched keypoints to consider redundant

    def update_keyframes(self, data_path, submap, graph, descriptors):
        """Update keyframes based on matched keypoints criteria"""
        if len(graph) == 0:
            # Initialize graph with first submap
            for img_name in submap['frames']:
                curr_node = ImageNode(
                    img_name, 
                    None, 
                    None, 
                    descriptors[img_name],
                    None, None, None, None, None, None, None, None
                )
                graph[curr_node.id] = curr_node
        else:
            # Process new frames
            db_descriptors = np.array([node.get_descriptor() for node in graph.values()], dtype=np.float32)
            for img_name in submap['frames']:
                curr_node = ImageNode(
                    img_name,
                    None,
                    None,
                    descriptors[img_name],
                    None, None, None, None, None, None, None, None
                )

                # Find the closest node in the graph
                query_descriptor = curr_node.get_descriptor().reshape(1, -1)
                dis, pred = perform_knn_search(db_descriptors, query_descriptor, query_descriptor.shape[1], [1])
                for idx, node in enumerate(graph.values()):
                    if idx == pred[0][0]:
                        closest_node = node
                        break

                ##### Forward
                img0 = matcher.load_image(os.path.join(data_path, closest_node.id))
                img1 = matcher.load_image(os.path.join(data_path, curr_node.id))
                result = matcher(img0, img1)
                match_count = len(result['inlier_kpts0'])
                if match_count > self.Mkpts_Threshold: continue

                # Add new frame to the graph
                graph[curr_node.id] = curr_node
                # print(curr_node.id, closest_node.id, match_count)
                # match_count = num_mkpts.get((closest_node.id, curr_node.id), 0)
                edge_info = {
                    'match_count': match_count
                }
                closest_node.add_edge(curr_node, edge_info)

            # Check against all existing keyframes
            nodes_to_remove = []
            for db_node in graph.values():
                # The newest keyframe is not considered for deletion
                if not db_node.edges: continue

                ##### Backward
                max_keep = max(
                    (edge[1]['match_count'], edge[0]) 
                    for edge in db_node.edges.values()
                )
                match_count, node_to_viz = max_keep
                if match_count > self.Mkpts_Threshold:
                    nodes_to_remove.append(db_node)
                    print(f"Replace {db_node.id} with {node_to_viz.id} with match_count:{match_count}")

            for node in nodes_to_remove:
                if node.id in graph:
                    graph.pop(node.id)
                    
    def select_keyframes(self, 
                         data_path, 
                         descriptors, 
                         submap_database):
        """
        Main method for keyframe selection
        - num_mkpts: Dictionary containing matched keypoint counts between image pairs
        """
        graph = dict()
        
        # Process each submap
        for submap in submap_database:
            self.update_keyframes(
                data_path, 
                submap, 
                graph, 
                descriptors
            )
        
        # Return selected keyframes
        keyframes = list(graph.keys())
        print(f'Selected {len(keyframes)} keyframes')
        # print(', '.join(key for key in keyframes))
        
        return keyframes
