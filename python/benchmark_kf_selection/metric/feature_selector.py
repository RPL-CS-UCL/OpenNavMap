import numpy as np
from image_node import ImageNode
from image_graph import ImageGraph
from utils.utils_vpr_method import perform_knn_search

class FeatureSelector:
    def __init__(self):
        # Keypoint matching threshold
        self.Mkpts_Threshold = 100  # Minimum matched keypoints to consider redundant

    def update_keyframes(self, submap, graph, descriptors, num_mkpts):
        """Update keyframes based on matched keypoints criteria"""
        if graph.get_num_node() == 0:
            # Initialize graph with first submap
            for img_name in submap['frames']:
                curr_node = ImageNode(
                    img_name, 
                    None, 
                    None, 
                    descriptors[img_name],
                    None, None, None, None, None, None, None, None
                )
                graph.add_node(curr_node)
        else:
            # Process new frames
            nodes_to_remove = []
            db_descriptors = np.array([node.get_descriptor() for node in graph.nodes.values()], dtype=np.float32)
            for img_name in submap['frames']:
                curr_node = ImageNode(
                    img_name,
                    None,
                    None,
                    descriptors[img_name],
                    None, None, None, None, None, None, None, None
                )
                graph.add_node(curr_node)

                # Find the closest node in the graph
                query_descriptor = curr_node.get_descriptor().reshape(1, -1)
                dis, pred = perform_knn_search(db_descriptors, query_descriptor, query_descriptor.shape[1], [1])
                for idx, node in enumerate(graph.nodes.values()):
                    if idx == pred[0][0]:
                        closest_node = node
                        break

                # Add new frame to the graph
                match_count = num_mkpts.get((closest_node.id, curr_node.id), 0)                    
                edge_info = {'match_count': match_count}
                closest_node.add_edge(curr_node, edge_info)

            # Check against all existing keyframes
            for db_node in graph.nodes.values():
                if not db_node.edges: 
                    continue
                max_keep = max((edge[1]['match_count'], edge[0]) for edge in db_node.edges)
                match_count, node_to_viz = max_keep
                if match_count > self.Mkpts_Threshold:
                    nodes_to_remove.append(db_node)
                    print(f"Replace {db_node.id} with {node_to_viz.id} with match_count:{match_count}")

            graph.remove_node_list(nodes_to_remove)

    def select_keyframes(self, descriptors, num_mkpts, submap_database):
        """
        Main method for keyframe selection
        - num_mkpts: Dictionary containing matched keypoint counts between image pairs
        """
        graph = ImageGraph(map_root=None)
        
        # Process each submap
        for submap in submap_database:
            self.update_keyframes(submap, graph, descriptors, num_mkpts)
        
        # Return selected keyframes
        keyframes = list(graph.nodes.keys())
        print(f'Selected {len(keyframes)} keyframes')
        print(', '.join(key for key in keyframes))
        
        return keyframes
