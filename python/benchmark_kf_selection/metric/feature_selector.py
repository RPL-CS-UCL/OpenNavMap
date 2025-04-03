import numpy as np
from image_node import ImageNode
from image_graph import ImageGraph
from utils.utils_vpr_method import perform_knn_search

class FeatureSelector:
    def __init__(self):
        # Keypoint matching threshold
        self.Mkpts_Threshold = 300  # Minimum matched keypoints to consider redundant

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
            nodes_to_add, nodes_to_remove = [], set()
            db_descriptors = np.array([node.get_descriptor() for node in graph.nodes.values()], dtype=np.float32)
            for img_name in submap['frames']:
                curr_node = ImageNode(
                    img_name,
                    None,
                    None,
                    descriptors[img_name],
                    None, None, None, None, None, None, None, None
                )
                nodes_to_add.append(curr_node)

                # Find the closest node in the graph
                query_descriptor = curr_node.get_descriptor().reshape(1, -1)
                dis, pred = perform_knn_search(db_descriptors, query_descriptor, query_descriptor.shape[1], [1])
                for idx, node in enumerate(graph.nodes.values()):
                    if idx == pred[0][0]:
                        closest_node = node
                        break

                # Check against all existing keyframes
                is_redundant = False
                for map_node in graph.nodes.values():
                    # Get matched keypoints count
                    match_count = num_mkpts.get((map_node.id, curr_node.id), 0)                    
                    if match_count > self.Mkpts_Threshold:
                        nodes_to_remove.add(map_node)
                
                    print(f"{map_node.id} -> {curr_node.id} with mkpts {match_count}")

                print('Remove keyframs: ' + ', '.join([node.id for node in nodes_to_remove]))

                # Remove the old keyframe if redundant
                if node in nodes_to_remove:
                    graph.remove_node(node)

                for node in nodes_to_add:
                    graph.add_node(node)

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
