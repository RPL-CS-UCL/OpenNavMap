import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

import numpy as np
import math
from pathlib import Path

from image_node import ImageNode
from utils.utils_vpr_method import perform_knn_search

class LandmarkSelector:
    def __init__(self):
        # Parameters for probability calculation
        self.Q_th = 19.0     # Midpoint for quality sigmoid (the command threshold for low-light and motion blur)
        self.k_Q = 0.1       # Quality sigmoid steepness (higher, more sensitive)

        self.G_th = 40.0      # Information gain threshold
        self.k_G = 0.06       # Information gain sensitivity (higher, more sensitive)
        
        self.T_th = 24 * 3600.0   # Timestamp threshold (second) -> one day
        self.lambda_T = 0.02     # Timestamp sensitivity (very slow decay) (higher value: more sensitive) -> 3 months with 0.165 prob decay

        self.P_iqa_th = 0.5
        self.P_acc_th = 0.35
        self.P_keep_th = 0.3

    # The prbability of keeping the frame
    def quality_probability(self, Q):
        """Sigmoid function for image quality (0-100). Higher is better."""
        return 1 / (1 + math.exp(-self.k_Q * (Q - self.Q_th)))

    def delta_quality_probability(self, Q):
        """Sigmoid function for image quality (-100-100). Higher is better."""
        return 1 / (1 + math.exp(-self.k_Q * Q))

    def gain_probability(self, G):
        """Sigmoid increase function for information gain (0-1). Higher is better."""
        return 1 / (1 + math.exp(-self.k_G * (G * 100.0 - self.G_th)))

    def time_probability(self, T):
        """Exponential decay based on time elapsed. Smaller (recent) is better."""
        """use the min() to ensure the probability is always between 0 and 1"""
        return min(1.0, math.exp(-self.lambda_T * T / self.T_th))
    
    def time_boost_probability(self, T):
        """Exponential increase for forward pass. Larger time difference encourages acceptance."""
        """Returns value >= 1.0 to boost acceptance probability for temporally diverse frames"""
        return math.exp(self.lambda_T * T / self.T_th)

    def compute_forward_prob(self, Q, G, T, use_iqa=True, use_ig=True, use_td=True):
        """Calculate input probability to determine whether accepting a new keyframe."""
        P_Q = self.quality_probability(Q) if use_iqa else 1.0
        P_G = self.gain_probability(G) if use_ig else 1.0
        P_T_boost = self.time_boost_probability(T) if use_td else 1.0
        acc_prob = P_Q * P_G * P_T_boost       

        return acc_prob

    def compute_backward_prob(self, G, T, use_ig=True, use_td=True):
        """Calculate posterior probability for a keyframe."""
        P_G = self.gain_probability(G) if use_ig else 1.0
        P_T = self.time_probability(T) if use_td else 1.0
        keep_prob = P_G * P_T + 1e-6
        
        return keep_prob

    def print_prefilter_prob(self, Q, use_iqa=True):
        """Print the probability of the frame being accepted by the prefilter."""
        P_Q = self.quality_probability(Q) if use_iqa else 1.0
        return f"IQA: {Q:.2f}. P_Q: {P_Q:.2f}"

    def print_each_forward_prob(self, Q, G, dT, use_iqa=True, use_ig=True, use_td=True):
        """Calculate posterior probability for a keyframe."""
        P_Q = self.quality_probability(Q) if use_iqa else 1.0
        P_G = self.gain_probability(G) if use_ig else 1.0
        P_T_boost = self.time_boost_probability(dT) if use_td else 1.0
        P = P_Q * P_G * P_T_boost + 1e-6

        return f"Q: {Q:.2f}, G: {G:.2f}, dT: {dT:.2f}. P_Q: {P_Q:.2f}, P_G: {P_G:.2f}, P_T_boost: {P_T_boost:.2f}, P: {P:.2f}"

    def print_each_backward_prob(self, G, dT, use_ig=True, use_td=True):
        """Calculate posterior probability for a keyframe."""
        P_G = self.gain_probability(G) if use_ig else 1.0
        P_T = self.time_probability(dT) if use_td else 1.0
        P = P_G * P_T + 1e-6

        return f"G: {G:.2f}, dT: {dT:.2f}. P_G: {P_G:.2f}, P_T: {P_T:.2f}, P: {P:.2f}"

    def update_keyframes(self, map_root, submap, graph, timestamps, descriptors, iqa_scores, info_gain):
        if len(graph) == 0:
            for img_name in submap['frames']:
                curr_node = ImageNode(img_name, None, None, descriptors[img_name], timestamps[img_name][0], None, None, None, None, None, None, None)
                curr_node.rgb_img_name = str(map_root / img_name)
                curr_node.iqa_score = iqa_scores[img_name][0]
                graph[curr_node.id] = curr_node
        else:
            db_descriptors = np.array([node.get_descriptor() for node in graph.values()], dtype=np.float32)
            for img_name in submap['frames']:
                curr_node = ImageNode(
                    img_name, 
                    None, 
                    None, 
                    descriptors[img_name], 
                    timestamps[img_name][0], 
                    None, None, None, None, None, None, None
                )
                curr_node.rgb_img_name = str(map_root / img_name)
                curr_node.iqa_score = iqa_scores[img_name][0]

                # Find the closest node in the graph
                query_descriptor = curr_node.get_descriptor().reshape(1, -1)
                dis, pred = perform_knn_search(db_descriptors, query_descriptor, query_descriptor.shape[1], [1])
                for idx, node in enumerate(graph.values()):
                    if idx == pred[0][0]:
                        closest_node = node
                        break

                ##### Forward
                # Determine whether to add new frame
                time_diff = abs(curr_node.time - closest_node.time)
                acc_prob = self.compute_forward_prob(
                    curr_node.iqa_score, 
                    info_gain[(curr_node.id, closest_node.id)], # how much information is gained by curr_node
                    time_diff # time difference to encourage temporal diversity
                )
                # print(f"Accept prob {acc_prob:.3f}: {curr_node.id}")
                if not acc_prob > self.P_acc_th: continue
                
                # Add new frame to the graph
                graph[curr_node.id] = curr_node
                edge_info = {
                    'G': info_gain[(closest_node.id, curr_node.id)],
                    'dt': curr_node.time - closest_node.time,
                }
                closest_node.add_edge(curr_node, edge_info)
            
            # Check whether old keyframe should be deleted
            nodes_to_remove = []
            for db_node in graph.values():
                # The newest keyframe is not considered for deletion
                if not db_node.edges: continue
                
                ##### Backward
                # Compute the keeping probability
                min_keep = min(
                    (self.compute_backward_prob(
                        edge[1]['G'], 
                        edge[1]['dt']
                    ), edge[0])
                    for edge in db_node.edges.values()
                )
                P_keep, node_to_viz = min_keep
                # Check whether remove the old node
                if P_keep < self.P_keep_th:
                    nodes_to_remove.append(db_node)
                    print(f"Replace {db_node.id} with {node_to_viz.id} with Prob:{P_keep:.3f}")

            for node in nodes_to_remove:
                if node.id in graph:
                    graph.pop(node.id)

    def select_keyframes(self, 
                         data_path, 
                         timestamps, 
                         descriptors, 
                         iqa_scores, 
                         info_gain, 
                         submap_database):
        """
        Main method to select keyframes from provided data.
        timestamps, 
        descriptors, iqa_scores, info_gain: metadata dictionaries
        submap_database: list of submap dicts containing frame names
        """

        # Graph to store keyframes and their overlapping relationships
        graph = dict()
        
        # Process each submap
        for submap in submap_database:
            self.update_keyframes(
                Path(data_path), 
                submap, 
                graph, 
                timestamps, 
                descriptors, 
                iqa_scores, 
                info_gain
            )
            
        keyframes = [key for key in graph.keys()]
        print(f'Selected {len(keyframes)} keyframes')
        # print(', '.join(key for key in keyframes))

        return keyframes

if __name__ == '__main__':
    lm_selector = LandmarkSelector()
    PQ = lm_selector.quality_probability(19.0)
    print(f"Prob Quality: {PQ:.3f}")

    PdQ = lm_selector.delta_quality_probability(12.0)
    print(f"Prob Quality: {PdQ:.3f}")    

    PG = lm_selector.gain_probability(0.3)
    print(f"Prob Gain: {PG:.3f}")

    PT = lm_selector.time_probability(3600.0 * 24 * 30 * 3) # 3 months (backward)
    print(f"Prob Time (backward, decay): {PT:.3f}")
    
    PT_boost = lm_selector.time_boost_probability(3600.0 * 24 * 30 * 3) # 3 months (forward)
    print(f"Prob Time (forward, boost): {PT_boost:.3f}")