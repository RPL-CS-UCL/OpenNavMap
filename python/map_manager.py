#! /usr/bin/env python

# import os
# import numpy as np

import numpy as np
from pathlib import Path
from typing import Type
import shutil

from point_graph import PointGraph, PointGraphLoader
from image_graph import ImageGraph, ImageGraphLoader
from utils.utils_geom import convert_matrix_to_vec

class MapManager:
    def __init__(self, map_root: Path, map_id: int = 0):
        self.map_root = map_root
        self.map_id = map_id
        self._graphs = dict()
        
    def load_graphs(self, graph_configs):
        """Load multiple graphs with type-checked configurations"""
        self._graphs.clear()

        for graph_type, config in graph_configs.items():
            if graph_type == 'odom' or graph_type == 'trav':
                self._load_point_graph(graph_type, config)
            elif graph_type == 'covis':
                self._load_image_graph(graph_type, config)
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")
                
        graph = self._graphs[next(iter(self._graphs))]
        num_node = graph.get_num_node()
        max_num_node = graph.get_max_node_id()
        for graph_type in graph_configs.keys():
            assert self._graphs[graph_type].get_num_node() == num_node, \
                f"Number of nodes in {graph_type} does not match {num_node}"
            assert self._graphs[graph_type].get_max_node_id() == max_num_node, \
                f"Number of nodes in {graph_type} does not match {max_num_node}"

        print(f"Loaded graphs: {list(self._graphs.keys())}")

    def init_graphs(self, graph_configs):
        """Initialize multiple graphs"""
        for graph_type, config in graph_configs.items():
            if graph_type == 'odom' or graph_type == 'trav':
                self._graphs[graph_type] = PointGraph(self.map_root, graph_type)
            elif graph_type == 'covis':
                self._graphs[graph_type] = ImageGraph(self.map_root, graph_type)
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")

        print(f"Initialize graphs: {list(self._graphs.keys())}")

    def get_max_node_id(self) -> int:
        """Get maximum node ID across all graphs"""
        return max([graph.get_max_node_id() for graph in self._graphs.values()])

    def update_node_poses(self, estimate_pose):
        """Update node poses across all graphs using pose estimates"""
        for graph in self._graphs.values():
            for node in graph.nodes.values():
                pose_matrix = estimate_pose.atPose3(node.id).matrix()
                trans, quat = convert_matrix_to_vec(pose_matrix)
                node.set_pose(trans, quat)

    def adjust_all_ids(self, offset: int):
        """Adjust node IDs across all graphs"""
        for graph_type in self._graphs:
            self._adjust_graph_ids(graph_type, offset)

    def merge_graphs_from(self, other_map: 'MapManager'):
        """Merge nodes from another map's graphs"""
        for graph_type, other_graph in other_map._graphs.items():
            if graph_type in self._graphs:
                for node in other_graph.nodes.values():
                    self._graphs[graph_type].add_node(node)

    def copy_sensor_data(self, graph: 'ImageGraph'):
        """Copy sensor data files from source map"""
        for node in graph.nodes.values():
            # Handle RGB images
            if node.rgb_img_name:
                src = graph.map_root / node.rgb_img_name
                if src.exists():
                    new_name = f"seq/{node.id:06d}.color.jpg"
                    dest = self.map_root / new_name
                    shutil.copy(src, dest)
                    node.rgb_img_name = new_name

            # Handle depth images
            if node.depth_img_name:
                src = graph.map_root / node.depth_img_name
                if src.exists():
                    new_name = f"seq/{node.id:06d}.depth.png"
                    dest = self.map_root / new_name
                    shutil.copy(src, dest)
                    node.depth_img_name = new_name

    def add_inter_edges(self, edges, weight_func):
        for graph in self._graphs.values():
            graph.add_inter_edges(edges, weight_func)

    def save_to_file(self):
        """Save all graphs to files"""
        for graph_type, graph in self._graphs.items():
            graph.save_to_file()

    def _load_point_graph(self, graph_type: str, config):
        """Helper method for loading point-based graphs"""
        self._graphs[graph_type] = PointGraphLoader.load_data(
            map_root=self.map_root,
            edge_type=graph_type
        )

    def _load_image_graph(self, graph_type: str, config):
        """Helper method for loading image-based graphs"""
        self._graphs[graph_type] = ImageGraphLoader.load_data(
            map_root=self.map_root,
            resize=config['resize'],
            depth_scale=config['depth_scale'],
            load_rgb=config['load_rgb'],
            load_depth=config['load_depth'],
            normalized=config['normalized'],
            edge_type=graph_type
        )

    def _adjust_graph_ids(self, graph_type: str, offset: int):
        """Adjust node IDs for a specific graph"""
        graph = self._graphs.get(graph_type)
        if graph:
            for node in graph.nodes.values():
                node.id += offset

    @property
    def odom(self):
        """Access odometry graph with type hinting"""
        return self._graphs.get('odom')

    @property
    def trav(self):
        """Access traversability graph with type hinting"""
        return self._graphs.get('trav')

    @property
    def covis(self):
        """Access covisibility graph with type hinting"""
        return self._graphs.get('covis')

    @property
    def is_empty(self) -> bool:
        """Check if any graphs are loaded"""
        num_node = sum([graph.get_num_node() for graph in self._graphs.values()])
        return not (num_node > 0)

class TestMapManager():
	def __init__(self):
		pass
	
	def run_test(self):
		# Initialize the point graph
		map_root = Path('/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_aria/s00001/out_map0')
		graph_configs = {
			'odom': {},
			'trav': {},
			'covis': {
				'resize': (512, 288),
				'depth_scale': 0.0,
				'load_rgb': True
			},
		}
		map = MapManager(map_root)
		map.load_graphs(graph_configs)

if __name__ == '__main__':
	test_map_manager = TestMapManager()
	test_map_manager.run_test()