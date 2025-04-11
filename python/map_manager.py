#! /usr/bin/env python

# import os
# import numpy as np

import numpy as np
from pathlib import Path
from typing import Type

from point_graph import PointGraph, PointGraphLoader
from image_graph import ImageGraph, ImageGraphLoader
from utils.utils_geom import convert_matrix_to_vec

class MapManager:
	def __init__(self, map_root: Path, map_id: int = 0):
		self.map_root = map_root
		self.map_id = map_id
		self._graphs = dict()

	def __str__(self):
		out_str = 'Visualize the current map:\n'
		for graph_type, graph in self.graphs.items():
			out_str += f"{graph_type}: {graph} {type(graph)}\n"
		return out_str

	def load_graphs(self, graph_configs):
		"""Load multiple graphs with type-checked configurations"""
		assert len(graph_configs) > 0

		self.graphs.clear()
		for graph_type, config in graph_configs.items():
			if graph_type == 'odom':
				self._load_point_graph(graph_type, config)
			elif graph_type == 'trav':
				self._load_point_graph(graph_type, config)
			elif graph_type == 'covis':
				self._load_image_graph(graph_type, config)
			else:
				raise ValueError(f"Unknown graph type: {graph_type}")

		# The number of nodes in each graph are not necessarily the same since node removal happens
		# But the maximum node ID should be the same across all graphs since they have been imported 
		# 	the same number of nodes
		max_node_id = [graph.get_max_node_id() for graph in self.graphs.values()]
		assert all(n == max_node_id[0] for n in max_node_id), \
			f"Maximum number of nodes in {graph_type} does not match {max_node_id[0]}"

		print(f"Loaded graphs: {list(self.graphs.keys())}")

	def init_graphs(self, graph_configs):
		"""Initialize multiple graphs"""
		for graph_type, config in graph_configs.items():
			if graph_type == 'odom' or graph_type == 'trav':
				self.graphs[graph_type] = PointGraph(self.map_root, graph_type)
			elif graph_type == 'covis':
				self.graphs[graph_type] = ImageGraph(self.map_root, graph_type)
			else:
				raise ValueError(f"Unknown graph type: {graph_type}")

		print(f"Initialize graphs: {list(self.graphs.keys())}")

	def get_max_node_id(self) -> int:
		"""Get maximum node ID across all graphs"""
		return max([graph.get_max_node_id() for graph in self.graphs.values()])

	def update_node_poses(self, estimate_pose):
		"""Update node poses across all graphs using pose estimates"""
		for graph in self.graphs.values():
			for node in graph.nodes.values():
				pose_matrix = estimate_pose.atPose3(node.id).matrix()
				trans, quat = convert_matrix_to_vec(pose_matrix)
				node.set_pose(trans, quat)

	def adjust_all_ids(self, offset: int):
		"""Adjust node IDs across all graphs"""
		# Need adjustment
		if offset > 0:
			for graph in self.graphs.values():
				self._adjust_graph_ids(graph, offset)

	def merge_graphs_from(self, other_map: 'MapManager'):
		"""Merge nodes from another map's graphs"""
		for graph_type, other_graph in other_map._graphs.items():
			if graph_type in self.graphs:
				for node in other_graph.nodes.values():
					self.graphs[graph_type].add_node(node)

	def add_inter_edges(self, edges, weight_func):
		for graph in self.graphs.values():
			graph.add_inter_edges(edges, weight_func)

	def save_to_file(self):
		"""Save all graphs to files"""
		if 'covis' in self.graphs:
			self.covis.save_to_file(edge_only=False)
		if 'odom' in self.graphs:
			self.odom.save_to_file(edge_only=False)
		if 'trav' in self.graphs:
			self.trav.save_to_file(edge_only=True)

	def _load_point_graph(self, graph_type: str, config):
		"""Helper method for loading point-based graphs"""
		self.graphs[graph_type] = PointGraphLoader.load_data(
			map_root=self.map_root,
			edge_type=graph_type
		)

	def _load_image_graph(self, graph_type: str, config):
		"""Helper method for loading image-based graphs"""
		self.graphs[graph_type] = ImageGraphLoader.load_data(
			map_root=self.map_root,
			resize=config['resize'],
			depth_scale=config['depth_scale'],
			load_rgb=config['load_rgb'],
			load_depth=config['load_depth'],
			normalized=config['normalized'],
			edge_type=graph_type
		)

	def _adjust_graph_ids(self, graph: 'ImageGraph | PointGraph', offset: int):
		"""Adjust node IDs for a specific graph, including adjusting the key of nodes and edges"""
		new_nodes = {}
		for node in graph.nodes.values():
			new_nodes[node.id + offset] = node
			new_nodes[node.id + offset].id += offset
		graph.set_node(new_nodes)

		for node in graph.nodes.values():
			new_edges = {}
			for edge in node.edges.values():
				new_edges[edge[0].id] = edge				
			node.set_edge(new_edges)

	def update_edges(self, src_edges, dst_graph_type):
		"""Convert edges between graph types using list comprehension"""
		dst_graph = self.graphs[dst_graph_type]
		for n0, n1, attr, weight in src_edges:
			if dst_graph.contain_node(n0) and dst_graph.contain_node(n1):
				yield (dst_graph.get_node(n0.id), dst_graph.get_node(n1.id), attr, weight)
	@property
	def graphs(self):
		return self._graphs

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
				'load_rgb': True,
				'load_depth': False,
				'normalized': False
			},
		}
		cur_map = MapManager(map_root)
		cur_map.load_graphs(graph_configs)
		print(cur_map)
		print(type(cur_map))

if __name__ == '__main__':
	test_map_manager = TestMapManager()
	test_map_manager.run_test()