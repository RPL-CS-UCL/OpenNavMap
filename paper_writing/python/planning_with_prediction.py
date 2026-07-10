#!/usr/bin/env python

import _bootstrap_imports  # noqa: F401

import argparse
import numpy as np
import heapq
import open3d as o3d
import pathlib
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
from utils.utils_setting_color_font import setting_font, acquire_color_palette 
import matplotlib.pyplot as plt

PALLETE = acquire_color_palette()

class PathPlanner:
	def __init__(self, args):
		self.args = args
		self.planner_name = args.planner
		
		if self.planner_name == "plan_with_graph":
			self.load_graph_data()
		elif self.planner_name == "plan_with_point":
			self.load_point_cloud(args.point_cloud)
			self.resolution = 0.2  # Grid resolution in meters

	def load_graph_data(self):
		"""Load pre-built traversability graph"""
		from point_graph import PointGraphLoader as GraphLoader
		map_root = pathlib.Path(self.args.map_path)
		self.point_graph = GraphLoader.load_data(map_root, edge_type='trav')
		self.map_node_position = np.array([node.trans for _, node in self.point_graph.nodes.items()])
		print(f"Loaded Traversability Graph: {str(self.point_graph)}")

	def load_point_cloud(self, path):
		"""Load and process point cloud data"""
		pcd = o3d.io.read_point_cloud(path)
		self.point_cloud = np.asarray(pcd.points)
		print(f"Loaded point cloud with {len(self.point_cloud)} points")

	def create_occupancy_grid(self):
		"""Create 2D occupancy grid for current height slice"""
		# Filter points within ±0.3m of start height
		z_slice = (self.start_point[2] - 0.2 <= self.point_cloud[:,2]) & \
				 (self.point_cloud[:,2] <= self.start_point[2] + 0.2)
		self.point_cloud = self.point_cloud[z_slice]
		slice_points = self.point_cloud[:,:2]  # Use only XY coordinates
				
		# Initialize and populate grid
		self.grid_shape = ((self.max_bounds - self.min_bounds) / self.resolution).astype(int) + 1		
		grid = np.zeros(self.grid_shape, dtype=bool)
		indices = ((slice_points - self.min_bounds) / self.resolution).astype(int)
		
		for idx in indices:
			if 0 <= idx[0] < self.grid_shape[0] and 0 <= idx[1] < self.grid_shape[1]:
				grid[tuple(idx)] = True
				
		return grid

	def graph_planning(self):
		"""Dijkstra's algorithm on pre-built graph"""
		start_idx = np.argmin(cdist([self.start_point], self.map_node_position))
		end_idx = np.argmin(cdist([self.end_point], self.map_node_position))
		
		from utils.utils_shortest_path import dijk_shortest_path
		_, path_nodes = dijk_shortest_path(self.point_graph,
										 self.point_graph.get_node(start_idx),
										 self.point_graph.get_node(end_idx))
		for node in path_nodes:
			print(node.id)
		return [node.trans for node in path_nodes]

	def a_star_planning(self, grid, min_bounds):
		"""Optimal path planning using A* algorithm"""
		# Convert world coordinates to grid indices
		start_idx = ((self.start_point[:2] - min_bounds) / self.resolution).astype(int)
		goal_idx = ((self.end_point[:2] - min_bounds) / self.resolution).astype(int)
		print(start_idx, goal_idx)
		
		# Initialize data structures
		open_set = []
		heapq.heappush(open_set, (0, start_idx[0], start_idx[1]))
		came_from = {}
		g_score = {tuple(start_idx): 0}
		
		directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]        
		while open_set:
			current = heapq.heappop(open_set)
			current_idx = (current[1], current[2])
			if current_idx == tuple(goal_idx):
				return self.reconstruct_path(came_from, current_idx, min_bounds)
				
			for dx, dy in directions:
				neighbor = (current_idx[0]+dx, current_idx[1]+dy)
				
				# Check grid bounds and collisions
				if (0 <= neighbor[0] < grid.shape[0] and 
					0 <= neighbor[1] < grid.shape[1] and 
					not grid[neighbor]):
					
					# Calculate movement cost (diagonal vs straight)
					cost = np.sqrt(dx**2 + dy**2) * self.resolution
					tentative_g = g_score[current_idx] + cost
					
					if (neighbor not in g_score or tentative_g < g_score[neighbor]):
						came_from[neighbor] = current_idx
						g_score[neighbor] = tentative_g
						f_score = tentative_g + self.heuristic(neighbor, goal_idx)
						heapq.heappush(open_set, (f_score, neighbor[0], neighbor[1]))
		
		return []

	def heuristic(self, a, b):
		"""Euclidean distance heuristic"""
		return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) * self.resolution

	def reconstruct_path(self, came_from, current, min_bounds):
		"""Convert grid path back to world coordinates"""
		path = []
		while current in came_from:
			world_x = min_bounds[0] + current[0] * self.resolution
			world_y = min_bounds[1] + current[1] * self.resolution
			path.append(np.array([world_x, world_y, self.start_point[2]]))
			current = came_from[current]
		
		path.append(self.start_point)
		return path[::-1]  # Reverse to start->goal order

	def smooth_path(self, path):
		"""Cubic spline path smoothing"""
		if len(path) < 3: return path
		
		path = np.array(path)
		t = np.linspace(0, 1, len(path))
		return CubicSpline(t, path, bc_type='natural')(np.linspace(0, 1, 100))

	def save_path(self, path):
		"""Save path to txt file with position and orientation"""
		filename = f"{self.planner_name}_path.txt"
		with open(filename, 'w') as f:
			for point in path:
				line = f"{point[0]} {point[1]} {point[2]} 0 0 0 1\n"
				f.write(line)
		print(f"Path saved to {filename}")

	def visualize_path(self, path):
		"""Visualize path based on planner type"""
		plt.figure(figsize=(10, 8))
		
		if self.planner_name == "plan_with_graph":
			self.plot_graph_path(path)
		else:
			self.plot_point_cloud_path(path)
			
		plt.savefig(f"/Rocket_ssd/dataset/data_litevloc/raw_data_out_general/ucl_campus_robot/bag_succeed/planner_path/{self.planner_name}_path.png", dpi=300)
		plt.close()

	def plot_graph_path(self, path):
		"""Visualization for graph-based path"""
		ax = plt.gca()
		path_array = np.array(path)
		
		# Plot path
		ax.plot(path_array[:,0], path_array[:,1], c=PALLETE[0], linewidth=2)
		ax.scatter(path_array[:, 0], path_array[:, 1], c=PALLETE[0], s=50)
		ax.scatter(*path_array[0,:2], s=100, marker='*', c=PALLETE[5], label='Start Point')
		ax.scatter(*path_array[-1,:2], s=100, marker='^', c=PALLETE[5], label='End Point')
		ax.set_title("Graph-based Path Planning")
		self.format_axes()

	def plot_point_cloud_path(self, path):
		"""Visualize 2D occupancy grid map and path"""
		ax = plt.gca()
		
		# Plot occupancy grid map
		if hasattr(self, 'grid') and hasattr(self, 'min_bounds'):
			# Calculate grid boundaries
			x_min, y_min = self.min_bounds[0], self.min_bounds[1]
			x_max = x_min + self.grid.shape[0] * self.resolution
			y_max = y_min + self.grid.shape[1] * self.resolution
			
			# Create grid visualization (transposed for correct orientation)
			ax.imshow(
				self.grid.T,  # Transpose for correct XY orientation
				cmap='gray_r',
				extent=[x_min, x_max, y_min, y_max],
				origin='lower',
				interpolation='none',
				alpha=0.7,
				aspect='auto'
			)
		
		# Plot path
		ax.plot(path[:,0], path[:,1], c=PALLETE[0], linewidth=2, label='Optimal Path')
		ax.scatter(*path[0,:2], s=100, marker='*', c=PALLETE[5], label='Start')
		ax.scatter(*path[-1,:2], s=100, marker='^', c=PALLETE[5], label='Goal')
		
		ax.set_title("Grid-based Path Planning")
		self.format_axes()

	def format_axes(self):
		"""Common axis formatting"""
		ax = plt.gca()
		# ax.set_xlim(self.min_bounds[0]-0.5, self.max_bounds[0]+0.5)
		# ax.set_ylim(self.min_bounds[1]-0.5, self.max_bounds[1]+0.5)
		ax.axis('equal')
		ax.set_xlabel("X (m)")
		ax.set_ylabel("Y (m)")
		ax.grid(True, linestyle='--', alpha=0.7)
		ax.legend()

	def plan(self):
		"""Main planning routine"""
		# Example start and end points (should be parsed from args in real use)
		self.start_point = np.array([-14.79, -5.2891, 0.11])
		self.end_point = np.array([-16.739, -15.524, 0.6154])

		# Create 2D grid bounds
		self.min_bounds = np.min([self.start_point[:2], self.end_point[:2]], axis=0)
		self.min_bounds[0] -= 5.0
		self.min_bounds[1] -= 0.5
		self.max_bounds = np.max([self.start_point[:2], self.end_point[:2]], axis=0)
		self.max_bounds += 0.5
		
		if self.planner_name == "plan_with_graph":
			path = self.graph_planning()
			path = np.array(path)
		else:
			grid = self.create_occupancy_grid()
			raw_path = self.a_star_planning(grid, self.min_bounds)
			path = self.smooth_path(raw_path) if raw_path else np.array([])
			self.grid = grid
		
		if path.any():
			self.save_path(path)
			self.visualize_path(path)
		else:
			print("Path planning failed!")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Optimal Path Planner")
	parser.add_argument("--planner", required=True,
						choices=["plan_with_graph", "plan_with_point"],
						help="Planning method selection")
	parser.add_argument("--map_path", help="Path to graph map data")
	parser.add_argument("--point_cloud", help="Path to point cloud data")
	
	args = parser.parse_args()
	planner = PathPlanner(args)
	planner.plan()
