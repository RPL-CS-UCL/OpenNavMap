#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import argparse
import numpy as np
import random
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, DBSCAN, AffinityPropagation
from matplotlib import pyplot as plt

import copy
from evo.core.trajectory import PosePath3D
from utils.utils_trajectory import align_trajectory
import pycpptools.src.python.utils_math as pytool_math

class PlaceRecognitionSeqMatching:
	def __init__(self, seqLen=12, enable_ransac=False):
		self.wContrast = 10
		self.enhance = False  # False for learning-based VPR methods

		# NOTE(gogojjh): 
		# seqLen = 20 is robust, but may reject sequences with litter overlap
		# seqLen = 12 is acceptable
		self.seqLen = seqLen   # Length for the sequence matching
		self.vMin = 0.5        
		self.vMax = 2.0       # vMax * seqLen. <= db_descriptors.shape[0] - 1
		self.numVel = 20      # Number of velocities to enumerate

		self.matchWindow = 20 # window size for selecting the best score < number of template

		self.prev_pred = -1

		self.ENABLE_RANSAC = enable_ransac
		self.RANSAC_ITERATIONS = 100
		self.RANSAC_LINE_DIS_THRESHOLD = 3.0
		self.RANSAC_LINE_MIN_ANGLE = np.rad2deg(np.arctan2(0.55, 1)) # 26deg velocity of database is 0.5 times of query
		self.RANSAC_LINE_MAX_ANGLE = np.rad2deg(np.arctan2(3, 1)) # 71deg velocity of database is 3 times of query
		# NOTE(gogojjh):
		# DIFF_MATRIX_SCORE = 1.20 is not a strict threshold
		# DIFF_MATRIX_SCORE = 1.05 is a strict threshold
		self.DIFF_MATRIX_SCORE = 1.2

	def initialize_model(self, db_descriptors, recall_values=5):
		self.db_descriptors = db_descriptors
		self.recall_values = recall_values
		
	def match(self, query_descriptors, backward=False):
		"""
			Return:
				recall_preds: list of int, top recall values
				pred: int, best match
				score: float, score of the best match
				backward: bool, True for backward sequence matching
		"""   
		if query_descriptors.shape[0] < self.seqLen:
			query_desc = query_descriptors[-1, :]
			dists = self._compute_dist_desc(query_desc)
			recall_preds = np.argsort(dists)[:self.recall_values]
			pred = recall_preds[0]
			score = dists[pred]
			self.prev_pred = pred
			return recall_preds, pred, score

		D = self._compute_diff_matrix(query_descriptors)
		if self.enhance: D = self._enhance_contrast(D)

		# Use the last prediction to shorten the search range		
		if self.prev_pred >= 0:
			top_row_idx = max(0, self.prev_pred - int(self.seqLen * 2))
			bottom_row_idx = min(D.shape[0], self.prev_pred + int(self.seqLen * 2))
			D_cut = D[top_row_idx:bottom_row_idx, :]

			# N: number of database descriptors
			# L: number of query descriptors with sequence length
			self.N, self.L = D_cut.shape
			template_scores, template_velocities = self._score_ref_templates(D_cut)
			recall_preds, pred, score = self._locate_best_match(template_scores, template_velocities, backward)

			pred += top_row_idx
			recall_preds = [p + top_row_idx for p in recall_preds]

			# If the new match is not consistent with previous, search the whole difference matrix
			if abs(pred - self.prev_pred) > self.seqLen:
				self.N, self.L = D.shape
				template_scores, template_velocities = self._score_ref_templates(D)
				recall_preds, pred, score = self._locate_best_match(template_scores, template_velocities, backward)

			self.prev_pred = pred
		
		return recall_preds, pred, score

	def ransac_check_match(self, D_all: np.array, db_query_indices: list):
		"""
		Performs RANSAC-based line fitting on a set of connected indices.

		Args:
			db_query_indices (list): List of connected indices representing edges.
				edge[0]: database_node id
				edge[1]: query_ndoe_id

		Returns:
			tuple: A tuple containing the best indices and lines coefficients.

		"""
		best_indices = []
		lines_coeff = []

		# Perform clustering
		x_query_indices = [db_query_indice[1] for db_query_indice in db_query_indices]
		y_db_indices = [db_query_indice[0] for db_query_indice in db_query_indices]
		data = np.column_stack((x_query_indices, y_db_indices))

		if len(x_query_indices) > 50:
			data_cluster = AffinityPropagation()
			labels = data_cluster.fit_predict(data)
		else:
			data_cluster = GaussianMixture(n_components=2, random_state=42)
			labels = data_cluster.fit_predict(data)		

		data = np.array(data)
		labels = np.array(labels)
		num_clusters = np.max(labels) + 1
		print(f"Number of clusters: {num_clusters}")
		
		# Perform RANSAC-based line fitting
		for l in range(num_clusters):
			cur_data = data[labels == l, :]
			best_inliers_error = 10000
			best_inliers_ind = []
			line_coeff = None

			for _ in range(self.RANSAC_ITERATIONS):
				if cur_data.shape[0] < self.seqLen: break
				sample_indices = random.sample(range(cur_data.shape[0]), 2)
				x1, y1 = cur_data[sample_indices[0], :]
				x2, y2 = cur_data[sample_indices[1], :]
				if x1 > x2:
					tmp = x1; x1 = x2; x2 = tmp
					tmp = y1; y1 = y2; y2 = tmp
				if abs(x2 - x1) < 1e-12: continue
				m = (y2 - y1) / (x2 - x1)
				b = y1 - m * x1
				
				distances = np.abs(m * cur_data[:, 0] + b - cur_data[:, 1]) / np.sqrt(m**2 + 1)
				inliers_ind = np.where(distances < self.RANSAC_LINE_DIS_THRESHOLD)[0]
				inliers_count = len(inliers_ind)
				
				if inliers_count < self.seqLen: continue
				if inliers_count < cur_data.shape[0] * 0.4: continue
				if np.rad2deg(np.arctan2(m, 1)) < self.RANSAC_LINE_MIN_ANGLE or \
					np.rad2deg(np.arctan2(m, 1)) > self.RANSAC_LINE_MAX_ANGLE: 
					continue
				if np.sum(distances) < best_inliers_error:
					best_inliers_error = np.sum(distances)
					best_inliers_ind = inliers_ind
					line_coeff = (m, b)

			if line_coeff is not None:
				score = np.mean(D_all[cur_data[best_inliers_ind, 1], cur_data[best_inliers_ind, 0]])
				if score < self.DIFF_MATRIX_SCORE:
					m, b = line_coeff
					print(f"Fitting line angle: {np.rad2deg(np.arctan2(m, 1)):.3f} - Score: {score:.3f}")
					best_indices = best_indices + \
								   [(cur_data[ind, 1], cur_data[ind, 0], score) for ind in best_inliers_ind]
					lines_coeff.append(line_coeff)
		
		return best_indices, lines_coeff, data, labels

	def _compute_dist_desc(self, descriptor) -> np.ndarray:
		##### Option 1: cosine similarity
		# dists = np.sqrt(2 - 2 * np.dot(self.db_descriptors, descriptor.reshape(-1)))
		##### Option 2: euclidean distance
		dists = np.linalg.norm(self.db_descriptors - descriptor, axis=1)
		return dists

	def _compute_diff_matrix(self, query_descriptors) -> np.ndarray:
		"""
			Return:
				D: np.ndarray, num_of_database x sequence_length
				query_descriptors: np.ndarray, sequence_length x descriptor_dim
		"""
		##### Option 1: cosine similarity
		# D = np.sqrt(2 - 2 * np.dot(self.db_descriptors, query_descriptors.transpose()))
		##### Option 2: euclidean distance
		D = np.linalg.norm(query_descriptors[None, :, :] - self.db_descriptors[:, None, :], axis=2)
		return D

	def _enhance_contrast(self, D):
		nref = D.shape[0]
		Denhanced = np.empty_like(D)
		for i in range(nref):
			# reference indices of window around each reference image
			idx_lower = max(i - int(self.wContrast / 2), 0)
			idx_upper = min(i + int(self.wContrast / 2) + 1, nref - 1)
			# local normalization of window given by indices above
			Denhanced[i, :] = (
				D[i, :] - np.mean(D[idx_lower:idx_upper, :], axis=0)
			) / np.std(D[idx_lower:idx_upper, :], axis=0)
		return Denhanced

	def _score_ref_templates(self, D):
		# Add rows on D for exceeding
		rows_to_add = int(self.seqLen * self.vMax)
		add_matrix = np.full((rows_to_add, D.shape[1]), 10)
		D_aug = np.vstack((D, add_matrix))

		# v = vMin, vMin+vStep, ..., vMax
		velocities = np.linspace(self.vMin, self.vMax, self.numVel + 1)
		# t = 0, ..., L
		times = np.arange(self.L)
		# i = 0, ..., max_ind <- truncated so line search not cut off
		max_ind = int(self.N - 1 - self.vMin * self.L)
		# last template image to begin sequence matching on: 0, 1, ..., max_ind - 1
		refs = np.arange(max_ind) 
		# D score for best velocity for each starting point (template image)
		# optD[i]: best score for sequence starting at template i
		optD = np.empty(max_ind); optD[:] = np.inf
		optV = np.empty(max_ind); optV[:] = np.inf
		for vel in velocities:
			# indices in D for line search given a particular velocity
			# include all template number
			row_indices = (
				np.floor(refs[:, np.newaxis] + vel * times[np.newaxis, :])
				.astype(int)
			) # a vector with dimension as max_ind x L
			# remove indices outside of D.shape[0]
			# row_indices[row_indices >= self.N] = self.N - 1
			row_indices = row_indices.reshape(-1)
			# line search indices for the query sequence
			col_indices = np.tile(times, max_ind)
			# evaluate D at indices and sum to get aggregate difference
			Dsum = np.sum(D_aug[row_indices, col_indices].reshape(max_ind, self.L), axis=1)
			# for sequence matching scores better than
			# prior scores (under different velocities), update
			ind_better = Dsum < optD
			optD[ind_better] = Dsum[ind_better]
			optV[ind_better] = vel

		return optD, optV

	def _locate_best_match(self, template_scores, template_velocities, backward=False):
		# indices of best match and window around it
		iOpt = np.argmin(template_scores)
		iOptV = template_velocities[iOpt]
		iWinL = np.maximum(iOpt - int(self.matchWindow / 2), 0)
		iWinU = np.minimum(iOpt + int(self.matchWindow / 2), len(template_scores))
		# check best match outside window
		outside_scores = np.concatenate((template_scores[:iWinL], template_scores[iWinU:]))
		optOutside = min(outside_scores)
		# for negative scores, u \in [0, 1]
		# increases the score... adjust
		if optOutside > 0:
			mu = template_scores[iOpt] / optOutside
		else:
			mu = optOutside / template_scores[iOpt]

		if not backward:
			pred = min(np.floor(iOpt + iOptV * (self.seqLen - 1)).astype(int), self.N - 1)
			indices = np.argsort(template_scores)[:self.recall_values]
			recall_preds = [min(self.N - 1, \
								np.floor(i + template_velocities[i] * (self.seqLen - 1)).astype(int)) \
								for i in indices]
		else:
			pred = iOpt
			recall_preds = np.argsort(template_scores)[:self.recall_values]

		return recall_preds, pred, 1.0 - mu

	def save_diff_matrix_fitting(self, out_dir, connected_indices, best_indices, 
								 D_all, db_map, query_map, lines_coeff, cluster_data, cluster_labels):
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

		im1 = ax1.imshow(D_all, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
		for edge in connected_indices:
			if db_map is not None and query_map is not None:
				db_node = db_map.get_node(edge[0])
				query_node = query_map.get_node(edge[1])
				dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(
					query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt
				)
				if dis_tsl < 20.0:
					ax1.plot(edge[1], edge[0], 'go', markersize=5)
				else:
					ax1.plot(edge[1], edge[0], 'ro', markersize=5)
			else:
				ax1.plot(edge[1], edge[0], 'ro', markersize=5)

		fig.colorbar(im1, ax=ax1)
		ax1.set_xlabel('Query Desc Index')
		ax1.set_ylabel('Database Desc Index')
		ax1.set_title("Diff Matrix [Before RANSAC]")

		im2 = ax2.imshow(D_all, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
		for edge in best_indices:
			if db_map is not None and query_map is not None:
				db_node = db_map.get_node(edge[0])
				query_node = query_map.get_node(edge[1])
				dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(
					query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt
				)
				if dis_tsl < 20.0:
					ax2.plot(edge[1], edge[0], 'go', markersize=5)
				else:
					ax2.plot(edge[1], edge[0], 'ro', markersize=5)
			else:
				ax2.plot(edge[1], edge[0], 'ro', markersize=5)

		for line_coeff in lines_coeff:
			m, b = line_coeff
			x_vals = np.linspace(0, D_all.shape[1], 100)
			y_vals = m * x_vals + b
			ax2.plot(x_vals, y_vals, 'r-', linewidth=1)

		fig.colorbar(im2, ax=ax2)
		ax2.set_xlabel('Query Desc Index')
		ax2.set_ylabel('Database Desc Index')
		ax2.set_title(f"Diff Matrix [After RANSAC]")
		ax2.set_xlim(0, D_all.shape[1])
		ax2.set_ylim(0, D_all.shape[0])
		ax2.invert_yaxis()

		im3 = ax3.imshow(D_all, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
		scatter = ax3.scatter(cluster_data[:, 0], cluster_data[:, 1], c=cluster_labels, cmap='rainbow', s=20)
		fig.colorbar(scatter, ax=ax3)
		ax3.set_xlabel('Query Desc Index')
		ax3.set_ylabel('Database Desc Index')
		ax3.set_title(f"Cluster")
		ax3.set_xlim(0, D_all.shape[1])
		ax3.set_ylim(0, D_all.shape[0])
		ax3.invert_yaxis()

		plt.savefig(f"{out_dir}/difference_matrix_fitting.jpg", dpi=300, bbox_inches='tight')
		plt.close()

if __name__ == "__main__":
	import os
	import sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
	import argparse
	from image_graph import ImageGraphLoader as GraphLoader
	from tqdm import tqdm
	import time

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--db_map_path", type=str, help="Path to the database map file")
	parser.add_argument("--query_map_path", type=str, help="Path to the query map file")
	args = parser.parse_args()

	# Load database and query
	db_map = GraphLoader.load_data(
		args.db_map_path,
		[512, 288],
		depth_scale=0.0,
		load_rgb=True,
		load_depth=False,
		normalized=False
	)
	query_map = GraphLoader.load_data(
		args.query_map_path,
		[512, 288],
		depth_scale=0.0,
		load_rgb=True,
		load_depth=False,
		normalized=False
	)

	# Extract descriptors
	db_descriptors = np.array([node.get_descriptor() for _, node in db_map.nodes.items()], dtype="float32")
	db_poses = np.zeros((db_map.get_num_node(), 7), dtype="float32")
	for indices, (_, node) in enumerate(db_map.nodes.items()):
		db_poses[indices, :3] = node.trans
		db_poses[indices, 3:] = node.quat
	query_descriptors = np.array([node.get_descriptor() for _, node in query_map.nodes.items()], dtype="float32")
	query_poses = np.zeros((query_map.get_num_node(), 7), dtype="float32")
	for indices, (_, node) in enumerate(query_map.nodes.items()):
		query_poses[indices, :3] = node.trans
		query_poses[indices, 3:] = node.quat

	# Create sequence matching model
	model = PlaceRecognitionSeqMatching()
	model.initialize_model(db_descriptors)

	# Perform sequence matching
	connected_indices = []
	start_time = time.time()
	for node in tqdm(query_map.nodes.values()):
		query_descs = query_descriptors[max(0, node.id-model.seqLen+1) : node.id+1]
		recall_preds, pred, score = model.match(query_descs, backward=False)
		connected_indices.append((pred, node.id, score))
	print(f"Sequence Matching Costs: {time.time() - start_time:.3f}s")

	################################################
	D_all = model._compute_diff_matrix(query_descriptors)
	init_indices = connected_indices[:model.seqLen]
	best_indices, lines_coeff, cluster_data, cluster_labels = model.ransac_check_match(D_all, connected_indices[int(model.seqLen/2):])
	best_indices += init_indices
	best_indices = list(dict.fromkeys(best_indices))

	################################################ 
	tp, tn, fp, fn = 0, 0, 0, 0
	for edge in best_indices:
		db_node, query_node = db_map.get_node(edge[0]), query_map.get_node(edge[1])
		dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(
			query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt
		)
		if dis_tsl < 20.0:
			tp += 1
		else:
			fp += 1
	if tp + fp < 1:
		precision = 0
	else:
		precision = tp / (tp+fp)
	print(f"Precision: {precision:.3f} - {tp}/{tp+fp}")
	################################################

	################################################ Visualization
	os.makedirs(f"{args.query_map_path}/preds", exist_ok=True)
	fig, ax = plt.subplots(figsize=(10, 10))
	for node_id, node in db_map.nodes.items():
		ax.plot(node.trans_gt[0], node.trans_gt[1], 'ko', markersize=5)
		for edge in node.edges:
			next_node = edge[0]
			ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=1)
	for node_id, node in query_map.nodes.items():            
		ax.plot(node.trans_gt[0], node.trans_gt[1], 'bo', markersize=5)
		for edge in node.edges:
			next_node = edge[0]
			ax.plot([node.trans_gt[0], next_node.trans_gt[0]], [node.trans_gt[1], next_node.trans_gt[1]], 'k-', linewidth=1)    
	ax.grid(ls='--', color='0.7')
	plt.axis('equal')
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	plt.title(f"Precision: {precision:.3f} - {tp}/{tp+fp}")
	plt.savefig(f"{args.query_map_path}/preds/result_PR.jpg")
	plt.close()
	################################################

	################################################
	model.save_diff_matrix_fitting(\
		f"{args.query_map_path}/preds", 
		connected_indices, best_indices, 
		D_all, db_map, query_map, 
		lines_coeff, cluster_data, cluster_labels)
	################################################

	################################################
	# for edge in best_indices:
	# 	db_node, query_node = db_map.get_node(edge[0]), query_map.get_node(edge[1])
	# 	dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
	# 		query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt)
	# 	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	# 	ax[0].imshow(db_node.rgb_image.permute(1, 2, 0))
	# 	ax[0].set_title("Database")
	# 	ax[0].set_axis_off()
	# 	ax[1].imshow(query_node.rgb_image.permute(1, 2, 0))
	# 	ax[1].set_title("Query")
	# 	ax[1].set_axis_off()
	# 	if dis_tsl < 20.0:
	# 		plt.suptitle(f"Correct Prediction: DB {db_node.id} - Query {query_node.id} - Score {edge[2]:.3f}")
	# 		plt.savefig(f"{args.query_map_path}/preds/db_query_{query_node.id}_correct.jpg")
	# 	else:
	# 		plt.suptitle(f"Wrong Prediction: DB {db_node.id} - Query {query_node.id} - Score {edge[2]:.3f}")		
	# 		plt.savefig(f"{args.query_map_path}/preds/db_query_{query_node.id}_wrong.jpg")
	# 	plt.close()
	################################################
