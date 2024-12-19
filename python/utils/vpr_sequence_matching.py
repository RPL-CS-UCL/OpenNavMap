#! /usr/bin/env python

import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class PlaceRecognitionSeqMatching:
	def __init__(self):
		self.wContrast = 15
		self.enhance = False  # False for learning-based VPR methods

		self.seqLen = 10      # Length for the sequence matching
		self.vMin = 0.25         
		self.vMax = 2.0       # vMax * seqLen. <= db_descriptors.shape[0] - 1
		self.numVel = 20      # Number of velocities to enumerate

		self.matchWindow = 20 # window size for selecting the best score < number of template
	
	def initialize_model(self, db_descriptors, recall_values=5):
		self.db_descriptors = db_descriptors
		self.recall_values = recall_values

	def match(self, query_descriptors, node_id, backward=False):
		"""
			Return:
				recall_preds: list of int, top recall values
				pred: int, best match
				score: float, score of the best match
				backward: bool, True for backward sequence matching
		"""       
		assert query_descriptors.shape[0] == self.seqLen, f"Query descriptors must have length {self.seqLen}"

		D = self._compute_diff_matrix(query_descriptors)
		if self.enhance: D = self._enhance_contrast(D)
		# N: number of database descriptors
		# L: number of query descriptors with sequence length
		self.N, self.L = D.shape

		template_scores, template_velocities = self._score_ref_templates(D)
		recall_preds, pred, score, db_ind_seq_match = self._locate_best_match(template_scores, template_velocities, backward)

		################################
		# ind = np.argmin(template_scores)
		# plt.figure(figsize=(8, 8))
		# plt.imshow(D, cmap='viridis', aspect='auto')
		# x = np.arange(D.shape[1])
		# y = np.floor(np.linspace(ind, ind + template_velocities[ind] * (self.seqLen - 1), self.seqLen)).astype(int)
		# y[y >= self.N] = self.N - 1
		# plt.plot(x, y, 'r')
		# plt.colorbar(label='Difference')
		# plt.xlabel('Query Descriptor Index')
		# plt.ylabel('Database Descriptor Index')
		# plt.title('Difference Matrix')
		# diff_matrix_path = f"/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map4/preds/diff_matrix_euc_{node_id}_{backward}.png"
		# plt.savefig(diff_matrix_path)
		# plt.close()
		################################

		return recall_preds, pred, score, db_ind_seq_match

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
			row_indices[row_indices >= self.N] = self.N - 1
			row_indices = row_indices.reshape(-1)
			# line search indices for the query sequence
			col_indices = np.tile(times, max_ind)
			# evaluate D at indices and sum to get aggregate difference
			Dsum = np.sum(D[row_indices, col_indices].reshape(max_ind, self.L), axis=1)
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
			db_ind_seq_match = np.linspace(iOpt, np.floor(iOpt + iOptV * (self.seqLen - 1)), self.seqLen).astype(int)
			db_ind_seq_match[db_ind_seq_match >= self.N] = self.N - 1		
		else:
			pred = iOpt
			recall_preds = np.argsort(template_scores)[:self.recall_values]
			db_ind_seq_match = np.linspace(iOpt, np.floor(iOpt + iOptV * (self.seqLen - 1)), self.seqLen).astype(int)[::-1]
			db_ind_seq_match[db_ind_seq_match >= self.N] = self.N - 1		

		return recall_preds, pred, mu, db_ind_seq_match

if __name__ == "__main__":
	import os
	import sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
	import argparse
	from image_graph import ImageGraphLoader as GraphLoader
	import pycpptools.src.python.utils_math as pytool_math
	from tqdm import tqdm
	from vpr_single_matching import PlaceRecognitionSingleMatching

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
	# Performance test
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
	single_img_model = PlaceRecognitionSingleMatching()    
	single_img_model.initialize_model(db_descriptors)
	# Perform sequence matching
	preds, scores = [], []
	connected_edges = []
	rmse_list = []
	for node in tqdm(query_map.nodes.values()):
		# if node.id != 32: continue
		if node.id - model.seqLen + 1 >= 0:
			query_descs = query_descriptors[node.id-model.seqLen+1:node.id+1]
			recall_preds, pred, score, db_ind_seq_match = model.match(query_descs, node.id, backward=False)

			##### Check matching result by trajectory aignment
			# from utils_trajectory import align_trajectory, plot_aligned_traj
			# from evo.core.trajectory import PosePath3D
			# traj_ref = PosePath3D(positions_xyz=db_poses[db_ind_seq_match, :3], 
			# 					  orientations_quat_wxyz=np.roll(db_poses[db_ind_seq_match, 3:], 1))
			# traj_est = PosePath3D(positions_xyz=query_poses[node.id-model.seqLen+1:node.id+1, :3], 
			# 					  orientations_quat_wxyz=np.roll(query_poses[node.id-model.seqLen+1:node.id+1, 3:], 1))
			# traj_ref, traj_est_aligned, ape_metric = align_trajectory(traj_ref, traj_est)
			# ape_statistics = ape_metric.get_all_statistics()
			# rmse_list.append(ape_statistics['rmse'])
			# plot_aligned_traj(traj_ref, traj_est_aligned, ape_metric)
			# exit()
			##### 
			preds.append(recall_preds)
			scores.append(score)
			connected_edges.append((db_map.get_node(pred), node))
		else:
			recall_preds, pred, score = single_img_model.match(node.get_descriptor().reshape(1, -1))
			preds.append(recall_preds)
			scores.append(score)
			connected_edges.append((db_map.get_node(pred), node))

	# Doing Ransac
	# Create Trajectory
	from evo.core.trajectory import PosePath3D
	from utils_trajectory import align_trajectory, plot_aligned_traj
	RANSAC_ITERATIONS = 100
	min_error = float('inf')
	best_edges = []
	import random
	for _ in range(RANSAC_ITERATIONS):
		selected_edges = random.sample(connected_edges, max(1, len(connected_edges) // 10))
		db_indices = [edge[0].id for edge in selected_edges]
		query_indices = [edge[1].id for edge in selected_edges]

		traj_ref = PosePath3D(positions_xyz=db_poses[db_indices, :3], 
							  orientations_quat_wxyz=np.roll(db_poses[db_indices, 3:], 1))
		traj_est = PosePath3D(positions_xyz=query_poses[query_indices, :3], 
							  orientations_quat_wxyz=np.roll(query_poses[query_indices, 3:], 1))
		traj_ref, traj_est_aligned, ape_metric = align_trajectory(traj_ref, traj_est)
		ape_statistics = ape_metric.get_all_statistics()
		if ape_statistics['rmse'] < min_error:
			min_error = ape_statistics['rmse']
			best_edges = selected_edges

	print(f"Best RMSE: {min_error}")
	for edge in best_edges:
		print(f"Edge: query {edge[1].id} - DB: {edge[0].id}")

	succ = 0
	for edge in best_edges:
		query_node = edge[0]
		ref_map_node = edge[1]
		dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
			query_node.trans_gt, query_node.quat_gt, ref_map_node.trans_gt, ref_map_node.quat_gt)
		if dis_tsl < 10.0:
			succ += 1
			print(f"Correct prediction: Query {query_node.id} - DB: {ref_map_node.id}")
		else:
			print(f"Wrong prediction: Query {query_node.id} - DB: {ref_map_node.id}")
	print(f"Success Rate: {succ / len(best_edges)}")
	################3

	# succ = 0
	# for i, node in enumerate(query_map.nodes.values()):
	# 	ref_map_node = db_map.get_node(preds[i][0])
	# 	dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
	# 		node.trans_gt, node.quat_gt, ref_map_node.trans_gt, ref_map_node.quat_gt)
	# 	if dis_tsl < 10.0:
	# 		succ += 1
	# 		print(f"Correct prediction: Query {node.id} - DB: {preds[i][0]} - {scores[i]:.3f} - {rmse_list[i]:.3f}")
	# 	else:
	# 		print(f"Wrong prediction: Query {node.id} - DB: {preds[i][0]} - {scores[i]:.3f} - {rmse_list[i]:.3f}")
	# print(f"Success Rate: {succ / len(query_map.nodes)}")

	fig, ax = plt.subplots(figsize=(10, 10))
	for i, node in enumerate(query_map.nodes.values()):
		ref_map_node = db_map.get_node(preds[i][0])
		ax.plot(node.trans_gt[0], node.trans_gt[1], 'ko', markersize=5)
		dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
			node.trans_gt, node.quat_gt, ref_map_node.trans_gt, ref_map_node.quat_gt)
		if dis_tsl < 10.0:
			ax.plot(node.trans_gt[0], node.trans_gt[1], 'ro', markersize=5)
	ax.grid(ls='--', color='0.7')
	plt.axis('equal')
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	plt.title(f"Success Rate: {succ / len(query_map.nodes)}")
	plt.savefig("/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map4/preds/result_place_recognition.png")