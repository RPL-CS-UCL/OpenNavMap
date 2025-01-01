#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import argparse
import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from evo.core.trajectory import PosePath3D
from utils.utils_trajectory import align_trajectory

class PlaceRecognitionSeqMatching:
	def __init__(self):
		self.RANSAC_ITERATIONS = 1000

		self.wContrast = 10
		self.enhance = False  # False for learning-based VPR methods

		self.seqLen = 20      # Length for the sequence matching
		self.vMin = 0.4        
		self.vMax = 2.0       # vMax * seqLen. <= db_descriptors.shape[0] - 1
		self.numVel = 20      # Number of velocities to enumerate

		self.matchWindow = 20 # window size for selecting the best score < number of template

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
			return recall_preds, pred, score

		D = self._compute_diff_matrix(query_descriptors)
		if self.enhance: D = self._enhance_contrast(D)
		# N: number of database descriptors
		# L: number of query descriptors with sequence length
		self.N, self.L = D.shape

		template_scores, template_velocities = self._score_ref_templates(D)
		recall_preds, pred, score, db_ind_seq_match = self._locate_best_match(template_scores, template_velocities, backward)

		################################
		# DEBUG(gogojjh):
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
		# diff_matrix_path = f"/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus/s00000/out_map8/preds/diff_matrix_euc_{node_id}_{backward}.png"
		# plt.savefig(diff_matrix_path)
		# plt.close()
		################################

		return recall_preds, pred, score


	def ransac_check_match(self, db_poses, query_poses, connected_indices):
		best_min_rmse, best_indices, best_align_R_t_s = float('inf'), [], None
		for _ in range(self.RANSAC_ITERATIONS):
			# sampled_indices = random.sample(connected_indices, min(max(5, len(connected_indices) // 10), len(connected_indices)))
			sampled_indices = random.sample(connected_indices, 10)
			db_indices = [edge[0] for edge in sampled_indices]
			query_indices = [edge[1] for edge in sampled_indices]
			traj_ref = PosePath3D(positions_xyz=db_poses[db_indices, :3],
								  orientations_quat_wxyz=np.roll(db_poses[db_indices, 3:], -1))
			traj_est = PosePath3D(positions_xyz=query_poses[query_indices, :3],
								  orientations_quat_wxyz=np.roll(query_poses[query_indices, 3:], -1))
			try:
				traj_ref, traj_est_aligned, ape_metric, align_R_t_s = align_trajectory(traj_ref, traj_est)
				rmse = ape_metric.get_all_statistics()['rmse']
				if rmse < best_min_rmse:
					best_min_rmse = rmse
					best_indices = sampled_indices
					best_align_R_t_s = align_R_t_s
			except Exception as e: 
				pass		
		return best_min_rmse, best_indices, best_align_R_t_s
	
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

	# RANSAC-based reliable edges extraction
	# RMSE_THRESHOLD = 7.5
	# best_min_rmse, best_indices, best_align_R_t_s = None, None, None
	# for k in range(1):
	# 	best_min_rmse, best_indices, best_align_R_t_s = \
	# 		model.ransac_check_match(db_poses, query_poses, connected_indices)
	# 	print(f"Error: {best_min_rmse:.3f} - Candidates Size: {len(connected_indices)} - Best Indices Size: {len(best_indices)}")
	# 	if best_min_rmse <= RMSE_THRESHOLD: break
	# 	best_min_rmse, best_indices, best_align_R_t_s = None, None, None
	# if best_min_rmse is None: exit()
	
	# edge_str = {f"{edge[0]}_{edge[1]}" for edge in best_indices}
	# augment_indices = random.sample(connected_indices, min(max(10, len(connected_indices)), len(connected_indices)))
	# R, t, s = best_align_R_t_s[0], best_align_R_t_s[1], best_align_R_t_s[2]
	# for edge in augment_indices:
	# 	if f"{edge[0]}_{edge[1]}" in edge_str: continue
	# 	db_node, query_node = db_map.get_node(edge[0]), query_map.get_node(edge[1])
	# 	dis = np.linalg.norm(R @ query_node.trans + t - db_node.trans)
	# 	if dis >= RMSE_THRESHOLD: continue
	# 	best_indices.append(edge)
	# 	edge_str.add(f"{edge[0]}_{edge[1]}")

	# print(f"Error: {best_min_rmse:.3f} - Candidates Size: {len(connected_indices)} - Best Indices Size (after aug): {len(best_indices)}")
	# print(f"All edges: {len(connected_indices)} - Augmented edges: {len(best_indices)} - Best edges: {len(best_indices)}")

	best_indices = connected_indices
	edge_str =  {f"{edge[0]}_{edge[1]}" for edge in connected_indices}
	print(f"Candidates Size: {len(connected_indices)} - Best Indices Size (after aug): {len(best_indices)}")

	tp, tn, fp, fn = 0, 0, 0, 0
	for edge in best_indices:
		db_node, query_node = db_map.get_node(edge[0]), query_map.get_node(edge[1])
		dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
			query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt)
		if dis_tsl < 7.5:
			tp += 1
			# print(f"Correct prediction: Query {query_node.id} - DB: {db_node.id} - Score: {edge[2]:.3f}")
		else:
			# print(f"Wrong prediction: Query {query_node.id} - DB: {db_node.id} - Score: {edge[2]:.3f}")
			fp += 1
	if tp + fp < 1:
		precision = 0
	else:
		precision = tp / (tp+fp)
	print(f"Precision: {precision:.3f} - {tp}/{tp+fp}")

	################################################
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
	for edge in best_indices:
		db_node, query_node = db_map.get_node(edge[0]), query_map.get_node(edge[1])
		dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
			query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt)
		if dis_tsl < 7.5:
			ax.plot([query_node.trans_gt[0], db_node.trans_gt[0]], [query_node.trans_gt[1], db_node.trans_gt[1]], 'g-', linewidth=4)
		else:
			ax.plot([query_node.trans_gt[0], db_node.trans_gt[0]], [query_node.trans_gt[1], db_node.trans_gt[1]], 'r-', linewidth=4)
	ax.grid(ls='--', color='0.7')
	plt.axis('equal')
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	plt.title(f"Precision: {precision:.3f} - {tp}/{tp+fp}")
	plt.savefig(f"{args.query_map_path}/preds/result_PR.jpg")
	plt.close()

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
	# 	if dis_tsl < 7.5:
	# 		plt.suptitle(f"Correct Prediction: DB {db_node.id} - Query {query_node.id} - Score {edge[2]:.3f}")
	# 		plt.savefig(f"{args.query_map_path}/preds/db_query_{query_node.id}_correct.jpg")
	# 	else:
	# 		plt.suptitle(f"Wrong Prediction: DB {db_node.id} - Query {query_node.id} - Score {edge[2]:.3f}")		
	# 		plt.savefig(f"{args.query_map_path}/preds/db_query_{query_node.id}_wrong.jpg")
	# 	plt.close()

	################################################
	D = model._compute_diff_matrix(query_descriptors)
	plt.imshow(D, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
	for edge in best_indices:
		db_node, query_node = db_map.get_node(edge[0]), query_map.get_node(edge[1])
		dis_tsl, dis_angle = pytool_math.tools_eigen.compute_relative_dis(\
			query_node.trans_gt, query_node.quat_gt, db_node.trans_gt, db_node.quat_gt)
		if dis_tsl < 7.5:
			plt.plot(edge[1], edge[0], 'go', markersize=5)
		else:
			plt.plot(edge[1], edge[0], 'ro', markersize=5)
	plt.colorbar(label='Difference')
	plt.xlabel('Query Descriptor Index')
	plt.ylabel('Database Descriptor Index')
	plt.title(f"Difference Matrix - Precision: {precision:.3f} - {tp}/{tp+fp}")
	plt.savefig(f"{args.query_map_path}/preds/difference_matrix.jpg")
	plt.close()

	###############################
	from sklearn.cluster import KMeans
	from sklearn.cluster import DBSCAN, HDBSCAN
	from sklearn.cluster import SpectralClustering
	from sklearn.mixture import GaussianMixture
	N_CLUSTER = 4
	x = [edge[1] for edge in best_indices]
	y = [edge[0] for edge in best_indices]
	data = np.column_stack((x, y))
	# kmeans = KMeans(n_clusters=N_CLUSTER, random_state=42)
	# labels = kmeans.fit_predict(data)
	# dbscan = DBSCAN(eps=3.0, min_samples=5)
	# labels = dbscan.fit_predict(data)
	# data = np.column_stack((x, y))
	gmm = GaussianMixture(n_components=N_CLUSTER, random_state=42)
	labels = gmm.fit_predict(data)
	# sc = SpectralClustering(n_clusters=N_CLUSTER)
	# labels = sc.fit_predict(data)
	# hdbscan = HDBSCAN()
	# labels = hdbscan.fit_predict(data)	
	plt.figure()
	plt.imshow(D, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
	scatter = plt.scatter(x, y, c=labels, cmap='rainbow', s=20)
	plt.colorbar(scatter, label='Cluster Label')
	plt.xlabel("Query Descriptor Index")
	plt.ylabel("Database Descriptor Index")
	plt.title(f"Clustering")
	plt.grid(True)
	plt.savefig(f"{args.query_map_path}/preds/gaussian_mixture.jpg", dpi=300)
	plt.close()
	
	###############################
	plt.imshow(D, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)

	seqLen = 15
	RANSAC_ITERATIONS = 10
	distance_threshold = 3.0

	data = np.array(data)
	labels = np.array(labels)
	for l in range(N_CLUSTER):
		cur_data = data[labels == l, :]
		for _ in range(1):
			best_inliers_error = 10000
			best_inliers_count = 0
			best_inliers = []
			best_line = (0, 0)
			for _ in range(RANSAC_ITERATIONS):
				if cur_data.shape[0] < 2: break			

				sample_indices = random.sample(range(cur_data.shape[0]), 2)
				x1, y1 = cur_data[sample_indices[0], :]
				x2, y2 = cur_data[sample_indices[1], :]
				if abs(x2 - x1) < 1e-12: continue
				m = (y2 - y1) / (x2 - x1)
				b = y1 - m * x1				
				
				distances = np.abs(m * cur_data[:, 0] + b - cur_data[:, 1]) / np.sqrt(m**2 + 1)
				inliers_idx = np.where(distances < distance_threshold)[0]
				inliers_count = len(inliers_idx)
				
				if inliers_count < seqLen: continue
				# if inliers_count < cur_data.shape[0] * 0.5: continue
				if np.rad2deg(np.arctan2(m, 1)) < 30 or np.rad2deg(np.arctan2(m, 1)) > 70: continue

				if np.sum(distances) < best_inliers_error:
					best_inliers_error = np.sum(distances)
					best_inliers_count = inliers_count
					best_inliers = inliers_idx
					best_line = (m, b)

			if best_inliers_count > seqLen:			
				score = np.sum(D[cur_data[best_inliers, 1], cur_data[best_inliers, 0]]) / best_inliers_count
				print(f"Score: {score:.3f}")
				if score < 1.20:
					print(np.rad2deg(np.arctan2(m, 1)))
					m, b = best_line
					x_vals = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
					y_vals = m * x_vals + b
					print(f"{np.rad2deg(np.arctan2(m, 1)):.3f}")
					plt.plot(cur_data[best_inliers, 0], cur_data[best_inliers, 1], 'ro', markersize=5)
					plt.plot(x_vals, y_vals, 'm-', linewidth=1)
					np.delete(cur_data, best_inliers, axis=0)


	plt.xlim(0, D.shape[1])
	plt.ylim(0, D.shape[0])
	plt.gca().invert_yaxis()	
	plt.xlabel("Query Descriptor Index")
	plt.ylabel("Database Descriptor Index")
	plt.title(f"Loops")
	plt.grid(True)
	plt.savefig(f"{args.query_map_path}/preds/difference_matrix_fitting.jpg", dpi=300)
	plt.close()	
	###############################