#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import argparse
import numpy as np
import random
from matplotlib import pyplot as plt

import numpy as np

class PlaceRecognitionSeqMatchingAdaptive:
	def __init__(self, seqLen, enable_ransac=False):
		# Base sequence parameters
		self.seqLen = seqLen
		self.max_seq_len = seqLen  # Maximum sequence length to try
		self.min_seq_len = 3       # Minimum sequence length (adjust based on dataset)
		self.len_step = 2          # Step size for length reduction
		self.lambda_len = 0.1      # Weight for length vs cost tradeoff
		self.matchWindow = 10

		# Velocity parameters (expanded range)
		self.vMin = 0.4                
		self.vMax = 2.5
		self.numVel = 20
		
		# Original parameters remain
		self.wContrast = 10
		self.enhance = False

		self.MAX_DIST = 2.0
		self.ENABLE_RANSAC = enable_ransac

	def initialize_model(self, db_descriptors, recall_values=5):
		self.db_descriptors = db_descriptors
		self.recall_values = recall_values

	def match(self, query_descriptors, backward=False):
		"""Main entry point for sequence matching"""
		if query_descriptors.shape[0] < self.max_seq_len:
			return self._fallback_match(query_descriptors)

		# Precompute integral image for fast window sum calculation
		D_all = self.compute_diff_matrix(query_descriptors)
		if self.enhance:
			D_all = self._enhance_contrast(D)

		self.N, self.L = D_all.shape
		best_score, best_result = 0.0, None

		# Multi-length search from longest to shortest
		for seq_len in range(self.max_seq_len, self.min_seq_len-1, -self.len_step):
			D = D_all[:, -seq_len:]
			# Calculate scores for current sequence length
			template_scores, template_velocities = \
				self._score_ref_templates(D, seq_len)
			current_preds, current_pred, current_mu = \
				self._locate_best_match(template_scores, template_velocities, seq_len, backward)
			
			# Combined score considering both matching quality and sequence length
			combined_score = (self.MAX_DIST - current_mu) - self.lambda_len * (1.0 / seq_len)
			if combined_score > best_score:
				best_score = combined_score
				best_result = (current_preds, current_pred, self.MAX_DIST - current_mu, seq_len)

		return best_result[:3] if best_result[:3] else self._fallback_match(query_descriptors)

	def compute_diff_matrix(self, query_descriptors) -> np.ndarray:
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

	def _fallback_match(self, query_descriptors):
		"""Handle short query sequences"""
		query_desc = query_descriptors[-1, :]
		dists = self._compute_dist_desc(query_desc)
		recall_preds = np.argsort(dists)[:self.recall_values]
		pred = recall_preds[0]
		score = self.MAX_DIST - dists[pred]
		return recall_preds, pred, score
	
	def _compute_dist_desc(self, descriptor) -> np.ndarray:
		##### Option 1: cosine similarity
		# dists = np.sqrt(2 - 2 * np.dot(self.db_descriptors, descriptor.reshape(-1)))
		##### Option 2: euclidean distance
		dists = np.linalg.norm(self.db_descriptors - descriptor, axis=1)
		return dists

	def _score_ref_templates(self, D, seq_len):
		# v = vMin, vMin+vStep, ..., vMax
		velocities = np.linspace(self.vMin, self.vMax, self.numVel + 1)
		# i = 0, ..., max_ind <- truncated so line search not cut off
		max_ind = int(self.N - 1 - self.vMax * seq_len)
		# last template image to begin sequence matching on: 0, 1, ..., max_ind - 1
		refs = np.arange(max_ind) 
		# D score for best velocity for each starting point (template image)
		# optD[i]: best score for sequence starting at template i
		optD = np.empty(max_ind); optD[:] = np.inf
		optV = np.empty(max_ind); optV[:] = np.inf

		# t = 0, ..., L
		times = np.arange(seq_len)
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
			Dsum = np.sum(D[row_indices, col_indices].reshape(max_ind, seq_len), axis=1)
			# for sequence matching scores better than
			# prior scores (under different velocities), update
			ind_better = Dsum < optD
			optD[ind_better] = Dsum[ind_better]
			optV[ind_better] = vel

		return optD, optV

	def _locate_best_match(self, template_scores, template_velocities, seq_len, backward=False):
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
			pred = min(np.floor(iOpt + iOptV * (seq_len - 1)).astype(int), self.N - 1)
			indices = np.argsort(template_scores)[:self.recall_values]
			recall_preds = [min(self.N - 1, \
								np.floor(i + template_velocities[i] * (seq_len - 1)).astype(int)) \
								for i in indices]
		else:
			pred = iOpt
			recall_preds = np.argsort(template_scores)[:self.recall_values]

		return recall_preds, pred, mu

	def _calculate_confidence(self, template_scores, iOpt):
		"""Calculate matching confidence score"""
		iWinL = max(iOpt - self.matchWindow//2, 0)
		iWinU = min(iOpt + self.matchWindow//2, len(template_scores))
		outside_scores = np.concatenate((template_scores[:iWinL], template_scores[iWinU:]))
		return template_scores[iOpt] / np.min(outside_scores) if np.min(outside_scores) > 0 else 0.0

	def save_diff_matrix_fitting(self, out_dir, connected_indices, best_indices, 
								 D_all, db_map, query_map, 
								 lines_coeff=None, 
								 cluster_data=None, 
								 cluster_labels=None):
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

		im1 = ax1.imshow(D_all, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
		for edge in connected_indices:
			if db_map is not None and query_map is not None:
				db_node = db_map.get_node(edge[0])
				query_node = query_map.get_node(edge[1])
				dis_tsl, _ = query_node.compute_distance(db_node)
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

		if lines_coeff is not None:
			im2 = ax2.imshow(D_all, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
			for edge in best_indices:
				if db_map is not None and query_map is not None:
					db_node = db_map.get_node(edge[0])
					query_node = query_map.get_node(edge[1])
					dis_tsl, _ = query_node.compute_distance(db_node)
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

		if cluster_data is not None:
			im3 = ax3.imshow(D_all, cmap='viridis', aspect='auto', vmin=0, vmax=2.0)
			scatter = ax3.scatter(cluster_data[:, 0], cluster_data[:, 1], c=cluster_labels, cmap='rainbow', s=20)
			fig.colorbar(scatter, ax=ax3)
			ax3.set_xlabel('Query Desc Index')
			ax3.set_ylabel('Database Desc Index')
			ax3.set_title(f"Cluster")
			ax3.set_xlim(0, D_all.shape[1])
			ax3.set_ylim(0, D_all.shape[0])
			ax3.invert_yaxis()

		plt.savefig(f"{out_dir}/difference_matrix_fitting_{self.seqLen}.jpg", dpi=300, bbox_inches='tight')
		plt.close() 