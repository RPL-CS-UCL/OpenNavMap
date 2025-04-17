#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
from utils.vpr_sequence_matching import PlaceRecognitionSeqMatching

class PlaceRecognitionSeqMatchingAdaptive(PlaceRecognitionSeqMatching):
	def __init__(self, seqLen, enable_ransac=False):
		super().__init__(seqLen, enable_ransac)

		# Base sequence parameters
		self.max_seq_len = seqLen  # Maximum sequence length to try
		self.min_seq_len = 4       # Minimum sequence length (adjust based on dataset)
		self.len_step = 2          # Step size for length reduction
		self.lambda_len = 0.1      # Weight for length vs cost tradeoff

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
		# check whether the matchWindow larger than the length of template_scores 
		if np.any(outside_scores):
			optOutside = min(outside_scores)
			# for negative scores, u \in [0, 1]
			# increases the score... adjust
			if optOutside > 0:
				mu = template_scores[iOpt] / optOutside
			else:
				mu = optOutside / template_scores[iOpt]
		else:
			mu = template_scores[iOpt]

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

