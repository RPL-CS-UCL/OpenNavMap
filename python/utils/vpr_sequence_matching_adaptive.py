#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
from utils.vpr_sequence_matching import PlaceRecognitionSeqMatching

class PlaceRecognitionSeqMatchingAdaptive(PlaceRecognitionSeqMatching):
	def __init__(self, seqLen):
		super().__init__(seqLen, enable_ransac=False)

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
