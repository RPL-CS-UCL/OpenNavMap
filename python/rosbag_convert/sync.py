"""Time-stamp association and trajectory length helpers."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def associate_by_stamp(query_stamps: np.ndarray, ref_stamps: np.ndarray,
                       tolerance: float) -> List[Tuple[int, int]]:
    """Pair each query stamp with its nearest reference stamp within ``tolerance``.

    Args:
        query_stamps: Stamps to look up (any order).
        ref_stamps: Reference stamps, sorted ascending.
        tolerance: Max absolute difference (same unit as the stamps).

    Returns:
        ``(query_index, ref_index)`` pairs in query order; queries with no
        reference inside the tolerance are omitted.
    """
    ref = np.asarray(ref_stamps, dtype=np.float64)
    if np.any(np.diff(ref) < 0):
        raise ValueError("ref_stamps must be sorted ascending")
    pairs: List[Tuple[int, int]] = []
    for qi, q in enumerate(np.asarray(query_stamps, dtype=np.float64)):
        hi = int(np.searchsorted(ref, q))
        candidates = [i for i in (hi - 1, hi) if 0 <= i < len(ref)]
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(ref[i] - q))
        if abs(ref[best] - q) <= tolerance:
            pairs.append((qi, best))
    return pairs


def cumulative_path_length(positions: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean path length along ``positions`` (N, 3); first entry is 0."""
    pos = np.asarray(positions, dtype=np.float64)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])
