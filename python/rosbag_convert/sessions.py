"""Split one continuous recording into contiguous simulated sessions."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from rosbag_convert.sync import cumulative_path_length

Range = Tuple[int, int]


def _check_ranges(ranges: List[Range], total: int) -> List[Range]:
    if any(end <= start for start, end in ranges):
        raise ValueError(f"Split produced an empty session: {ranges}")
    if ranges[0][0] != 0 or ranges[-1][1] != total:
        raise ValueError(f"Split does not cover all {total} frames: {ranges}")
    return ranges


def split_by_distance(positions: np.ndarray, num_sessions: int) -> List[Range]:
    """Cut the trajectory into ``num_sessions`` parts of equal path length.

    Args:
        positions: (N, 3) positions in time order.
        num_sessions: Number of parts (>= 1).

    Returns:
        ``[start, end)`` index ranges, contiguous and covering every frame.

    Raises:
        ValueError: If a part would be empty (e.g. zero total path length).
    """
    total = len(positions)
    if num_sessions == 1:
        return [(0, total)]
    cum = cumulative_path_length(positions)
    targets = cum[-1] * np.arange(1, num_sessions) / num_sessions
    cuts = [int(np.searchsorted(cum, t)) for t in targets]
    bounds = [0] + cuts + [total]
    return _check_ranges(list(zip(bounds[:-1], bounds[1:])), total)


def split_by_time(stamps: np.ndarray, boundaries_rel_s: List[float]) -> List[Range]:
    """Cut at explicit times measured from the first stamp."""
    total = len(stamps)
    rel = np.asarray(stamps, dtype=np.float64) - stamps[0]
    cuts = [int(np.searchsorted(rel, b)) for b in boundaries_rel_s]
    bounds = [0] + cuts + [total]
    return _check_ranges(list(zip(bounds[:-1], bounds[1:])), total)
