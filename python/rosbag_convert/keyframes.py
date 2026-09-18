"""Distance/angle keyframe selection, matching the released-dataset rule."""
from __future__ import annotations

from typing import List

import numpy as np

from utils.utils_geom import compute_pose_error


def select_keyframes(poses: np.ndarray, trans_thresh_m: float, rot_thresh_deg: float) -> List[int]:
    """Greedy keyframe selection along a trajectory.

    The first pose is always kept. A pose is kept when, relative to the
    last kept pose, the robot moved more than ``trans_thresh_m`` OR turned
    more than ``rot_thresh_deg``. Consecutive keyframes therefore always
    differ by at least one of the two thresholds.

    Args:
        poses: (N, 4, 4) camera-to-world (or body-to-world) poses in time order.
        trans_thresh_m: Translation threshold in metres.
        rot_thresh_deg: Rotation threshold in degrees.

    Returns:
        Indices into ``poses`` of the selected keyframes, ascending.
    """
    if len(poses) == 0:
        return []
    kept = [0]
    for i in range(1, len(poses)):
        trans_err, rot_err = compute_pose_error(poses[kept[-1]], poses[i], mode="matrix")
        if trans_err > trans_thresh_m or rot_err > rot_thresh_deg:
            kept.append(i)
    return kept
