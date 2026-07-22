#! /usr/bin/env python
"""Geometry/embedding helpers for the L4 object graph (OpenGoalNav T1.2).

Kept in the main fork (not litevloc) per the fork discipline. 3D IoU uses an
axis-aligned approximation of the two OBBs (rotation is retained on the node but
not used for the overlap test); this is sufficient for the association double
threshold and is documented as an approximation.
"""
import numpy as np

from object_node import OBB


def aabb_from_obb(obb: OBB) -> tuple:
    """Return (min_corner, max_corner) of the axis-aligned box around an OBB.

    The 8 OBB corners are rotated into world frame and bounded, giving a loose
    AABB that is orientation-aware without a full OBB intersection.
    """
    half = np.asarray(obb.size, float).reshape(3) / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = (obb.R @ (signs * half).T).T + np.asarray(obb.center, float).reshape(3)
    return corners.min(axis=0), corners.max(axis=0)


def iou_3d(obb_a: OBB, obb_b: OBB) -> float:
    """Approximate 3D IoU of two OBBs via their world-axis-aligned bounds."""
    min_a, max_a = aabb_from_obb(obb_a)
    min_b, max_b = aabb_from_obb(obb_b)
    inter_dims = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
    if np.any(inter_dims <= 0):
        return 0.0
    inter = float(np.prod(inter_dims))
    vol_a = float(np.prod(max_a - min_a))
    vol_b = float(np.prod(max_b - min_b))
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity of two vectors; 0 if either is zero-length."""
    a = np.asarray(vec_a, float).reshape(-1)
    b = np.asarray(vec_b, float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def normalize(vec: np.ndarray) -> np.ndarray:
    """Return the unit vector (unchanged if zero-length)."""
    v = np.asarray(vec, float).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v
