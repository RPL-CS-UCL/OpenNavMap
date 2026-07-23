#! /usr/bin/env python
"""Geometry/embedding helpers for the L4 object graph (OpenGoalNav T1.2).

3D IoU and OBB fusion are numpy ports of BOXER's implementation
(`facebookresearch/boxer`, via openintmap) so behavior matches the reference,
without importing boxer (whose `utils` package collides with litevloc's):
- ``iou_exact7`` : analytic yaw-only OBB IoU (Z 1D overlap x XY Sutherland-Hodgman).
- ``weighted_yaw_mean`` / ``align_boxes_r90`` : pi-/90-degree-symmetric yaw+size fusion.
- ``robust_weights`` : confidence x Huber-on-MAD inlier weighting.

Rotation about the ``up_axis`` (default 2 = z, matching BOXER's z-up gravity axis).
"""
import numpy as np

from object_node import OBB


# ---------------------------------------------------------------- 2D polygon ---
def _rect_corners(cx: float, cy: float, w: float, h: float, yaw: float) -> np.ndarray:
    """4 CCW corners of a 2D rotated rectangle."""
    hw, hh = w / 2.0, h / 2.0
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def _seg_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    d1, d2 = p2 - p1, p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return p2
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def _clip_by_edge(poly: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: keep the part of poly on the inward (left) side of a->b."""
    if len(poly) == 0:
        return poly
    normal = np.array([-(b[1] - a[1]), b[0] - a[0]])
    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        cur_in = np.dot(cur - a, normal) >= 0
        nxt_in = np.dot(nxt - a, normal) >= 0
        if cur_in and nxt_in:
            out.append(nxt)
        elif cur_in and not nxt_in:
            out.append(_seg_intersect(cur, nxt, a, b))
        elif not cur_in and nxt_in:
            out.append(_seg_intersect(cur, nxt, a, b))
            out.append(nxt)
    return np.array(out) if out else np.zeros((0, 2))


def _poly_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _poly_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    clipped = poly1
    n2 = len(poly2)
    for i in range(n2):
        if len(clipped) == 0:
            break
        clipped = _clip_by_edge(clipped, poly2[i], poly2[(i + 1) % n2])
    return _poly_area(clipped)


# ------------------------------------------------------------------ OBB IoU ---
def yaw_from_R(rot: np.ndarray, up_axis: int = 2) -> float:
    """Extract the yaw (rotation about ``up_axis``) from a 3x3 rotation matrix."""
    plane = [i for i in range(3) if i != up_axis]
    x_axis = np.asarray(rot, float).reshape(3, 3)[:, 0]
    return float(np.arctan2(x_axis[plane[1]], x_axis[plane[0]]))


def iou_exact7(
    center_a: np.ndarray, size_a: np.ndarray, yaw_a: float,
    center_b: np.ndarray, size_b: np.ndarray, yaw_b: float, up_axis: int = 2,
) -> float:
    """Analytic yaw-only 3D OBB IoU (numpy port of BOXER iou_exact7)."""
    ca, sa = np.asarray(center_a, float).reshape(3), np.asarray(size_a, float).reshape(3)
    cb, sb = np.asarray(center_b, float).reshape(3), np.asarray(size_b, float).reshape(3)
    a0, a1 = [i for i in range(3) if i != up_axis]

    z_overlap = max(
        0.0,
        min(ca[up_axis] + sa[up_axis] / 2, cb[up_axis] + sb[up_axis] / 2)
        - max(ca[up_axis] - sa[up_axis] / 2, cb[up_axis] - sb[up_axis] / 2),
    )
    if z_overlap <= 0:
        return 0.0
    rect_a = _rect_corners(ca[a0], ca[a1], sa[a0], sa[a1], yaw_a)
    rect_b = _rect_corners(cb[a0], cb[a1], sb[a0], sb[a1], yaw_b)
    inter = z_overlap * _poly_intersection_area(rect_a, rect_b)
    vol_a, vol_b = float(np.prod(sa)), float(np.prod(sb))
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 1e-8 else 0.0


def iou_3d(obb_a: OBB, obb_b: OBB, up_axis: int = 2) -> float:
    """Yaw-only 3D IoU of two OBBs (yaw extracted about ``up_axis``)."""
    return iou_exact7(
        obb_a.center, obb_a.size, yaw_from_R(obb_a.R, up_axis),
        obb_b.center, obb_b.size, yaw_from_R(obb_b.R, up_axis), up_axis,
    )


# ------------------------------------------------------------ OBB fusion -------
def weighted_yaw_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean of pi-periodic yaw angles (BOXER port)."""
    phi = 2.0 * np.asarray(angles, float)
    w = np.asarray(weights, float)
    x, y = float(np.sum(w * np.cos(phi))), float(np.sum(w * np.sin(phi)))
    if np.hypot(x, y) < 1e-8:
        return 0.0
    return 0.5 * float(np.arctan2(y, x))


def _angular_distance(a: float, b: float) -> float:
    diff = abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return np.pi - diff if diff > np.pi / 2 else diff


def align_boxes_r90(sizes: np.ndarray, yaws: np.ndarray, weights: np.ndarray, up_axis: int = 2):
    """Resolve 90-degree width/height ambiguity against the weighted-mean yaw."""
    sizes = np.array(sizes, float)
    yaws = np.array(yaws, float)
    a0, a1 = [i for i in range(3) if i != up_axis]
    ref = weighted_yaw_mean(yaws, weights)
    for i in range(len(sizes)):
        d_a = _angular_distance(yaws[i], ref)
        cand = yaws[i] + np.pi / 2 if _angular_distance(yaws[i] + np.pi / 2, ref) < \
            _angular_distance(yaws[i] - np.pi / 2, ref) else yaws[i] - np.pi / 2
        if _angular_distance(cand, ref) < d_a:
            sizes[i, a0], sizes[i, a1] = sizes[i, a1], sizes[i, a0]
            yaws[i] = cand
    return sizes, yaws


def robust_weights(confidences: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Confidence x Huber-on-MAD inlier weights over an observation set (BOXER-style)."""
    conf = np.asarray(confidences, float)
    if len(conf) <= 2:
        return conf
    centers = np.asarray(centers, float)
    resid = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
    mad = np.median(resid) + 1e-6
    scaled = resid / (2.5 * mad)
    huber = np.where(scaled <= 1.0, 1.0, 1.0 / np.maximum(scaled, 1e-6))
    return conf * huber


# -------------------------------------------------------------- embeddings -----
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, float).reshape(-1)
    b = np.asarray(vec_b, float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, float).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v
