"""Readers for the text files of one map directory (numpy/scipy only, no torch, no litevloc).

poses.txt rows are world-to-camera; every reader here returns camera-to-world so the frontend
never inverts a pose (spec §5.6).
"""
import re
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

_INT = re.compile(r"(\d+)")


class Poses(NamedTuple):
    names: List[str]
    pos: np.ndarray  # f32[N,3] camera centres in the world frame
    quat: np.ndarray  # f32[N,4] camera-to-world rotation, xyzw
    valid: np.ndarray  # bool[N]; False for all-zero rows (frames without GT)


def frame_index(name: str) -> int:
    """'seq/004869.color.jpg' -> 4869: the last integer in the basename."""
    found = _INT.findall(Path(name).name)
    if not found:
        raise ValueError(f"no frame index in {name!r}")
    return int(found[-1])


def read_poses_c2w(path: Path) -> Poses:
    names: List[str] = []
    rows: List[List[float]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        names.append(parts[0])
        rows.append([float(v) for v in parts[1:8]])
    if not rows:
        return Poses(names, np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros(0, bool))
    arr = np.asarray(rows, dtype=np.float64)
    q_wxyz, t = arr[:, :4], arr[:, 4:7]
    valid = np.linalg.norm(q_wxyz, axis=1) > 1e-9
    q_safe = q_wxyz.copy()
    q_safe[~valid] = (1.0, 0.0, 0.0, 0.0)
    rot_c2w = Rotation.from_quat(q_safe[:, [1, 2, 3, 0]]).inv()  # scipy quaternions are xyzw
    pos = -rot_c2w.apply(t)  # camera centre c = -R^T t
    pos[~valid] = 0.0
    return Poses(names, pos.astype(np.float32), rot_c2w.as_quat().astype(np.float32), valid)


def read_edges(path: Path) -> np.ndarray:
    """[[a, b, weight], ...] as float64 [E,3]; missing/empty file -> (0,3); a missing weight column is 1.0."""
    if not path.is_file() or path.stat().st_size == 0:
        return np.zeros((0, 3))
    arr = np.loadtxt(path, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, 3))
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.ones((arr.shape[0], 1))])
    return arr[:, :3]


def read_timestamps(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = float(parts[1])
    return out


def read_node_ids(path: Path) -> np.ndarray:
    """Frame indices of the first column of a per-frame file (intrinsics.txt = covisibility-graph nodes)."""
    if not path.is_file():
        return np.zeros(0, dtype=np.int64)
    ids = [frame_index(line.split()[0]) for line in path.read_text().splitlines() if line.strip()]
    return np.asarray(ids, dtype=np.int64)


def read_g2o_vertices(path: Path) -> Dict[int, np.ndarray]:
    """VERTEX_SE3:QUAT id x y z qx qy qz qw -> {id: [x y z qx qy qz qw]} (these vertices are camera-to-world)."""
    out: Dict[int, np.ndarray] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0] == "VERTEX_SE3:QUAT":
            out[int(parts[1])] = np.asarray([float(v) for v in parts[2:9]], dtype=np.float64)
    return out


def connected_components(n: int, pairs: np.ndarray) -> np.ndarray:
    """Union-find over `pairs` (int [E,2], endpoints outside [0,n) ignored); labels are ranked by size, 0 = largest."""
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for a, b in np.asarray(pairs).reshape(-1, 2):
        a, b = int(a), int(b)
        if 0 <= a < n and 0 <= b < n:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    roots = np.asarray([find(i) for i in range(n)])
    _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inverse].astype(np.int32)
