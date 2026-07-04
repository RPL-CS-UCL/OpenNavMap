from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
import numpy as np


@dataclass
class PoseEntry:
    img_name: str
    quat_xyzw: list[float]  # [qx, qy, qz, qw]
    position: list[float]   # [tx, ty, tz]


@dataclass
class EdgeEntry:
    src: int
    dst: int
    weight: float


@dataclass
class IntrinsicsEntry:
    K: list[list[float]]  # 3x3
    img_size: list[int]   # [w, h]


def detect_merge_dirs(results_dir: Path) -> list[Path]:
    """Detect merge_* subdirectories in results_dir, sorted by merge order.

    Sorting key: number of underscore-separated parts (merge_0=1, merge_0_1=2, ...).
    Files (like merge_finalmap) are excluded.
    """
    candidates = [
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("merge_")
    ]
    return sorted(candidates, key=lambda d: d.name.count("_"))


def load_poses(poses_file: Path) -> list[PoseEntry]:
    """Parse poses.txt: 'img_name qx qy qz qw tx ty tz' per line."""
    result: list[PoseEntry] = []
    for line in poses_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        img_name = parts[0]
        qx, qy, qz, qw = (float(x) for x in parts[1:5])
        tx, ty, tz = (float(x) for x in parts[5:8])
        result.append(PoseEntry(img_name, [qx, qy, qz, qw], [tx, ty, tz]))
    return result


def load_edges(edges_file: Path) -> list[EdgeEntry]:
    """Parse edges_*.txt: 'src dst weight' per line."""
    result: list[EdgeEntry] = []
    for line in edges_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        result.append(EdgeEntry(int(parts[0]), int(parts[1]), float(parts[2])))
    return result


def load_intrinsics(intrinsics_file: Path) -> dict[str, IntrinsicsEntry]:
    """Parse intrinsics.txt: 'img_name fx fy cx cy w h' per line."""
    result: dict[str, IntrinsicsEntry] = {}
    for line in intrinsics_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        img_name = parts[0]
        fx, fy, cx, cy = (float(x) for x in parts[1:5])
        w, h = int(parts[5]), int(parts[6])
        K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        result[img_name] = IntrinsicsEntry(K, [w, h])
    return result


def load_descriptors(desc_file: Path) -> dict[str, np.ndarray]:
    """Parse database_descriptors.txt: 'img_name d1 d2 ... d256' per line."""
    result: dict[str, np.ndarray] = {}
    for line in desc_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_name = parts[0]
        desc = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        result[img_name] = desc
    return result


def identify_new_nodes(
    prev_img_names: list[str], curr_img_names: list[str]
) -> list[int]:
    """Return indices of curr_img_names whose image name is not in prev_img_names."""
    prev_set = set(prev_img_names)
    return [i for i, name in enumerate(curr_img_names) if name not in prev_set]


def compute_dmatrix(
    ref_descs: dict[str, np.ndarray], query_descs: dict[str, np.ndarray]
) -> np.ndarray:
    """Compute cosine similarity matrix between reference and query descriptors.

    Returns matrix of shape (len(ref), len(query)).
    """
    ref_keys = list(ref_descs.keys())
    query_keys = list(query_descs.keys())
    if not ref_keys or not query_keys:
        return np.zeros((len(ref_keys), len(query_keys)), dtype=np.float32)

    ref_mat = np.stack([ref_descs[k] for k in ref_keys])
    query_mat = np.stack([query_descs[k] for k in query_keys])

    ref_norm = ref_mat / (np.linalg.norm(ref_mat, axis=1, keepdims=True) + 1e-8)
    query_norm = query_mat / (np.linalg.norm(query_mat, axis=1, keepdims=True) + 1e-8)

    return ref_norm @ query_norm.T


def get_new_descriptors(
    prev_descs: dict[str, np.ndarray], curr_descs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return descriptors in curr_descs whose key is not in prev_descs."""
    prev_keys = set(prev_descs.keys())
    return {k: v for k, v in curr_descs.items() if k not in prev_keys}
