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
