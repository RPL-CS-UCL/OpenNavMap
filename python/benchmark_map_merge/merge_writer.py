"""Writes merged map output in OpenNavMap format, including submap_disc_0
layout, symlinks, and format conversion utilities."""

import os
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def read_poses(file_path: str) -> Dict[str, np.ndarray]:
    """Read mapfree-format poses. Returns {img_name: array([qw,qx,qy,qz,tx,ty,tz])}."""
    result = {}
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            img_name = parts[0]
            result[img_name] = np.array([float(x) for x in parts[1:8]], dtype=np.float64)
    return result


def read_timestamps(file_path: str) -> Dict[str, float]:
    result = {}
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            result[parts[0]] = float(parts[1])
    return result


def _vec_to_matrix(trans: np.ndarray, quat: np.ndarray, mode: str = "wxyz") -> np.ndarray:
    from scipy.spatial.transform import Rotation
    if mode == "wxyz":
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = trans
    return T


def _matrix_to_vec(T: np.ndarray, mode: str = "wxyz") -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation
    R = T[:3, :3]
    t = T[:3, 3]
    q_xyzw = Rotation.from_matrix(R).as_quat()
    if mode == "wxyz":
        q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    elif mode == "xyzw":
        q = q_xyzw
    else:
        raise ValueError(f"Unknown quat mode: {mode}")
    return t, q


def estimate_umeyama(
    est_poses: List[np.ndarray],
    vio_poses: List[np.ndarray],
) -> np.ndarray:
    """Estimate similarity transform from vio frame to estimated frame using Umeyama.

    est_poses: 4x4 camera-to-world matrices from HLoc PnP (in reference frame)
    vio_poses: 4x4 camera-to-world matrices from VIO (in incoming submap frame)
    Returns 4x4 transform T_ref_incoming.
    """
    from numpy.linalg import svd
    est_pts = np.array([p[:3, 3] for p in est_poses])
    vio_pts = np.array([p[:3, 3] for p in vio_poses])

    n = est_pts.shape[0]
    e_mean = est_pts.mean(axis=0)
    v_mean = vio_pts.mean(axis=0)
    est_centered = est_pts - e_mean
    vio_centered = vio_pts - v_mean

    C = est_centered.T @ vio_centered / n
    U, _, Vt = svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = np.trace(C.T @ R) / np.trace(vio_centered.T @ vio_centered)
    t = e_mean - s * R @ v_mean

    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def apply_transform(pose_dict: Dict[str, np.ndarray], T: np.ndarray) -> Dict[str, np.ndarray]:
    """Apply 4x4 transform to all poses in dict. Poses are camera-to-world."""
    result = {}
    for img, pose_vec in pose_dict.items():
        quat = pose_vec[0:4]
        trans = pose_vec[4:7]
        cam_to_world = _vec_to_matrix(trans, quat, mode="wxyz")
        cam_to_world_new = T @ cam_to_world
        t_new, q_new = _matrix_to_vec(cam_to_world_new, mode="wxyz")
        result[img] = np.concatenate([q_new, t_new])
    return result


def merge_poses(
    merged: Dict[str, np.ndarray],
    incoming: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    result = dict(merged)
    result.update(incoming)
    return result


def write_summary_json(summary: dict, path: Path):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"summary written to {path}")


def write_poses_txt(pose_dict: Dict[str, np.ndarray], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for img in sorted(pose_dict.keys()):
            vec = pose_dict[img]
            f.write(f"{img} {vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f} "
                    f"{vec[3]:.6f} {vec[4]:.6f} {vec[5]:.6f} {vec[6]:.6f}\n")


def write_timestamps_txt(timestamps: Dict[str, float], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for img in sorted(timestamps.keys()):
            f.write(f"{img} {timestamps[img]:.9f}\n")


def create_merge_dir(
    result_root: Path,
    merge_name: str,
) -> Path:
    merge_dir = result_root / merge_name / "submap_disc_0"
    merge_dir.mkdir(parents=True, exist_ok=True)
    return merge_dir


def create_finalmap_symlink(result_root: Path, merge_name: str):
    symlink = result_root / "merge_finalmap"
    target = result_root / merge_name
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    os.symlink(str(target), str(symlink))
    logger.info(f"merge_finalmap -> {merge_name}")


def copy_ref_data(ref_submap_dir: Path, result_merge_dir: Path, images: List[str]):
    poses = read_poses(str(ref_submap_dir / "poses.txt"))
    poses_gt = read_poses(str(ref_submap_dir / "poses_abs_gt.txt"))
    timestamps = read_timestamps(str(ref_submap_dir / "timestamps.txt"))

    ref_poses = {img: poses[img] for img in images if img in poses}
    ref_gt = {img: poses_gt[img] for img in images if img in poses_gt}
    ref_ts = {img: timestamps[img] for img in images if img in timestamps}

    write_poses_txt(ref_poses, result_merge_dir / "poses.txt")
    write_poses_txt(ref_gt, result_merge_dir / "poses_abs_gt.txt")
    write_timestamps_txt(ref_ts, result_merge_dir / "timestamps.txt")
    return ref_poses, ref_gt, ref_ts


def reindex_dict(
    data: Dict[str, np.ndarray],
    images_ordered: List[str],
    global_offset: int,
) -> Dict[str, np.ndarray]:
    """Rename image keys from local seq/XXXXXX to global seq/YYYYYY.

    Args:
        data: {img_name: value} dict using local seq/XXXXXX naming
        images_ordered: ordered list of local image names for this submap
        global_offset: number of frames already in the merged map before this submap

    Returns:
        New dict with keys renamed to seq/{global_offset+i:06d}.color.jpg
    """
    result = {}
    local_to_global = {
        img: f"seq/{global_offset + i:06d}.color.jpg"
        for i, img in enumerate(images_ordered)
    }
    for local_name, value in data.items():
        if local_name in local_to_global:
            result[local_to_global[local_name]] = value
        else:
            logger.warning("reindex_dict: %s not in images_ordered, dropped", local_name)
    return result


def read_intrinsics(file_path: str) -> Dict[str, np.ndarray]:
    """Read intrinsics file. Returns {img_name: array([fx,fy,cx,cy,w,h])}."""
    data = {}
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 7:
                data[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float64)
    return data


def read_gps(file_path: str) -> Dict[str, np.ndarray]:
    """Read gps_data file. Returns {img_name: array([lat,lon,nan,nan,alt])}."""
    data = {}
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                vals = []
                for x in parts[1:]:
                    vals.append(float('nan') if x == 'nan' else float(x))
                data[parts[0]] = np.array(vals, dtype=np.float64)
    return data


def read_edges_odom(file_path: str) -> List[Tuple[int, int, float]]:
    """Read edges_odom file. Returns list of (node_a, node_b, distance).

    Returns empty list if file does not exist (e.g. full_data submap has no edges_odom.txt).
    """
    if not Path(file_path).exists():
        return []
    edges = []
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                edges.append((int(parts[0]), int(parts[1]), float(parts[2])))
    return edges


def write_intrinsics_txt(intrinsics: Dict[str, np.ndarray], path: Path) -> None:
    """Write intrinsics: img_name fx fy cx cy width height."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for img in sorted(intrinsics.keys()):
            v = intrinsics[img]
            f.write(f"{img} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                    f"{v[3]:.6f} {int(v[4])} {int(v[5])}\n")


def write_gps_txt(gps: Dict[str, np.ndarray], path: Path) -> None:
    """Write gps_data: img_name lat lon nan nan alt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for img in sorted(gps.keys()):
            v = gps[img]
            lat, lon, alt = v[0], v[1], v[4]
            f.write(f"{img} {lat:.6f} {lon:.6f} nan nan {alt:.6f}\n")


def write_edges_odom_txt(edges: List[Tuple[int, int, float]], path: Path) -> None:
    """Write edges_odom: node_a node_b distance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for a, b, d in edges:
            f.write(f"{a} {b} {d:.6f}\n")


def merge_edges_with_offset(
    existing: List[Tuple[int, int, float]],
    new_edges: List[Tuple[int, int, float]],
    offset: int,
) -> List[Tuple[int, int, float]]:
    """Append new_edges to existing after shifting new node IDs by offset.

    Args:
        existing: already-merged edge list
        new_edges: edges from the incoming submap (local node IDs start at 0)
        offset: total frames in merged map BEFORE this submap (= current merged size)
    """
    shifted = [(a + offset, b + offset, d) for a, b, d in new_edges]
    return existing + shifted
