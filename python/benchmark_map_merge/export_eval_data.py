"""Export merged map poses to TUM format and copy to slam_trajectory_evaluation
directory structure for ATE/RPE computation."""

import os
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from .merge_writer import read_poses, read_timestamps

logger = logging.getLogger(__name__)


def _vec_to_matrix_c2w(quat: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """quat=[qw,qx,qy,qz], trans=[tx,ty,tz] → 4x4 camera-to-world matrix."""
    from scipy.spatial.transform import Rotation
    q_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = trans
    return T


def _matrix_to_vec_w2c(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 world-to-camera → (trans_xyz, quat_xyzw)."""
    from scipy.spatial.transform import Rotation
    R = T[:3, :3]
    t = T[:3, 3]
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return t, q_xyzw


def _convert_mapfree_to_tum_data(
    pose_dict: Dict[str, np.ndarray],
    timestamp_dict: Dict[str, float],
) -> np.ndarray:
    """Convert mapfree-format poses to TUM array.

    mapfree: img_name qw qx qy qz tx ty tz (camera-to-world, wxyz)
    TUM: timestamp tx ty tz qx qy qz qw (world-to-camera, xyzw)
    """
    entries = []
    for img_name, pose_vec in pose_dict.items():
        if img_name not in timestamp_dict:
            continue
        quat_wxyz = pose_vec[0:4]
        trans_xyz = pose_vec[4:7]
        c2w = _vec_to_matrix_c2w(quat_wxyz, trans_xyz)
        w2c = np.linalg.inv(c2w)
        tum_trans, tum_quat = _matrix_to_vec_w2c(w2c)
        ts = timestamp_dict[img_name]
        entries.append([ts] + tum_trans.tolist() + tum_quat.tolist())
    return np.array(entries)


def export_tum_files(
    merge_dir: Path,
    output_gt_path: Path,
    output_est_path: Path,
):
    """Convert merged map poses to TUM format and save.

    merge_dir: directory containing poses.txt, poses_abs_gt.txt, timestamps.txt
    output_gt_path: path for GT TUM file
    output_est_path: path for estimated TUM file
    """
    poses_est = read_poses(str(merge_dir / "poses.txt"))
    poses_gt = read_poses(str(merge_dir / "poses_abs_gt.txt"))
    timestamps = read_timestamps(str(merge_dir / "timestamps.txt"))

    output_gt_path.parent.mkdir(parents=True, exist_ok=True)
    output_est_path.parent.mkdir(parents=True, exist_ok=True)

    gt_tum = _convert_mapfree_to_tum_data(poses_gt, timestamps)
    est_tum = _convert_mapfree_to_tum_data(poses_est, timestamps)

    np.savetxt(output_gt_path, gt_tum, fmt="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f")
    np.savetxt(output_est_path, est_tum, fmt="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f")
    logger.info(f"GT TUM: {output_gt_path} ({len(gt_tum)} poses)")
    logger.info(f"EST TUM: {output_est_path} ({len(est_tum)} poses)")


def export_to_eval_structure(
    merge_dir: Path,
    traj_eval_data_root: Path,
    dataset_order_name: str,
    method_name: str,
):
    """Export TUM files to slam_trajectory_evaluation directory structure.

    GT: <root>/groundtruth/traj/<dataset_order_name>.txt
    EST: <root>/algorithms/<method_name>/laptop/traj/<dataset_order_name>.txt
    """
    gt_path = traj_eval_data_root / "groundtruth" / "traj" / f"{dataset_order_name}.txt"
    est_path = (traj_eval_data_root / "algorithms" / method_name
                / "laptop" / "traj" / f"{dataset_order_name}.txt")

    export_tum_files(merge_dir, gt_path, est_path)
    return gt_path, est_path
