"""GTSAM pose graph optimizer for multi-session map merging.

Builds a factor graph with:
  - PriorFactor  : reference submap frames (anchor to W0 coordinate frame)
  - BetweenFactor: consecutive VIO frames within each submap (odometry edges)
  - PriorFactor  : incoming frames with successful absolute localization (e.g. PnP)

Can be reused by any merging method that provides VIO poses and absolute constraints.
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gtsam
from utils.gtsam_pose_graph import PoseGraph

logger = logging.getLogger(__name__)

# Noise sigmas: [rot_rad×3, trans_m×3]
_SIGMA_REF  = np.array([1e-4, 1e-4, 1e-4, 1e-3,  1e-3,  1e-3])   # ref locked tightly
_SIGMA_ODOM = np.array([0.01, 0.01, 0.01, 0.05,  0.05,  0.05])   # VIO odometry
_SIGMA_PNP  = np.array([0.05, 0.05, 0.05, 0.10,  0.10,  0.10])   # absolute pose constraint


def _vec_to_T(pose_vec: np.ndarray) -> np.ndarray:
    """[qw,qx,qy,qz,tx,ty,tz] camera-to-world → 4×4 numpy float64."""
    from scipy.spatial.transform import Rotation
    q = pose_vec[:4]  # qw qx qy qz
    t = pose_vec[4:7]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3,  3] = t
    return T


def _T_to_vec(T: np.ndarray) -> np.ndarray:
    """4×4 camera-to-world numpy → [qw,qx,qy,qz,tx,ty,tz]."""
    from scipy.spatial.transform import Rotation
    q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2],
                     T[0, 3], T[1, 3], T[2, 3]], dtype=np.float64)


def _to_pose3(T: np.ndarray) -> gtsam.Pose3:
    """4×4 camera-to-world numpy → gtsam.Pose3."""
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(T[:3, 3]))


class PoseGraphOptimizer:
    """Incrementally builds and optimizes a multi-submap pose graph.

    Usage:
        opt = PoseGraphOptimizer()
        offset0 = opt.add_submap(ref_poses_vio, ref_images, is_reference=True)
        offset1 = opt.add_submap(inc_poses_vio, inc_images, abs_poses_w0=pnp_results)
        all_pairs = [(offset0+i, img) for i, img in enumerate(ref_images)] + ...
        optimized = opt.optimize(all_pairs)
        # optimized: {global_key: [qw,qx,qy,qz,tx,ty,tz]}
    """

    def __init__(self) -> None:
        self._pg = PoseGraph()
        self._n_total = 0  # global key counter (= total frames added so far)

    def add_submap(
        self,
        poses_vio: Dict[str, np.ndarray],
        images_ordered: List[str],
        abs_poses_w0: Optional[Dict[str, np.ndarray]] = None,
        is_reference: bool = False,
    ) -> int:
        """Add one submap's frames to the factor graph.

        Args:
            poses_vio: {local_img_name: [qw,qx,qy,qz,tx,ty,tz]} VIO poses (local frame)
            images_ordered: ordered list of local image names for this submap
            abs_poses_w0: {local_img_name: 4×4 C2W numpy} absolute poses in W0 frame.
                          For the reference submap pass None; set is_reference=True instead.
            is_reference: if True, all frames get tight prior factors to anchor W0.

        Returns:
            global offset (key of the first frame of this submap).
        """
        offset = self._n_total
        abs_poses = abs_poses_w0 or {}

        for i, img in enumerate(images_ordered):
            if img not in poses_vio:
                continue
            key = offset + i
            T_vio = _vec_to_T(poses_vio[img])
            T_init = abs_poses[img] if img in abs_poses else T_vio
            self._pg.add_init_estimate(key, _to_pose3(T_init))

            if is_reference:
                self._pg.add_prior_factor(key, _to_pose3(T_vio), _SIGMA_REF)
            elif img in abs_poses:
                self._pg.add_prior_factor(key, _to_pose3(abs_poses[img]), _SIGMA_PNP)

        for i in range(len(images_ordered) - 1):
            img_a = images_ordered[i]
            img_b = images_ordered[i + 1]
            if img_a not in poses_vio or img_b not in poses_vio:
                continue
            self._pg.add_odometry_factor(
                offset + i,     _to_pose3(_vec_to_T(poses_vio[img_a])),
                offset + i + 1, _to_pose3(_vec_to_T(poses_vio[img_b])),
                _SIGMA_ODOM,
            )

        self._n_total += len(images_ordered)
        return offset

    def optimize(
        self,
        all_frames: List[Tuple[int, str]],
    ) -> Dict[int, np.ndarray]:
        """Run GTSAM optimization and return optimized poses.

        Args:
            all_frames: list of (global_key, local_img_name) for all frames across submaps

        Returns:
            {global_key: [qw,qx,qy,qz,tx,ty,tz]} optimized camera-to-world poses.
            Keys with no graph constraints may be absent.
        """
        pg_result = self._pg.perform_optimization()
        current_estimate = pg_result['current_estimate']
        optimized: Dict[int, np.ndarray] = {}
        for key, _ in all_frames:
            try:
                T = current_estimate.atPose3(key).matrix()
                optimized[key] = _T_to_vec(T)
            except Exception:
                pass
        logger.info(f"Pose graph optimized: {len(optimized)}/{len(all_frames)} poses")
        return optimized
