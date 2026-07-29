"""Trajectory and classification metrics for the robust-PGO benchmark."""
from typing import Dict, Sequence

import gtsam
import numpy as np


def _umeyama(src: np.ndarray, dst: np.ndarray) -> gtsam.Pose3:
    """Rigid transform (no scale) mapping the src point cloud onto dst."""
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    cov = (dst - dst_mean).T @ (src - src_mean)
    u, _, vt = np.linalg.svd(cov)
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    rot = u @ correction @ vt
    return gtsam.Pose3(gtsam.Rot3(rot), gtsam.Point3(*(dst_mean - rot @ src_mean)))


def ate(reference: gtsam.Values, estimate: gtsam.Values) -> Dict[str, float]:
    """Absolute trajectory error after SE(3) alignment.

    Both inputs must hold Pose3 values under the same keys. Translation RMSE is
    in metres, rotation RMSE in degrees.
    """
    keys = sorted(int(k) for k in reference.keys())
    ref_poses = [reference.atPose3(k) for k in keys]
    est_poses = [estimate.atPose3(k) for k in keys]

    align = _umeyama(
        np.array([p.translation() for p in est_poses]),
        np.array([p.translation() for p in ref_poses]),
    )

    trans_errors, rot_errors = [], []
    for ref_pose, est_pose in zip(ref_poses, est_poses):
        aligned = align.compose(est_pose)
        trans_errors.append(
            float(np.linalg.norm(aligned.translation() - ref_pose.translation())))
        delta = ref_pose.rotation().between(aligned.rotation())
        rot_errors.append(float(np.rad2deg(np.linalg.norm(gtsam.Rot3.Logmap(delta)))))

    return {
        "trans_rmse": float(np.sqrt(np.mean(np.square(trans_errors)))),
        "rot_rmse": float(np.sqrt(np.mean(np.square(rot_errors)))),
    }


def outlier_classification(
    weights: np.ndarray,
    injected_indices: Sequence[int],
    candidate_indices: Sequence[int],
    weight_threshold: float = 0.5,
) -> Dict[str, float]:
    """Precision and recall of outlier *rejection* over the candidate factors.

    A candidate counts as rejected when its GNC weight falls below
    `weight_threshold`. Precision is 1.0 when nothing was rejected, recall is
    1.0 when nothing was injected.
    """
    injected = {int(i) for i in injected_indices}
    rejected = {int(i) for i in candidate_indices if weights[i] < weight_threshold}
    true_positives = len(rejected & injected)
    return {
        "precision": true_positives / len(rejected) if rejected else 1.0,
        "recall": true_positives / len(injected) if injected else 1.0,
        "num_rejected": float(len(rejected)),
        "num_injected": float(len(injected)),
    }
