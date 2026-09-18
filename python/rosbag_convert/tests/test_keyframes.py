import numpy as np
from scipy.spatial.transform import Rotation

from rosbag_convert.keyframes import select_keyframes


def _translations(step: float, n: int) -> np.ndarray:
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * step
    return poses


def _yaw_rotations(step_deg: float, n: int) -> np.ndarray:
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        poses[i, :3, :3] = Rotation.from_euler("z", i * step_deg, degrees=True).as_matrix()
    return poses


def test_straight_line_keeps_every_threshold_crossing():
    # 0.1 m steps, 0.95 m threshold (not a multiple of the step, so float
    # rounding cannot flip the comparison): index 10 is the first beyond it
    assert select_keyframes(_translations(0.1, 25), 0.95, 30.0) == [0, 10, 20]


def test_in_place_rotation_triggers_on_angle():
    # 5 deg steps, 32 deg threshold: index 7 (35 deg) is the first beyond it
    assert select_keyframes(_yaw_rotations(5.0, 20), 1.0, 32.0) == [0, 7, 14]


def test_stationary_jitter_keeps_only_first():
    rng = np.random.default_rng(0)
    poses = np.tile(np.eye(4), (50, 1, 1))
    poses[:, :3, 3] = rng.normal(scale=1e-3, size=(50, 3))
    assert select_keyframes(poses, 1.0, 30.0) == [0]


def test_single_pose():
    assert select_keyframes(np.eye(4)[None], 1.0, 30.0) == [0]
