import numpy as np
import pytest

from rosbag_convert.sessions import split_by_distance, split_by_time


def _square_laps(laps: int, side: float = 10.0, step: float = 0.5) -> np.ndarray:
    n = int(side / step)
    corners = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=float)
    pts = []
    for _ in range(laps):
        for c0, c1 in zip(corners, np.roll(corners, -1, axis=0)):
            pts.extend(c0 + (c1 - c0) * (np.arange(n) / n)[:, None])
    pts = np.asarray(pts)
    return np.hstack([pts, np.zeros((len(pts), 1))])


def test_equal_distance_split_is_contiguous_and_balanced():
    pos = _square_laps(3)
    ranges = split_by_distance(pos, 3)
    assert ranges[0][0] == 0 and ranges[-1][1] == len(pos)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    lengths = [e - s for s, e in ranges]
    assert max(lengths) - min(lengths) <= 1


def test_single_session_returns_whole_range():
    pos = _square_laps(1)
    assert split_by_distance(pos, 1) == [(0, len(pos))]


def test_split_by_time_boundaries():
    stamps = np.arange(0.0, 10.0, 1.0)
    assert split_by_time(stamps, [3.0, 7.0]) == [(0, 3), (3, 7), (7, 10)]


def test_empty_session_raises():
    pos = np.zeros((5, 3))  # robot never moves: no distance to split on
    with pytest.raises(ValueError):
        split_by_distance(pos, 3)


def test_build_sessions_assigns_contiguous_names():
    from rosbag_convert.bag_reader import Trajectory
    from rosbag_convert.config import BagConversionConfig
    from rosbag_convert.convert_rosbag_to_multisession import build_sessions

    n = 300
    stamps = np.arange(n) * 0.0692
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * 0.1  # 30 m straight line
    cfg = BagConversionConfig(bag_path="x", output_root="y", kf_trans_thresh_m=1.0, num_sessions=3)
    sessions = build_sessions(cfg, Trajectory(stamps, poses), Trajectory(stamps, poses),
                              (stamps * 1e9).astype(np.int64))
    assert len(sessions) == 3
    assert [f.name for f in sessions[1]][:2] == ["seq/000000.color.jpg", "seq/000001.color.jpg"]
    assert all(len(s) >= 3 for s in sessions)
    assert sessions[0][-1].stamp_ns < sessions[1][0].stamp_ns
