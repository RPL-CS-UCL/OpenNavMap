import numpy as np

from rosbag_convert.sync import associate_by_stamp, cumulative_path_length


def test_exact_match_pairs_every_index():
    q = np.array([0.0, 0.1, 0.2])
    assert associate_by_stamp(q, q.copy(), 0.02) == [(0, 0), (1, 1), (2, 2)]


def test_shifted_reference_pairs_with_next_index():
    # odometry[i] carries the stamp of image[i-1] (measured on the real bag)
    img = np.array([0.0, 0.1, 0.2, 0.3])
    odo = np.array([-0.1, 0.0, 0.1, 0.2]) + 1.2e-6
    assert associate_by_stamp(img, odo, 0.02) == [(0, 1), (1, 2), (2, 3)]


def test_beyond_tolerance_is_dropped():
    q = np.array([0.0, 0.1, 0.5])
    ref = np.array([0.0, 0.1, 0.2])
    assert associate_by_stamp(q, ref, 0.02) == [(0, 0), (1, 1)]


def test_picks_nearer_neighbour():
    q = np.array([0.14])
    ref = np.array([0.1, 0.15, 0.2])
    assert associate_by_stamp(q, ref, 0.02) == [(0, 1)]


def test_cumulative_path_length_starts_at_zero():
    pos = np.array([[0, 0, 0], [3, 4, 0], [3, 4, 2]], dtype=float)
    np.testing.assert_allclose(cumulative_path_length(pos), [0.0, 5.0, 7.0])
