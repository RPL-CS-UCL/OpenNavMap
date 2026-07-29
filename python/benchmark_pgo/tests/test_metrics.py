"""Metric tests for the robust-PGO benchmark."""
import gtsam
import numpy as np

from benchmark_pgo import metrics


def _values(translations, rotations=None):
    values = gtsam.Values()
    for i, t in enumerate(translations):
        rot = rotations[i] if rotations is not None else gtsam.Rot3()
        values.insert(i, gtsam.Pose3(rot, gtsam.Point3(*t)))
    return values


def test_ate_is_zero_for_identical_trajectories():
    v = _values([(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)])
    result = metrics.ate(v, v)
    assert result["trans_rmse"] < 1e-9
    assert result["rot_rmse"] < 1e-9


def test_ate_is_invariant_to_rigid_transform():
    """A globally shifted and rotated estimate must align back to zero error."""
    ref = _values([(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)])
    shift = gtsam.Pose3(gtsam.Rot3.Yaw(0.7), gtsam.Point3(5, -3, 2))
    est = gtsam.Values()
    for k in ref.keys():
        est.insert(k, shift.compose(ref.atPose3(k)))
    result = metrics.ate(ref, est)
    assert result["trans_rmse"] < 1e-9
    assert result["rot_rmse"] < 1e-6


def test_ate_detects_translation_error():
    ref = _values([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
    est = _values([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 4, 0)])
    assert metrics.ate(ref, est)["trans_rmse"] > 0.5


def test_outlier_classification_perfect_score():
    weights = np.array([1.0, 1.0, 0.0, 0.0])
    result = metrics.outlier_classification(
        weights, injected_indices=[2, 3], candidate_indices=[0, 1, 2, 3])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["num_rejected"] == 2


def test_outlier_classification_penalises_false_positive():
    weights = np.array([1.0, 0.0, 0.0, 0.0])
    result = metrics.outlier_classification(
        weights, injected_indices=[2, 3], candidate_indices=[0, 1, 2, 3])
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0


def test_outlier_classification_penalises_miss():
    weights = np.array([1.0, 1.0, 1.0, 0.0])
    result = metrics.outlier_classification(
        weights, injected_indices=[2, 3], candidate_indices=[0, 1, 2, 3])
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
