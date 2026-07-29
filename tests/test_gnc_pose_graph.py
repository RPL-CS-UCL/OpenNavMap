"""GNC robust pose graph optimization tests."""
import gtsam
import numpy as np
import pytest

from utils.gtsam_pose_graph import PoseGraph


def _square_graph_with_outlier():
    """4-pose square: 3 odometry edges, 1 correct closure, 1 gross outlier, 1 prior.

    Factor indices: 0,1,2 = odometry; 3 = inlier closure; 4 = outlier; 5 = prior.
    """
    graph = gtsam.NonlinearFactorGraph()
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3) / 10)
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3))
    step = gtsam.Pose3(gtsam.Rot3.Yaw(np.pi / 2), gtsam.Point3(2, 0, 0))

    for i in range(3):
        graph.add(gtsam.BetweenFactorPose3(i, i + 1, step, odom_noise))
    graph.add(gtsam.BetweenFactorPose3(3, 0, step, loop_noise))
    graph.add(gtsam.BetweenFactorPose3(
        1, 3, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(50, 50, 50)), loop_noise))
    graph.add(gtsam.PriorFactorPose3(
        0, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4))))

    initial = gtsam.Values()
    pose = gtsam.Pose3()
    for i in range(4):
        initial.insert(i, pose)
        pose = pose.compose(step)
    return graph, initial


def test_gnc_rejects_outlier_and_keeps_inlier():
    graph, initial = _square_graph_with_outlier()
    result, weights = PoseGraph.optimize_pose_graph_with_GNC(
        graph, initial, known_inlier_indices=[0, 1, 2, 5], loss='TLS')

    assert weights.shape == (6,)
    assert weights[4] < 0.1, f"outlier edge should be rejected, got {weights[4]}"
    assert weights[3] > 0.9, f"inlier closure should be kept, got {weights[3]}"
    assert result.size() == 4


def test_gnc_honours_known_inliers():
    """Factors listed as known inliers keep weight 1 regardless of residual."""
    graph, initial = _square_graph_with_outlier()
    _, weights = PoseGraph.optimize_pose_graph_with_GNC(
        graph, initial, known_inlier_indices=[0, 1, 2, 4, 5], loss='TLS')
    assert weights[4] > 0.9, "explicitly trusted factor must not be down-weighted"


def test_gnc_gm_loss_runs():
    graph, initial = _square_graph_with_outlier()
    _, weights = PoseGraph.optimize_pose_graph_with_GNC(
        graph, initial, known_inlier_indices=[0, 1, 2, 5], loss='GM')
    assert weights[4] < weights[3]


def test_gnc_rejects_unknown_loss():
    graph, initial = _square_graph_with_outlier()
    with pytest.raises(ValueError, match="Unsupported GNC loss"):
        PoseGraph.optimize_pose_graph_with_GNC(graph, initial, loss='huber')


def test_factory_methods_return_factor_indices():
    pose_graph = PoseGraph()
    sigma = np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3)
    idx_a = pose_graph.add_odometry_factor(0, gtsam.Pose3(), 1, gtsam.Pose3(), sigma)
    idx_b = pose_graph.add_odometry_factor(1, gtsam.Pose3(), 2, gtsam.Pose3(), sigma)
    idx_c = pose_graph.add_prior_factor(0, gtsam.Pose3(), sigma)
    assert [idx_a, idx_b, idx_c] == [0, 1, 2]
