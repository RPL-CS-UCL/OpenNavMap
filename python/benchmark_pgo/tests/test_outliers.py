"""Outlier injection tests for the robust-PGO benchmark."""
import gtsam
import numpy as np

from benchmark_pgo import outliers


def _chain_graph(num_poses=20):
    """A straight odometry chain plus one genuine loop closure."""
    graph = gtsam.NonlinearFactorGraph()
    noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3))
    step = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0))

    initial = gtsam.Values()
    pose = gtsam.Pose3()
    for i in range(num_poses):
        initial.insert(i, pose)
        pose = pose.compose(step)
    for i in range(num_poses - 1):
        graph.add(gtsam.BetweenFactorPose3(i, i + 1, step, noise))
    graph.add(gtsam.BetweenFactorPose3(0, num_poses - 1, gtsam.Pose3(), noise))
    return graph, initial


def test_count_loop_closures_ignores_sequential_edges():
    graph, _ = _chain_graph()
    assert outliers.count_loop_closures(graph) == 1


def test_inject_appends_requested_number():
    graph, initial = _chain_graph()
    before = graph.size()
    out, injected = outliers.inject(graph, initial, num_outliers=5, seed=0)
    assert out.size() == before + 5
    assert len(injected) == 5
    assert injected == list(range(before, before + 5))


def test_inject_does_not_mutate_input():
    graph, initial = _chain_graph()
    before = graph.size()
    outliers.inject(graph, initial, num_outliers=5, seed=0)
    assert graph.size() == before


def test_inject_is_deterministic_given_seed():
    graph, initial = _chain_graph()
    out_a, _ = outliers.inject(graph, initial, num_outliers=5, seed=42)
    out_b, _ = outliers.inject(graph, initial, num_outliers=5, seed=42)
    for i in range(graph.size(), out_a.size()):
        assert out_a.at(i).measured().equals(out_b.at(i).measured(), 1e-12)
        assert list(out_a.at(i).keys()) == list(out_b.at(i).keys())


def test_inject_never_duplicates_or_uses_sequential_pairs():
    graph, initial = _chain_graph()
    out, injected = outliers.inject(graph, initial, num_outliers=8, seed=7)
    seen = set()
    for idx in injected:
        a, b = (int(k) for k in out.at(idx).keys())
        assert abs(a - b) > 1, "injected edge must not shadow an odometry edge"
        pair = (min(a, b), max(a, b))
        assert pair not in seen
        seen.add(pair)
        assert pair != (0, 19), "must not duplicate the genuine loop closure"
