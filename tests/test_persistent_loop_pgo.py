"""Persistent loop factors: historical loop edges must stay re-judgeable."""
import numpy as np
import pytest

gtsam = pytest.importorskip("gtsam")


class _Node:
    """Minimal stand-in for litevloc's ImageNode."""

    def __init__(self, node_id, trans):
        self.id = node_id
        self.trans = np.array(trans, dtype=float)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.edges = {}


class _Graph:
    def __init__(self, nodes):
        self.nodes = {n.id: n for n in nodes}


def _translation(dx, dy=0.0, dz=0.0):
    T = np.eye(4)
    T[:3, 3] = (dx, dy, dz)
    return T


def _merger(id_offset=100):
    from map_merge_pipeline import MergePipeline

    merger = MergePipeline.__new__(MergePipeline)
    merger.id_offset = id_offset
    merger.loop_edge_registry = {}
    return merger


def test_registry_edge_becomes_loop_factor_with_original_measurement():
    """A registered inter-submap edge must not be re-measured as odometry."""
    # Submap A: nodes 0,1. Submap B (already merged, global ids 10,11).
    a0, a1 = _Node(0, (0, 0, 0)), _Node(1, (1, 0, 0))
    b0, b1 = _Node(10, (50, 0, 0)), _Node(11, (51, 0, 0))
    a0.edges = {1: (a1, 1.0)}
    b0.edges = {11: (b1, 1.0)}
    # The inter-submap edge lives in the odom graph (as the pipeline writes it).
    a1.edges = {10: (b0, 1.0)}
    b0.edges[1] = (a1, 1.0)

    merger = _merger()
    # Registry says the original measurement was 2 m, but the current poses are 49 m apart.
    registry = {(1, 10): {'T_AB': _translation(2.0), 'conf': 1.0,
                          'first_step': 0, 'reject_count': 0, 'last_weight': 1.0}}

    new = _Node(0, (0, 0, 0))
    pose_graph, _, loop_indices, loop_keys = merger.create_pose_graph_from_map(
        _Graph([a0, a1, b0, b1]), _Graph([new]), [], loop_registry=registry)

    assert loop_keys == [(1, 10)]
    assert len(loop_indices) == 1
    factor = pose_graph.get_factor_graph().at(loop_indices[0])
    assert list(factor.keys()) == [1, 10]
    # The factor carries the registry measurement (2 m), not the 49 m pose delta.
    assert factor.measured().translation()[0] == pytest.approx(2.0)
    # And no odometry factor duplicates the same key pair.
    between = [pose_graph.get_factor_graph().at(i)
               for i in range(pose_graph.get_factor_graph().size())]
    key_pairs = [tuple(f.keys()) for f in between
                 if isinstance(f, gtsam.BetweenFactorPose3)]
    assert key_pairs.count((1, 10)) == 1


def test_empty_registry_preserves_legacy_factor_order():
    """With no registry the graph must be byte-identical to the old behaviour."""
    a0, a1 = _Node(0, (0, 0, 0)), _Node(1, (1, 0, 0))
    a0.edges = {1: (a1, 1.0)}
    b0, b1 = _Node(0, (10, 0, 0)), _Node(1, (11, 0, 0))
    b0.edges = {1: (b1, 1.0)}

    merger = _merger()
    inter_edges = [(a0, b0, np.eye(4), 0.8, 0.5)]
    pose_graph, subgraph_keys, loop_indices, loop_keys = merger.create_pose_graph_from_map(
        _Graph([a0, a1]), _Graph([b0, b1]), inter_edges, loop_registry=None)

    assert loop_indices == [2]
    assert loop_keys == [(0, 100)]
    # One connected component (the loop factor joins both submaps) -> one prior.
    assert len(subgraph_keys) == 1
    assert pose_graph.get_factor_graph().size() == 4


def _rot_z(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    R = np.eye(4)
    R[:2, :2] = ((c, -s), (s, c))
    return R


def test_gnc_overturns_a_registered_bad_edge_once_good_edges_disagree():
    """The whole point: a 180-deg edge accepted alone must lose when outvoted."""
    from utils.gtsam_pose_graph import PoseGraph

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    tight = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3) / 10)
    loop = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3))

    # Three submaps of two nodes each, laid out along +x at 0, 10, 20 m.
    for base, x0 in ((0, 0.0), (10, 10.0), (20, 20.0)):
        for k in (0, 1):
            initial.insert(base + k, gtsam.Pose3(
                gtsam.Rot3(), np.array([x0 + k, 0.0, 0.0])))
        graph.add(gtsam.BetweenFactorPose3(
            base, base + 1, gtsam.Pose3(gtsam.Rot3(), np.array([1.0, 0.0, 0.0])),
            tight))
    graph.add(gtsam.PriorFactorPose3(
        0, initial.atPose3(0),
        gtsam.noiseModel.Diagonal.Sigmas(
            np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3) / 100)))
    known_inliers = list(range(graph.size()))

    # The bad historical edge: submap 0 -> submap 1, flipped 180 deg.
    bad = gtsam.Pose3(_rot_z(180.0)).compose(
        gtsam.Pose3(gtsam.Rot3(), np.array([10.0, 0.0, 0.0])))
    graph.add(gtsam.BetweenFactorPose3(0, 10, bad, loop))
    # Good edges arriving with submap 2 that contradict it.
    graph.add(gtsam.BetweenFactorPose3(
        0, 20, gtsam.Pose3(gtsam.Rot3(), np.array([20.0, 0.0, 0.0])), loop))
    graph.add(gtsam.BetweenFactorPose3(
        10, 20, gtsam.Pose3(gtsam.Rot3(), np.array([10.0, 0.0, 0.0])), loop))
    graph.add(gtsam.BetweenFactorPose3(
        11, 20, gtsam.Pose3(gtsam.Rot3(), np.array([9.0, 0.0, 0.0])), loop))
    bad_index = graph.size() - 4

    _, weights = PoseGraph.optimize_pose_graph_with_GNC(
        graph, initial, known_inlier_indices=known_inliers,
        loss='TLS', barc_prob=0.99)

    assert weights[bad_index] < 0.5, "the 180-deg historical edge must be rejected"
    assert all(weights[i] >= 0.5 for i in range(bad_index + 1, graph.size())), \
        "the three consistent edges must survive"


def _two_submap_graph(loop_measurement_x=None):
    """Submap 0 (nodes 0,1) gauge-fixed; submap 1 (nodes 10,11) weakly anchored."""
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    tight = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3) / 10)
    for base, x0 in ((0, 0.0), (10, 10.0)):
        for k in (0, 1):
            initial.insert(base + k, gtsam.Pose3(
                gtsam.Rot3(), np.array([x0 + k, 0.0, 0.0])))
        graph.add(gtsam.BetweenFactorPose3(
            base, base + 1, gtsam.Pose3(gtsam.Rot3(), np.array([1.0, 0.0, 0.0])),
            tight))
    graph.add(gtsam.PriorFactorPose3(
        0, initial.atPose3(0),
        gtsam.noiseModel.Diagonal.Sigmas(
            np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3) / 100)))
    graph.add(gtsam.PriorFactorPose3(
        10, initial.atPose3(10),
        gtsam.noiseModel.Diagonal.Sigmas(
            np.array([np.deg2rad(30.0)] * 3 + [30.0] * 3))))
    known_inliers = list(range(graph.size()))
    if loop_measurement_x is not None:
        graph.add(gtsam.BetweenFactorPose3(
            0, 10, gtsam.Pose3(gtsam.Rot3(), np.array([loop_measurement_x, 0.0, 0.0])),
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3))))
    return graph, initial, known_inliers


def test_weak_anchor_holds_a_submap_with_no_surviving_loop():
    """A submap whose loop factors all got zero weight must stay where it was.

    GNC zeroing a factor's weight is equivalent to dropping it from the linear
    system, so this is modelled by omitting the loop factor entirely.
    """
    from utils.gtsam_pose_graph import PoseGraph

    graph, initial, known_inliers = _two_submap_graph(loop_measurement_x=None)
    result, _ = PoseGraph.optimize_pose_graph_with_GNC(
        graph, initial, known_inlier_indices=known_inliers,
        loss='TLS', barc_prob=0.99)

    drift = np.linalg.norm(
        result.atPose3(10).translation() - initial.atPose3(10).translation())
    assert np.isfinite(drift)
    assert drift < 0.5, f"weakly anchored submap drifted {drift:.3f} m"


def test_weak_anchor_does_not_fight_a_surviving_loop_factor():
    """The anchor must not freeze a submap: a loop factor still moves it fully.

    This is why non-gauge submaps get anchor_sigma (30 deg / 30 m) and not the
    tight prior_sigma the old per-component code applied -- a tight prior would
    pin every submap in place and leave PGO nothing to correct.
    """
    from utils.gtsam_pose_graph import PoseGraph

    graph, initial, known_inliers = _two_submap_graph(loop_measurement_x=12.0)
    result, _ = PoseGraph.optimize_pose_graph_with_GNC(
        graph, initial, known_inlier_indices=known_inliers,
        loss='TLS', barc_prob=0.99)

    # Loop sigma (0.1 m) outweighs the anchor (30 m) by ~300x, so the submap
    # ends up at the loop's 12 m, not held back near its 10 m initial guess.
    assert result.atPose3(10).translation()[0] == pytest.approx(12.0, abs=0.01)
