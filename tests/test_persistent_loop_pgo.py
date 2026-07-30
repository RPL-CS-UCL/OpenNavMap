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
