"""create_pose_graph_from_map must report which factors are loop closures."""
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


def test_loop_factor_indices_follow_odometry_factors():
    from argparse import Namespace

    from map_merge_pipeline import MergePipeline

    a0, a1 = _Node(0, (0, 0, 0)), _Node(1, (1, 0, 0))
    a0.edges = {1: (a1, 1.0)}
    b0, b1 = _Node(0, (10, 0, 0)), _Node(1, (11, 0, 0))
    b0.edges = {1: (b1, 1.0)}

    merger = MergePipeline.__new__(MergePipeline)
    merger.id_offset = 100
    merger.loop_edge_registry = {}
    merger.args = Namespace(
        pgo_loop_sigma_trans=0.1,
        pgo_loop_sigma_rot=1.0,
    )

    inter_edges = [(a0, b0, np.eye(4), 0.8, 0.5)]
    pose_graph, subgraph_keys, loop_indices, loop_keys = merger.create_pose_graph_from_map(
        _Graph([a0, a1]), _Graph([b0, b1]), inter_edges)

    assert len(loop_indices) == len(inter_edges)
    # Two odometry factors (one per submap) precede the single loop factor.
    assert loop_indices == [2]
    factor = pose_graph.get_factor_graph().at(loop_indices[0])
    assert list(factor.keys()) == [0, 100]
    # Priors are appended after the loop factors.
    assert pose_graph.get_factor_graph().size() > loop_indices[0] + 1
