"""GNC-rejected loop edges must be dropped before they reach the topology graphs."""
import numpy as np
import pytest

pytest.importorskip("gtsam")

from map_merge_pipeline import split_edges_by_gnc_weight
from utils_map_merging import PGO_INLIER_WEIGHT_THRESHOLD


class _Node:
    def __init__(self, node_id):
        self.id = node_id


def _edge(db_id, query_id):
    return (_Node(db_id), _Node(query_id), np.eye(4), 0.8, 0.5)


def _history(edges):
    """Edges reaching PGO were already recorded by the VPR stage."""
    return {(e[0].id, e[1].id): {"action": "added_by_vpr",
                                 "db_row": None, "query_row": None}
            for e in edges}


def test_splits_on_weight_threshold():
    edges = [_edge(1, 10), _edge(2, 20), _edge(3, 30)]
    weights = np.array([0.0, 0.0, 1.0, 0.02, 1.0])  # loop factors at 2, 3, 4
    history = _history(edges)
    inliers, rejected = split_edges_by_gnc_weight(
        edges, [2, 3, 4], weights, history)

    assert [e[0].id for e in inliers] == [1, 3]
    assert [e[0].id for e in rejected] == [2]
    assert history[(2, 20)]["action"] == "removed_by_pgo"
    assert history[(1, 10)]["action"] == "added_by_vpr"


def test_all_ones_keeps_every_edge():
    edges = [_edge(1, 10), _edge(2, 20)]
    weights = np.ones(4)
    inliers, rejected = split_edges_by_gnc_weight(
        edges, [2, 3], weights, _history(edges))
    assert len(inliers) == 2
    assert rejected == []


def test_threshold_is_an_inclusive_lower_bound():
    edges = [_edge(1, 10)]
    weights = np.array([PGO_INLIER_WEIGHT_THRESHOLD])
    inliers, rejected = split_edges_by_gnc_weight(
        edges, [0], weights, _history(edges))
    assert len(inliers) == 1, "weight exactly at the threshold counts as inlier"
