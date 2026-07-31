"""A submap connected by fewer than min_edges loop edges must not be merged.

With a single loop edge PGO can zero the residual by rigidly moving the whole
submap, so no robust loss can ever classify that edge as an outlier (observed
at merge step 40: a 176.7-degree flip edge kept weight 1.0 even with a fully
annealing GNC). The only safe move is to defer the merge.
"""
import numpy as np
import pytest

pytest.importorskip("gtsam")

from map_merge_pipeline import defer_low_connectivity_edges


class _Node:
	def __init__(self, node_id: int) -> None:
		self.id = node_id


def _edge(db_id: int, query_id: int):
	return (_Node(db_id), _Node(query_id), np.eye(4), 0.8, 0.5)


def _history(edges):
	return {(e[0].id, e[1].id): {"action": "added_by_vpr",
	                             "db_row": None, "query_row": None}
	        for e in edges}


def test_single_edge_is_deferred() -> None:
	edges = [_edge(1, 10)]
	history = _history(edges)
	kept, deferred = defer_low_connectivity_edges(edges, 2, history)
	assert kept == []
	assert [e[0].id for e in deferred] == [1]
	assert history[(1, 10)]["action"] == "removed_by_low_connectivity"


def test_enough_edges_pass_through_untouched() -> None:
	edges = [_edge(1, 10), _edge(2, 20)]
	history = _history(edges)
	kept, deferred = defer_low_connectivity_edges(edges, 2, history)
	assert kept == edges
	assert deferred == []
	assert history[(1, 10)]["action"] == "added_by_vpr"


def test_no_edges_is_a_noop() -> None:
	kept, deferred = defer_low_connectivity_edges([], 2, {})
	assert kept == [] and deferred == []


def test_min_edges_one_disables_the_guard() -> None:
	edges = [_edge(1, 10)]
	kept, deferred = defer_low_connectivity_edges(edges, 1, _history(edges))
	assert kept == edges and deferred == []
