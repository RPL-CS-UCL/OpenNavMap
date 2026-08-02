"""save_vis_edge_history must count edges alive at each stage, including
edges later removed by PGO or the low-connectivity guard (those passed GV and
CCM, so hiding them from every panel misreports the funnel)."""
import sys
import types
from typing import Dict, Tuple
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np

# utils_map_merging imports the heavy pose-estimator stack at module level;
# stub it out so this viz test stays lightweight and environment-independent.
if 'estimator' not in sys.modules:
    _estimator = types.ModuleType('estimator')
    _estimator.get_estimator = MagicMock()
    _estimator.available_models = []
    _estimator_utils = types.ModuleType('estimator.utils')
    _estimator_utils.to_numpy = MagicMock()
    _estimator.utils = _estimator_utils
    sys.modules['estimator'] = _estimator
    sys.modules['estimator.utils'] = _estimator_utils


class _FakeNode:
    def __init__(self, node_id: int, xy: Tuple[float, float]) -> None:
        self.id = node_id
        self.trans_gt = np.array([xy[0], xy[1], 0.0])
        self.edges: Dict = {}

    def compute_gt_distance(self, other: "_FakeNode") -> Tuple[float, float]:
        return float(np.linalg.norm(self.trans_gt - other.trans_gt)), 0.0


class _FakeGraph:
    def __init__(self, nodes: Dict[int, _FakeNode]) -> None:
        self.nodes = nodes

    def get_node(self, node_id: int) -> _FakeNode:
        return self.nodes[node_id]


def _make_graphs(n: int) -> Tuple[_FakeGraph, _FakeGraph]:
    db = _FakeGraph({i: _FakeNode(i, (float(i), 0.0)) for i in range(n)})
    query = _FakeGraph({i: _FakeNode(i, (float(i), 1.0)) for i in range(n)})
    return db, query


def test_edge_counts_per_stage_include_late_removals(tmp_path) -> None:
    from utils_map_merging import save_vis_edge_history

    db, query = _make_graphs(5)
    edge_history = {
        (0, 0): {'action': 'removed_by_gv'},
        (1, 1): {'action': 'removed_by_ccm'},
        (2, 2): {'action': 'removed_by_pgo'},
        (3, 3): {'action': 'removed_by_pgo'},
        (4, 4): {'action': 'added_by_vpr'},
    }

    _, _, edge_counts = save_vis_edge_history(
        str(tmp_path), db, query, edge_history
    )

    # VPR: all 5; GV survivors: 4; CCM survivors: 3 (PGO removals happen after
    # CCM); final survivors: 1.
    assert edge_counts == [5, 4, 3, 1]
    assert (tmp_path / "edge_history.png").exists()


def test_empty_history_returns_four_zero_panels(tmp_path) -> None:
    from utils_map_merging import save_vis_edge_history

    db, query = _make_graphs(2)

    _, _, edge_counts = save_vis_edge_history(str(tmp_path), db, query, {})

    assert edge_counts == [0, 0, 0, 0]
