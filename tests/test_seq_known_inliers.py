"""Sequentially-adjacent loop edges (endpoint capture times within a small
gap) are backed by odometry continuity and must be exempt from GNC outlier
classification: at merge step 42 GNC-GM rejected 19/20 such edges whose GT
errors were only 0.1-4.7 m, collapsing the merge to a single-edge mount."""
from typing import Dict, Optional


def _time_lookup(times: Dict[int, Optional[float]]):
    def get_node_time(node_id: int) -> Optional[float]:
        return times.get(node_id)
    return get_node_time


def test_selects_only_edges_within_time_gap():
    from map_merge_pipeline import select_seq_inlier_factor_indices

    times = {1: 100.0, 2: 150.0, 3: 5000.0, 4: 160.0}
    # (factor_idx, (id_a, id_b)): dt=50 -> in, dt=4900 -> out, dt=60 (== gap) -> in
    indices = [7, 8, 9]
    keys = [(1, 2), (1, 3), (1, 4)]

    selected = select_seq_inlier_factor_indices(
        indices, keys, _time_lookup(times), max_time_gap=60.0
    )

    assert selected == [7, 9]


def test_zero_gap_disables_selection():
    from map_merge_pipeline import select_seq_inlier_factor_indices

    selected = select_seq_inlier_factor_indices(
        [3], [(1, 2)], _time_lookup({1: 0.0, 2: 1.0}), max_time_gap=0.0
    )

    assert selected == []


def test_missing_timestamps_are_never_selected():
    from map_merge_pipeline import select_seq_inlier_factor_indices

    times = {1: 100.0, 2: None}
    selected = select_seq_inlier_factor_indices(
        [5, 6], [(1, 2), (3, 1)], _time_lookup(times), max_time_gap=60.0
    )

    assert selected == []
