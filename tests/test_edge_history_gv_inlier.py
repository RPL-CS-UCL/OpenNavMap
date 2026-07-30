"""update_edge_history must carry the GV inlier count for later debugging."""
from typing import Dict, Tuple


def test_gv_inlier_recorded_on_first_insert():
    from map_merge_pipeline import update_edge_history

    history: Dict[Tuple[int, int], dict] = {}
    update_edge_history(history, (3, 7), action='added_by_vpr', db_row=3, query_row=7)
    update_edge_history(history, (3, 7), action='added_by_vpr', gv_inlier=142)

    assert history[(3, 7)]['gv_inlier'] == 142
    assert history[(3, 7)]['action'] == 'added_by_vpr'
    assert history[(3, 7)]['db_row'] == 3


def test_gv_inlier_survives_later_action_updates():
    from map_merge_pipeline import update_edge_history

    history: Dict[Tuple[int, int], dict] = {}
    update_edge_history(history, (1, 2), action='added_by_vpr', db_row=1, query_row=2)
    update_edge_history(history, (1, 2), action='removed_by_gv', gv_inlier=9)
    update_edge_history(history, (1, 2), action='removed_by_pgo')

    assert history[(1, 2)]['action'] == 'removed_by_pgo'
    assert history[(1, 2)]['gv_inlier'] == 9


def test_gv_inlier_defaults_to_minus_one_when_never_set():
    from map_merge_pipeline import update_edge_history

    history: Dict[Tuple[int, int], dict] = {}
    update_edge_history(history, (5, 5), action='added_by_vpr', db_row=5, query_row=5)

    assert history[(5, 5)].get('gv_inlier', -1) == -1
