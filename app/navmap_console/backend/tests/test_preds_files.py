# tests/test_preds_files.py
import json
from pathlib import Path

import numpy as np

GNC = ("# pgo_robust=gnc_gm barc_prob=0.99\n# db_id,query_id are merged global node ids\n"
       "# db_id,query_id,weight,conf,trans_err,rot_err,origin\n"
       "4,48,0.818200,nan,nan,nan,hist\n4760,4782,0.004893,0.632,1.614,1.267,new\n")
HISTORY = ("Number of edges added by VPR: 109\nNumber of edges removed by GV: 65 (59.63%)\n"
           "Number of edges removed by CCM: 21 (19.27%)\nNumber of edges removed by PGO: 16 (14.68%)\n"
           "Number of edges removed by low connectivity: 0 (0.00%)\nNumber of edges retained: 7 (6.42%)\n"
           "Precision: 0.16,0.58,1.00\nRecall: 0.44,0.44,0.22\n"
           "121,0,removed_by_pgo,gv_inlier: 731\n130,5,removed_by_gv,gv_inlier: 12\n140,7,retained,gv_inlier: 900\n")
CULL = ("# Ablation Study Configuration:\n# IQA: True\n# IG: True\n# TD: True\n"
        "node_id,type,replaced_by,prob,method,prob_str\n"
        "55,query,,0.490,culled_by_iqa,IQA: 18.61. P_Q: 0.49\n"
        "4,query,4764,0.330,culled_by_forward,Q: 41.90, G: 0.31, P: 0.33\n"
        "4700,db,9,0.200,culled_by_backward,Q: 1.0, G: 0.2\n")


def test_read_gnc_weights(tmp_path: Path):
    from navmap_console.readers.preds_files import read_gnc_weights

    (tmp_path / "gnc_weights.txt").write_text(GNC)
    rows = read_gnc_weights(tmp_path / "gnc_weights.txt")
    assert [(r.db, r.query, r.origin) for r in rows] == [(4, 48, "hist"), (4760, 4782, "new")]
    assert rows[0].weight == 0.8182 and np.isnan(rows[0].conf) and rows[1].trans_err == 1.614
    assert read_gnc_weights(tmp_path / "missing.txt") == []


def test_read_edge_history_offsets_query_ids(tmp_path: Path):
    from navmap_console.readers.preds_files import read_edge_history

    (tmp_path / "edge_history.txt").write_text(HISTORY)
    h = read_edge_history(tmp_path / "edge_history.txt", id_offset=4782)
    assert h.counts == {"vpr": 109, "gv": 65, "ccm": 21, "pgo": 16, "low_connectivity": 0, "retained": 7}
    assert h.precision == [0.16, 0.58, 1.0] and h.recall == [0.44, 0.44, 0.22]
    assert [(r.db, r.query, r.action, r.gv_inliers) for r in h.rows] == [
        (121, 4782, "removed_by_pgo", 731), (130, 4787, "removed_by_gv", 12), (140, 4789, "retained", 900)]
    empty = read_edge_history(tmp_path / "missing.txt", 0)
    assert empty.rows == [] and empty.counts["vpr"] == 0


def test_read_lloc_history(tmp_path: Path):
    from navmap_console.readers.preds_files import read_lloc_history

    (tmp_path / "lloc_history.txt").write_text(
        "113,107,Conf: 2.709 - Error: 0.042 [m] and 0.391 [deg]\n12,3,Conf: 1.1 - Error: nan [m] and nan [deg]\n")
    rows = read_lloc_history(tmp_path / "lloc_history.txt", id_offset=100)
    assert (rows[0].db, rows[0].query, rows[0].conf, rows[0].trans_err, rows[0].rot_err) == (113, 207, 2.709, 0.042, 0.391)
    assert rows[1].query == 103 and np.isnan(rows[1].trans_err)


def test_read_cull_rows_makes_ids_global(tmp_path: Path):
    from navmap_console.readers.preds_files import read_cull_rows

    (tmp_path / "cull_node_info.txt").write_text(CULL)
    rows = read_cull_rows(tmp_path / "cull_node_info.txt", id_offset=4782)
    assert [(r.node_id, r.kind, r.other, r.method) for r in rows] == [
        (4837, "query", None, "culled_by_iqa"), (4786, "query", 4764, "culled_by_forward"),
        (4700, "db", 4791, "culled_by_backward")]
    assert rows[1].prob == 0.33 and rows[1].detail == "Q: 41.90, G: 0.31, P: 0.33"
    assert read_cull_rows(tmp_path / "missing.txt", 0) == []


def test_read_dmatrix(tmp_path: Path):
    from navmap_console.readers.preds_files import read_dmatrix

    assert read_dmatrix(tmp_path) is None
    np.save(tmp_path / "D_matrix.npy", np.arange(6, dtype=np.float16).reshape(2, 3))
    (tmp_path / "D_matrix_axes.json").write_text(json.dumps(
        {"rows": "db", "row_node_ids": [0, 1], "cols": "query", "col_node_ids": [10, 11, 12]}))
    mat, rows, cols = read_dmatrix(tmp_path)
    assert mat.dtype == np.float32 and mat.shape == (2, 3) and rows == [0, 1] and cols == [10, 11, 12]
