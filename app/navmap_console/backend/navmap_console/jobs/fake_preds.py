"""preds/ files for the fake pipeline, in the exact formats python/map_merge_pipeline.py writes.

Used by tests and by frontend development (NAVMAP_CONSOLE_FAKE_PIPELINE=1) so every panel has data.
Numbers are deterministic per `seed`. Exactly one new loop edge per step is accepted (weight >= 0.5),
so STEP_DONE registry=k stays true for the k-th step.
"""
import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def _quat_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _c2w_rows(poses_txt: Path) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """(node id, camera centre, c2w quaternion xyzw) per poses.txt row (rows are w2c)."""
    rows = []
    for i, line in enumerate(l for l in poses_txt.read_text().splitlines() if l.strip()):
        vals = [float(v) for v in line.split()[1:8]]
        rot_w2c = _quat_wxyz_to_matrix(vals[:4])
        centre = -rot_w2c.T @ np.asarray(vals[4:7])
        w, x, y, z = vals[:4]
        rows.append((i, centre, np.array([-x, -y, -z, w])))  # inverse rotation = conjugate
    return rows


def write_fake_preds(step_dir: Path, id_offset: int, n_nodes: int, hist: Sequence[Tuple[int, int]],
                     seed: int) -> Tuple[int, int]:
    preds = step_dir / "preds"
    preds.mkdir(exist_ok=True)
    rng = random.Random(seed)
    rows = _c2w_rows(step_dir / "poses.txt")
    g2o = []
    for i, centre, q in rows:
        noisy = centre + np.array([rng.uniform(-0.02, 0.02) for _ in range(3)])
        g2o.append("VERTEX_SE3:QUAT %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f" % (i, *noisy, *q))
    if id_offset <= 0 or n_nodes - id_offset < 3:
        (preds / "initial_pose_graph.g2o").write_text("\n".join(g2o) + "\n")
        return -1, -1
    db_ids = list(range(id_offset))
    q_ids = list(range(id_offset, n_nodes))
    new_pairs = [(rng.choice(db_ids), q) for q in rng.sample(q_ids, 3)]
    weights = [0.91, 0.23, 0.08]
    confs = [0.82, 0.41, 0.27]
    accepted = new_pairs[0]
    gnc = ["# pgo_robust=gnc_gm barc_prob=0.99", "# db_id,query_id are merged global node ids",
           "# db_id,query_id,weight,conf,trans_err,rot_err,origin"]
    for (a, b), w, c in zip(new_pairs, weights, confs):
        gnc.append("%d,%d,%.6f,%.3f,%.3f,%.3f,new" % (a, b, w, c, rng.uniform(0.01, 0.5), rng.uniform(0.1, 3.0)))
    for a, b in hist:
        gnc.append("%d,%d,%.6f,nan,nan,nan,hist" % (a, b, 0.95))
    (preds / "gnc_weights.txt").write_text("\n".join(gnc) + "\n")
    for a, b in new_pairs + list(hist):
        g2o.append("EDGE_SE3:QUAT %d %d 0 0 0 0 0 0 1 " % (a, b) + " ".join(["1"] * 21))
    (preds / "initial_pose_graph.g2o").write_text("\n".join(g2o) + "\n")
    extra = [(rng.choice(db_ids), q) for q in rng.sample(q_ids, 3)]
    actions = ["retained", "removed_by_pgo", "removed_by_pgo", "removed_by_gv", "removed_by_gv", "removed_by_ccm"]
    hist_lines = ["Number of edges added by VPR: 6", "Number of edges removed by GV: 2 (33.33%)",
                  "Number of edges removed by CCM: 1 (16.67%)", "Number of edges removed by PGO: 2 (33.33%)",
                  "Number of edges removed by low connectivity: 0 (0.00%)", "Number of edges retained: 1 (16.67%)",
                  "Precision: 0.50,0.50,1.00", "Recall: 0.40,0.40,0.20"]
    for (a, b), action in zip(new_pairs + extra, actions):
        hist_lines.append("%d,%d,%s,gv_inlier: %d" % (a, b - id_offset, action, rng.randint(20, 900)))
    (preds / "edge_history.txt").write_text("\n".join(hist_lines) + "\n")
    lloc = ["%d,%d,Conf: %.3f - Error: %.3f [m] and %.3f [deg]" % (a, b - id_offset, c, rng.uniform(0.01, 0.5),
                                                                    rng.uniform(0.1, 3.0))
            for (a, b), c in zip(new_pairs, confs)]
    (preds / "lloc_history.txt").write_text("\n".join(lloc) + "\n")
    head = ["# Ablation Study Configuration:", "# IQA: True", "# IG: True", "# TD: True"]
    culled_local = q_ids[-1] - id_offset
    (preds / "cull_node_info.txt").write_text("\n".join(head + [
        "node_id,type,replaced_by,prob,method,prob_str",
        "%d,query,%d,0.330,culled_by_forward,Q: 41.90, G: 0.31, P: 0.33" % (culled_local, accepted[0])]) + "\n")
    (preds / "not_cull_node_info.txt").write_text("\n".join(head + [
        "node_id,type,compared_to,prob,method,prob_str",
        "%d,query,%d,0.710,not_culled_by_backward,Q: 12.00, G: 0.80, P: 0.71" % (q_ids[0] - id_offset,
                                                                                  accepted[0])]) + "\n")
    matrix = np.array([[rng.uniform(0.4, 1.0) for _ in q_ids] for _ in db_ids], dtype=np.float32)
    for a, b in new_pairs:
        matrix[a, b - id_offset] = 0.05
    np.save(preds / "D_matrix.npy", matrix.astype(np.float16))
    (preds / "D_matrix_axes.json").write_text(json.dumps(
        {"rows": "db", "row_node_ids": db_ids, "cols": "query", "col_node_ids": q_ids}))
    reg = ["# a_id,b_id,conf,first_step,reject_count,last_weight,tx,ty,tz,qx,qy,qz,qw"]
    for k, (a, b) in enumerate(list(hist) + [accepted]):
        reg.append("%d,%d,0.800,%d,0,0.950000,0.000000000,0.000000000,0.000000000,"
                   "0.000000000,0.000000000,0.000000000,1.000000000" % (a, b, k))
    (preds / "loop_registry.txt").write_text("\n".join(reg) + "\n")
    (preds / "kf_vis").mkdir(exist_ok=True)
    return accepted
