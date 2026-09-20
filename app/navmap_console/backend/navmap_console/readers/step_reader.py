"""One merge_* step directory -> StepSummary (JSON) + scene arrays (scene.bin), cached under runs/<id>/cache/."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..models import LoopStats, StepRecord, StepSummary
from ..store import read_json, write_json_atomic
from .map_files import connected_components, read_edges, read_g2o_vertices, read_node_ids, read_poses_c2w
from .preds_files import read_cull_rows, read_edge_history, read_gnc_weights
from .scene_bundle import encode_scene_bundle

MOVE_THRESHOLD = 0.05  # metres; nodes the PGO moved further than this count as "moved"
INLIER_WEIGHT = 0.5  # PGO_INLIER_WEIGHT_THRESHOLD in python/utils_map_merging.py
NODE_NEW, NODE_CULLED, NODE_NOT_COVIS = 1, 2, 4
LOOP_ACCEPTED, LOOP_HIST, LOOP_OVERTURNED, LOOP_REJECTED_NEW = 1, 2, 4, 8
CACHE_VERSION = 1


def _edge_arrays(path: Path, n: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = read_edges(path)
    if len(edges) == 0:
        return np.zeros((0, 2), np.uint32), np.zeros(0, np.float32)
    idx = edges[:, :2].astype(np.int64)
    keep = (idx >= 0).all(axis=1) & (idx < n).all(axis=1)
    return idx[keep].astype(np.uint32), edges[keep, 2].astype(np.float32)


def _duration(record: StepRecord) -> Optional[float]:
    if not record.started_at or not record.finished_at:
        return None
    try:
        a, b = datetime.fromisoformat(record.started_at), datetime.fromisoformat(record.finished_at)
    except ValueError:
        return None
    return round((b - a).total_seconds(), 3)


def _node_steps(n: int, offsets: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Step index of the step that added each node id; ids below the first offset belong to the step before it."""
    if not offsets:
        return np.zeros(n, np.uint16)
    starts = np.asarray([o for o, _ in offsets], dtype=np.int64)
    steps = np.asarray([s for _, s in offsets], dtype=np.int64)
    slot = np.searchsorted(starts, np.arange(n), side="right") - 1
    before = max(int(steps[0]) - 1, 0)
    out = np.where(slot < 0, before, steps[np.clip(slot, 0, len(steps) - 1)])
    return out.astype(np.uint16)


def build_step(step_dir: Path, record: StepRecord,
               offsets: Sequence[Tuple[int, int]]) -> Tuple[StepSummary, Dict[str, np.ndarray]]:
    """`offsets` = [(id_offset, step_index), ...] ascending, for every step known so far (this one included)."""
    poses = read_poses_c2w(step_dir / "poses.txt")
    n = len(poses.names)
    id_offset = int(record.id_offset) if record.id_offset is not None else (offsets[-1][0] if offsets else 0)
    id_offset = min(max(id_offset, 0), n)
    odom, odom_w = _edge_arrays(step_dir / "edges_odom.txt", n)
    covis, covis_w = _edge_arrays(step_dir / "edges_covis.txt", n)
    trav, trav_w = _edge_arrays(step_dir / "edges_trav.txt", n)
    comp = connected_components(n, odom).astype(np.uint16)
    covis_ids: Set[int] = set(read_node_ids(step_dir / "intrinsics.txt").tolist())
    flags = np.zeros(n, np.uint8)
    flags[id_offset:] |= NODE_NEW
    if covis_ids:
        not_covis = np.fromiter((i not in covis_ids for i in range(n)), dtype=bool, count=n)
        flags[not_covis] |= NODE_NOT_COVIS
    preds = step_dir / "preds"
    for row in read_cull_rows(preds / "cull_node_info.txt", id_offset):
        if 0 <= row.node_id < n:
            flags[row.node_id] |= NODE_CULLED
    # PGO-before positions: initial_pose_graph.g2o vertices are camera-to-world (verified on the reference run)
    pos_pre = poses.pos.copy()
    vertices = read_g2o_vertices(preds / "initial_pose_graph.g2o")
    has_pre = bool(vertices)
    for vid, v in vertices.items():
        if 0 <= vid < n:
            pos_pre[vid] = v[:3]
    disp = np.linalg.norm(poses.pos - pos_pre, axis=1) if has_pre and n else np.zeros(n, np.float32)
    gnc = read_gnc_weights(preds / "gnc_weights.txt")
    loops = LoopStats()
    loop_idx = np.zeros((len(gnc), 2), np.uint32)
    loop_conf = np.zeros(len(gnc), np.float32)
    loop_weight = np.zeros(len(gnc), np.float32)
    loop_terr = np.zeros(len(gnc), np.float32)
    loop_rerr = np.zeros(len(gnc), np.float32)
    loop_flags = np.zeros(len(gnc), np.uint8)
    for i, g in enumerate(gnc):
        loop_idx[i] = (g.db, g.query)
        loop_conf[i], loop_weight[i], loop_terr[i], loop_rerr[i] = g.conf, g.weight, g.trans_err, g.rot_err
        accepted = g.weight >= INLIER_WEIGHT
        f = LOOP_ACCEPTED if accepted else 0
        loops.total += 1
        if g.origin == "hist":
            f |= LOOP_HIST
            loops.hist += 1
            if not accepted:
                f |= LOOP_OVERTURNED
                loops.overturned_hist += 1
        else:
            loops.new += 1
            if not accepted:
                f |= LOOP_REJECTED_NEW
                loops.rejected_new += 1
        if accepted:
            loops.accepted += 1
        loop_flags[i] = f
    keep = (loop_idx[:, 0] < n) & (loop_idx[:, 1] < n)
    history = read_edge_history(preds / "edge_history.txt", id_offset)
    summary = StepSummary(
        index=record.index, dir_name=step_dir.name, session_id=record.session_id, status=record.status,
        id_offset=id_offset, num_nodes=n, num_covis_nodes=len(covis_ids) if covis_ids else n,
        num_new=n - id_offset, num_culled=int((flags & NODE_CULLED).astype(bool).sum()),
        component_sizes=np.bincount(comp).tolist() if n else [],
        edge_counts={"odom": int(len(odom)), "covis": int(len(covis)), "trav": int(len(trav))},
        history=history.counts, precision=history.precision, recall=history.recall, loops=loops,
        pgo_error_initial=record.pgo_error_initial, pgo_error_final=record.pgo_error_final,
        max_displacement=float(disp.max()) if has_pre and n else 0.0,
        num_moved=int((disp > MOVE_THRESHOLD).sum()) if has_pre else 0,
        duration_s=_duration(record),
        has_dmatrix=(preds / "D_matrix.npy").is_file() and (preds / "D_matrix_axes.json").is_file(),
        has_pre_pgo=has_pre)
    arrays: Dict[str, np.ndarray] = {
        "node_id": np.arange(n, dtype=np.uint32), "node_pos": poses.pos, "node_pos_pre": pos_pre.astype(np.float32),
        "node_quat": poses.quat, "node_step": _node_steps(n, offsets), "node_comp": comp, "node_flags": flags,
        "edge_odom": odom, "edge_odom_w": odom_w, "edge_covis": covis, "edge_covis_w": covis_w,
        "edge_trav": trav, "edge_trav_w": trav_w,
        "loop_idx": loop_idx[keep], "loop_conf": loop_conf[keep], "loop_weight": loop_weight[keep],
        "loop_flags": loop_flags[keep], "loop_terr": loop_terr[keep], "loop_rerr": loop_rerr[keep],
    }
    return summary, arrays


class StepCache:
    """cache/steps/<dir_name>.summary.json + .scene.bin, keyed on poses.txt mtime/size and the step record."""

    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir / "cache" / "steps"

    @staticmethod
    def _source(step_dir: Path, record: StepRecord, offsets: Sequence[Tuple[int, int]]) -> str:
        st = (step_dir / "poses.txt").stat()
        extra = hashlib.sha1(json.dumps([record.model_dump(), list(map(list, offsets))], sort_keys=True)
                             .encode()).hexdigest()[:12]
        return f"{st.st_mtime_ns}:{st.st_size}:{extra}"

    def get(self, step_dir: Path, record: StepRecord,
            offsets: Sequence[Tuple[int, int]]) -> Tuple[StepSummary, bytes]:
        key = self._source(step_dir, record, offsets)
        summary_path = self.dir / f"{step_dir.name}.summary.json"
        scene_path = self.dir / f"{step_dir.name}.scene.bin"
        if summary_path.is_file() and scene_path.is_file():
            meta = read_json(summary_path)
            if meta.get("version") == CACHE_VERSION and meta.get("source") == key:
                return StepSummary.model_validate(meta["summary"]), scene_path.read_bytes()
        summary, arrays = build_step(step_dir, record, offsets)
        data = encode_scene_bundle(arrays)
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = scene_path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, scene_path)
        write_json_atomic(summary_path, {"version": CACHE_VERSION, "source": key, "summary": summary.model_dump()})
        return summary, data
