"""Read-only views over a run's output/ for the viewer (spec §5.6). No torch, no subprocesses."""
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..config import Settings
from ..models import StepRecord, StepSummary
from ..readers import step_reader as sr
from ..readers.events import EventIndex
from ..readers.image_index import ImageIndex, pair_image, thumbnail
from ..readers.map_files import read_poses_c2w, read_timestamps
from ..readers.preds_files import read_cull_rows, read_dmatrix, read_edge_history, read_gnc_weights
from .runs import RunService


class ResultService:
    def __init__(self, settings: Settings, runs: RunService) -> None:
        self.settings, self.runs = settings, runs
        self._events: Dict[str, EventIndex] = {}
        self._images: Dict[str, Tuple[int, ImageIndex]] = {}

    # ---- helpers -----------------------------------------------------------
    def _base(self, rid: str, run_id: str, prefix: str = "/api") -> str:
        return f"{prefix}/regions/{rid}/runs/{run_id}"

    def _done_steps(self, rid: str, run_id: str) -> List[StepRecord]:
        self.runs.runs.get(rid, run_id)  # KeyError -> 404
        steps = [s for s in self.runs.runs.read_steps(rid, run_id) if s.status == "done" and s.dir_name]
        return sorted(steps, key=lambda s: s.index)

    @staticmethod
    def _offsets(steps: Sequence[StepRecord]) -> List[Tuple[int, int]]:
        return [(int(s.id_offset or 0), s.index) for s in steps]

    def _step(self, rid: str, run_id: str, k: int) -> Tuple[StepRecord, Path, List[Tuple[int, int]]]:
        steps = self._done_steps(rid, run_id)
        for s in steps:
            if s.index == k:
                step_dir = self.runs.runs.run_dir(rid, run_id) / "output" / s.dir_name
                if not (step_dir / "poses.txt").is_file():
                    raise KeyError(f"step {k} directory is incomplete")
                return s, step_dir, self._offsets([x for x in steps if x.index <= k])
        raise KeyError(f"step {k} not found")

    def _cache(self, rid: str, run_id: str) -> sr.StepCache:
        return sr.StepCache(self.runs.runs.run_dir(rid, run_id))

    def _image_index(self, rid: str, run_id: str) -> ImageIndex:
        sources = self.runs.image_sources(rid, run_id)
        key = f"{rid}/{run_id}"
        cached = self._images.get(key)
        if cached is None or cached[0] != len(sources):
            cached = (len(sources), ImageIndex(sources))
            self._images[key] = cached
        return cached[1]

    def _thumb_dir(self, rid: str, run_id: str) -> Path:
        return self.runs.runs.run_dir(rid, run_id) / "cache" / "thumbs"

    # ---- endpoints ---------------------------------------------------------
    def summaries(self, rid: str, run_id: str) -> List[StepSummary]:
        steps = self._done_steps(rid, run_id)
        cache = self._cache(rid, run_id)
        run_dir = self.runs.runs.run_dir(rid, run_id)
        out: List[StepSummary] = []
        for i, s in enumerate(steps):
            step_dir = run_dir / "output" / s.dir_name
            if (step_dir / "poses.txt").is_file():
                out.append(cache.get(step_dir, s, self._offsets(steps[:i + 1]))[0])
        return out

    def scene(self, rid: str, run_id: str, k: int) -> bytes:
        rec, step_dir, offsets = self._step(rid, run_id, k)
        return self._cache(rid, run_id).get(step_dir, rec, offsets)[1]

    def _dmatrix(self, rid: str, run_id: str, k: int) -> Tuple[StepRecord, Path, Tuple[np.ndarray, List[int], List[int]]]:
        rec, step_dir, _ = self._step(rid, run_id, k)
        loaded = read_dmatrix(step_dir / "preds")
        if loaded is None:
            raise FileNotFoundError(f"step {k} has no preds/D_matrix.npy")
        return rec, step_dir, loaded

    def dmatrix_json(self, rid: str, run_id: str, k: int) -> Dict[str, Any]:
        rec, step_dir, (matrix, row_ids, col_ids) = self._dmatrix(rid, run_id, k)
        preds = step_dir / "preds"
        id_offset = int(rec.id_offset or 0)
        history = read_edge_history(preds / "edge_history.txt", id_offset)
        finite = matrix[np.isfinite(matrix)]
        return {
            "rows": "db", "cols": "query",
            "row_node_ids": row_ids,
            "col_node_ids": col_ids,
            "vmin": float(finite.min()) if finite.size else 0.0, "vmax": float(finite.max()) if finite.size else 1.0,
            "candidates": [{"db": r.db, "query": r.query, "stage": r.action, "gv_inliers": r.gv_inliers}
                           for r in history.rows],
            "factors": [{"db": g.db, "query": g.query, "weight": g.weight, "conf": g.conf,
                         "accepted": bool(g.weight >= sr.INLIER_WEIGHT), "origin": g.origin}
                        for g in read_gnc_weights(preds / "gnc_weights.txt")],
        }

    def dmatrix_png(self, rid: str, run_id: str, k: int) -> bytes:
        _, _, (matrix, _, _) = self._dmatrix(rid, run_id, k)
        m = matrix.astype(np.float32)
        finite = np.isfinite(m)
        lo, hi = (float(m[finite].min()), float(m[finite].max())) if finite.any() else (0.0, 1.0)
        scale = (hi - lo) or 1.0
        gray = np.where(finite, 255.0 * (1.0 - (m - lo) / scale), 0.0)  # small distance = bright
        buf = io.BytesIO()
        Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), "L").save(buf, "PNG")
        return buf.getvalue()

    def _kf_vis(self, preds: Path) -> Dict[Tuple[int, int], str]:
        """kf_rejection_query_{query_local}_{db}_{prob}.jpg -> (query_local, db) -> file name."""
        out: Dict[Tuple[int, int], str] = {}
        d = preds / "kf_vis"
        if d.is_dir():
            for p in d.iterdir():
                parts = p.stem.split("_")
                if len(parts) >= 5 and parts[:3] == ["kf", "rejection", "query"]:
                    try:
                        out[(int(parts[3]), int(parts[4]))] = p.name
                    except ValueError:
                        continue
        return out

    def culling(self, rid: str, run_id: str, k: int) -> Dict[str, Any]:
        rec, step_dir, _ = self._step(rid, run_id, k)
        preds = step_dir / "preds"
        id_offset = int(rec.id_offset or 0)
        base = self._base(rid, run_id)
        vis = self._kf_vis(preds)

        def row(r) -> Dict[str, Any]:
            local_q = (r.node_id - id_offset) if r.kind == "query" else (r.other - id_offset if r.other is not None else -1)
            db = r.other if r.kind == "query" else r.node_id
            name = vis.get((local_q, db)) if db is not None else None
            return {"node_id": r.node_id, "kind": r.kind, "other": r.other, "prob": r.prob, "method": r.method,
                    "detail": r.detail, "image_url": f"{base}/nodes/{r.node_id}/image",
                    "other_image_url": f"{base}/nodes/{r.other}/image" if r.other is not None else None,
                    "vis_url": f"{base}/steps/{k}/preds/kf_vis/{name}" if name else None}

        return {"culled": [row(r) for r in read_cull_rows(preds / "cull_node_info.txt", id_offset)],
                "kept": [row(r) for r in read_cull_rows(preds / "not_cull_node_info.txt", id_offset)]}

    def node_detail(self, rid: str, run_id: str, k: int, nid: int) -> Dict[str, Any]:
        rec, step_dir, offsets = self._step(rid, run_id, k)
        poses = read_poses_c2w(step_dir / "poses.txt")
        n = len(poses.names)
        if not 0 <= nid < n:
            raise KeyError(f"node {nid} not in step {k} (has {n} nodes)")
        summary, arrays = self._cache_arrays(rid, run_id, rec, step_dir, offsets)
        step_of = int(arrays["node_step"][nid])
        session = next((s.session_id for s in self._done_steps(rid, run_id) if s.index == step_of), "")
        degree = {name: int(((arrays[key][:, 0] == nid) | (arrays[key][:, 1] == nid)).sum())
                  for name, key in (("odom", "edge_odom"), ("covis", "edge_covis"), ("trav", "edge_trav"))}
        gt: Optional[List[float]] = None
        gt_path = step_dir / "poses_abs_gt.txt"
        if gt_path.is_file():
            g = read_poses_c2w(gt_path)
            if nid < len(g.names) and bool(g.valid[nid]):
                gt = [float(v) for v in g.pos[nid]]
        stamps = read_timestamps(step_dir / "timestamps.txt")
        cull = next((r for r in read_cull_rows(step_dir / "preds" / "cull_node_info.txt", int(rec.id_offset or 0))
                     if r.node_id == nid), None)
        loops = []
        for i in range(len(arrays["loop_idx"])):
            a, b = int(arrays["loop_idx"][i, 0]), int(arrays["loop_idx"][i, 1])
            if nid in (a, b):
                f = int(arrays["loop_flags"][i])
                loops.append({"other": b if a == nid else a, "weight": float(arrays["loop_weight"][i]),
                              "conf": float(arrays["loop_conf"][i]), "accepted": bool(f & sr.LOOP_ACCEPTED),
                              "origin": "hist" if f & sr.LOOP_HIST else "new"})
        base = self._base(rid, run_id)
        return {"node_id": nid, "step": step_of, "session_id": session, "frame": poses.names[nid],
                "timestamp": stamps.get(poses.names[nid]), "pos": [float(v) for v in poses.pos[nid]],
                "quat": [float(v) for v in poses.quat[nid]], "pos_pre": [float(v) for v in arrays["node_pos_pre"][nid]],
                "gt": gt, "degree": degree, "flags": int(arrays["node_flags"][nid]),
                "image_url": f"{base}/nodes/{nid}/image",
                "cull": None if cull is None else {"other": cull.other, "prob": cull.prob, "method": cull.method,
                                                   "detail": cull.detail},
                "loops": loops}

    def _cache_arrays(self, rid: str, run_id: str, rec: StepRecord, step_dir: Path,
                      offsets: List[Tuple[int, int]]) -> Tuple[StepSummary, Dict[str, np.ndarray]]:
        from ..readers.scene_bundle import decode_scene_bundle

        summary, data = self._cache(rid, run_id).get(step_dir, rec, offsets)
        return summary, decode_scene_bundle(data)

    def node_image(self, rid: str, run_id: str, nid: int, width: int) -> Path:
        src = self._image_index(rid, run_id).path(nid)
        if src is None:
            raise KeyError(f"no image for node {nid}")
        return thumbnail(src, width, self._thumb_dir(rid, run_id))

    def pair_image(self, rid: str, run_id: str, a: int, b: int, width: int) -> Path:
        idx = self._image_index(rid, run_id)
        left, right = idx.path(a), idx.path(b)
        if left is None or right is None:
            raise KeyError(f"no image for node pair {a},{b}")
        return pair_image(left, right, width, self._thumb_dir(rid, run_id))

    def preds_file(self, rid: str, run_id: str, k: int, name: str) -> Path:
        _, step_dir, _ = self._step(rid, run_id, k)
        preds = (step_dir / "preds").resolve()
        target = (preds / name).resolve()
        if preds != target and preds not in target.parents:
            raise ValueError("path escapes preds/")
        if not target.is_file():
            raise FileNotFoundError(name)
        return target

    def events(self, rid: str, run_id: str, step: Optional[int], types: Optional[Sequence[str]], limit: int,
               snapshots: bool) -> List[Dict[str, Any]]:
        self.runs.runs.get(rid, run_id)
        key = f"{rid}/{run_id}"
        if key not in self._events:
            self._events[key] = EventIndex(self.runs.runs.run_dir(rid, run_id) / "output" / "rerun_viz"
                                           / "demo_events.jsonl")
        idx = self._events[key]
        idx.refresh()
        return idx.read(step, types, limit, snapshots)
