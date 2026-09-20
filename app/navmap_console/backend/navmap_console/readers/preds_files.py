"""Parsers for merge_*/preds/*.txt written by python/map_merge_pipeline.py.

Formats were verified against the s00000 reference run. Convention: the db column is a global node
id, the query column is a submap-local id (+ id_offset); gnc_weights.txt is the exception (both global).
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _lines(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


@dataclass
class GncRow:
    db: int
    query: int
    weight: float
    conf: float
    trans_err: float
    rot_err: float
    origin: str  # "hist" (carried over from an earlier step) or "new"


def read_gnc_weights(path: Path) -> List[GncRow]:
    rows: List[GncRow] = []
    for line in _lines(path):
        if line.startswith("#"):
            continue
        p = [v.strip() for v in line.split(",")]
        if len(p) < 7:
            continue
        rows.append(GncRow(int(p[0]), int(p[1]), _float(p[2]), _float(p[3]), _float(p[4]), _float(p[5]), p[6]))
    return rows


@dataclass
class CandidateRow:
    db: int
    query: int
    action: str  # removed_by_gv | removed_by_ccm | removed_by_pgo | anything else = kept
    gv_inliers: int


@dataclass
class EdgeHistory:
    counts: Dict[str, int]
    precision: List[float]
    recall: List[float]
    rows: List[CandidateRow]


_COUNT_KEYS = (("by VPR", "vpr"), ("by GV", "gv"), ("by CCM", "ccm"), ("by PGO", "pgo"),
               ("low connectivity", "low_connectivity"), ("retained", "retained"))
_FIRST_INT = re.compile(r":\s*(\d+)")


def _float_list(line: str) -> List[float]:
    return [_float(v) for v in line.split(":", 1)[1].split(",") if v.strip()]


def read_edge_history(path: Path, id_offset: int) -> EdgeHistory:
    counts = {key: 0 for _, key in _COUNT_KEYS}
    precision: List[float] = []
    recall: List[float] = []
    rows: List[CandidateRow] = []
    for line in _lines(path):
        if line.startswith("Number of edges"):
            m = _FIRST_INT.search(line)
            for needle, key in _COUNT_KEYS:
                if needle in line and m:
                    counts[key] = int(m.group(1))
                    break
        elif line.startswith("Precision:"):
            precision = _float_list(line)
        elif line.startswith("Recall:"):
            recall = _float_list(line)
        else:
            head, sep, tail = line.partition("gv_inlier:")
            p = [v.strip() for v in head.split(",") if v.strip()]
            if len(p) < 3 or not p[0].isdigit() or not p[1].isdigit():
                continue
            inliers = int(tail.strip()) if sep and tail.strip().isdigit() else 0
            rows.append(CandidateRow(int(p[0]), int(p[1]) + id_offset, p[2], inliers))
    return EdgeHistory(counts, precision, recall, rows)


@dataclass
class LlocRow:
    db: int
    query: int
    conf: float
    trans_err: float
    rot_err: float


_LLOC = re.compile(r"^(\d+),(\d+),Conf:\s*(\S+)\s*-\s*Error:\s*(\S+)\s*\[m\]\s*and\s*(\S+)\s*\[deg\]")


def read_lloc_history(path: Path, id_offset: int) -> List[LlocRow]:
    rows: List[LlocRow] = []
    for line in _lines(path):
        m = _LLOC.match(line)
        if m:
            rows.append(LlocRow(int(m.group(1)), int(m.group(2)) + id_offset, _float(m.group(3)),
                                _float(m.group(4)), _float(m.group(5))))
    return rows


@dataclass
class CullRow:
    node_id: int  # global
    kind: str  # "query" or "db"
    other: Optional[int]  # global id of the frame it was replaced by / compared to
    prob: float
    method: str  # culled_by_iqa | culled_by_forward | culled_by_backward | not_culled_by_*
    detail: str  # free-text prob_str (contains commas)


def read_cull_rows(path: Path, id_offset: int) -> List[CullRow]:
    """cull_node_info.txt and not_cull_node_info.txt share this layout; ids come back global."""
    rows: List[CullRow] = []
    for line in _lines(path):
        if line.startswith("#") or line.startswith("node_id,"):
            continue
        p = [v.strip() for v in line.split(",", 5)]
        if len(p) < 5 or not p[0].isdigit():
            continue
        node, kind = int(p[0]), p[1]
        other = int(p[2]) if p[2].isdigit() else None
        if kind == "query":
            node += id_offset
        elif other is not None:
            other += id_offset
        rows.append(CullRow(node, kind, other, _float(p[3]), p[4], p[5] if len(p) > 5 else ""))
    return rows


def read_dmatrix(preds_dir: Path) -> Optional[Tuple[np.ndarray, List[int], List[int]]]:
    """preds/D_matrix.npy (float16, rows = db nodes, cols = query nodes) + D_matrix_axes.json."""
    npy, axes = preds_dir / "D_matrix.npy", preds_dir / "D_matrix_axes.json"
    if not npy.is_file() or not axes.is_file():
        return None
    meta = json.loads(axes.read_text())
    matrix = np.load(npy).astype(np.float32)
    return matrix, [int(v) for v in meta["row_node_ids"]], [int(v) for v in meta["col_node_ids"]]
