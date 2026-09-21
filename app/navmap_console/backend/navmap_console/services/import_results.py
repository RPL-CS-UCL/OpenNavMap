"""Turn a scripts/run_map_merging.sh result directory into an `imported` run (spec §5.3)."""
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..models import ImportRequest, RegionHead, Run, StepRecord, now_iso
from ..readers.map_files import connected_components, read_edges, read_node_ids
from ..store import new_id
from .ingest import resolve_allowed_path, session_from_dir
from .runs import RunService

_INDEXED = re.compile(r"^merge_(\d{3,})_(.+)$")
_CUMULATIVE = re.compile(r"^merge_(\d+(?:_\d+)*)$")


def scan_result_dir(result_dir: Path) -> List[Tuple[int, str, str]]:
    found: List[Tuple[int, str, str]] = []
    for p in sorted(result_dir.iterdir()):
        if p.is_symlink() or not p.is_dir() or not (p / "poses.txt").is_file():
            continue
        m = _INDEXED.match(p.name)
        if m:
            found.append((int(m.group(1)), m.group(2), p.name))
            continue
        m = _CUMULATIVE.match(p.name)
        if m:
            ids = m.group(1).split("_")
            found.append((len(ids) - 1, ids[-1], p.name))
    found.sort()
    if not found:
        raise ValueError(f"no merge_* step directory with poses.txt under {result_dir}")
    indices = [f[0] for f in found]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"step indices are not contiguous: {indices}")
    return found


def _session_dir(sessions_root: Path, tag: str) -> Optional[Path]:
    for cand in (sessions_root / tag, sessions_root / f"{int(tag):03d}" if tag.isdigit() else None):
        if cand is not None and (cand / "poses.txt").is_file():
            return cand
    return None


def _registry_rows(path: Path) -> int:
    """Accepted loop rows in preds/loop_registry.txt; a missing file counts as zero."""
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.startswith("#"))


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def import_results(service: RunService, rid: str, req: ImportRequest, allowed_roots: Sequence[Path]) -> Run:
    region = service.regions.get(rid)  # KeyError -> 404
    result_dir = resolve_allowed_path(req.result_dir, allowed_roots)
    found = scan_result_dir(result_dir)
    sessions_root = resolve_allowed_path(req.sessions_root, allowed_roots) if req.sessions_root else None
    run_id = new_id("run")
    run_dir = service.runs.run_dir(rid, run_id)
    for sub in ("cache", "evaluations"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "output").symlink_to(result_dir, target_is_directory=True)
    session_ids: List[str] = []
    steps: List[StepRecord] = []
    prev_nodes = 0
    for index, tag, dir_name in found:
        sid = tag
        src = _session_dir(sessions_root, tag) if sessions_root is not None else None
        if src is not None:
            sid = service.sessions.save(session_from_dir(rid, tag, "imported", src)).id
        session_ids.append(sid)
        step_dir = result_dir / dir_name
        n = len(read_node_ids(step_dir / "poses.txt"))
        odom = read_edges(step_dir / "edges_odom.txt")[:, :2].astype(int)
        steps.append(StepRecord(
            index=index, session_id=sid, dir_name=dir_name, status="done", id_offset=prev_nodes, odom_nodes=n,
            covis_nodes=len(read_node_ids(step_dir / "intrinsics.txt")),
            components=int(connected_components(n, odom).max()) + 1 if n else 0,
            registry_edges=_registry_rows(step_dir / "preds" / "loop_registry.txt"),
            finished_at=_mtime_iso(step_dir / "poses.txt")))
        prev_nodes = n
    stamp = now_iso()
    run = Run(id=run_id, region_id=rid, name=req.name or result_dir.name, kind="imported", start_step=found[0][0],
              session_ids=session_ids, status="succeeded", num_steps_expected=len(found),
              last_step_index=found[-1][0], created_at=stamp, finished_at=stamp,
              meta={"result_dir": str(result_dir), "sessions_root": str(sessions_root) if sessions_root else None})
    service.runs.save(run)
    service.runs.write_steps(rid, run_id, steps)
    if req.promote:
        region.head = RegionHead(run_id=run_id, step_index=found[-1][0], session_ids=session_ids, lineage=[run_id])
        service.regions.save(region)
    service.evals.enqueue_all_steps(rid, run_id)
    service.evals.enqueue_final_eval(rid, run_id)
    return run
