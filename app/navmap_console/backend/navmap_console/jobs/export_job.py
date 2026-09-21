"""Export job: bundle map / report / preds into <run_dir>/exports/<name>.tar.gz + <name>.json.

CLI: python export_job.py --run_dir DIR --kind map|report|preds --name NAME
     --out_json PATH [--steps 0,2] [--eval_dir DIR] [--sources d1:d2] [--verify]

map    : final/ when the run has one; otherwise (imported runs) the last merge_* step is
         consolidated into exports/.stage_<name>/ with map_merge_pack.consolidate_map
         (images gathered from --sources) and that staging dir is packed and removed.
report : the contents of --eval_dir (the report/ of one evaluation), at the archive root.
preds  : <step name>/preds/... of the steps named by --steps (all steps when omitted).
verify : extract the tarball to exports/.verify_<name>/, check it is readable and non-empty,
         remove the extraction and write "verified" into the metadata json.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[5]
PY_DIR = REPO_ROOT / "python"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _step_index(step_dir: Path) -> Optional[int]:
    """merge_0_1_2 -> 2 (cumulative step names end with the step index)."""
    tail = step_dir.name.split("_")[-1]
    return int(tail) if tail.isdigit() else None


def _step_dirs(run_dir: Path, steps: Optional[Sequence[int]]) -> List[Path]:
    found = [(idx, d) for d in (run_dir / "output").glob("merge_*")
             if d.is_dir() and (idx := _step_index(d)) is not None]
    return [d for idx, d in sorted(found) if steps is None or idx in steps]


def _write_tar(tar_path: Path, entries: Sequence[Tuple[Path, str]]) -> int:
    """Pack (directory, archive prefix) pairs file by file; returns the number of members."""
    count = 0
    with tarfile.open(tar_path, "w:gz") as tf:
        for root, prefix in entries:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    tf.add(str(p), arcname=f"{prefix}{p.relative_to(root).as_posix()}")
                    count += 1
    return count


def build_map_dir(run_dir: Path, sources: Sequence[Path], stage: Path) -> Path:
    """Directory holding the navigation map: final/ or a consolidated copy of the last step."""
    final = run_dir / "final"
    if (final / "poses.txt").is_file():
        return final
    steps = _step_dirs(run_dir, None)
    if not steps:
        raise RuntimeError("no final/ and no merge_* step directory to export")
    from map_merge_pack import consolidate_map

    if stage.exists():
        shutil.rmtree(stage)
    consolidate_map(steps[-1], [s for s in sources if s.is_dir()] or steps, stage)
    return stage


def _verify(tar_path: Path, target: Path) -> bool:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    try:
        with tarfile.open(tar_path) as tf:
            members = tf.getmembers()
            tf.extractall(target)
        return bool(members) and all((target / m.name).is_file() for m in members if m.isfile())
    finally:
        shutil.rmtree(target, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--kind", required=True, choices=["map", "report", "preds"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--steps", default=None, help="comma-separated step indices (preds only)")
    ap.add_argument("--eval_dir", default=None, help="report directory to pack (report only)")
    ap.add_argument("--sources", default="", help="colon-separated image source dirs (map without final/)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    out_json = Path(args.out_json)
    exports_dir = out_json.parent
    exports_dir.mkdir(parents=True, exist_ok=True)
    tar_path = exports_dir / f"{args.name}.tar.gz"
    # keep the placeholder written by ExportService.create() (created_at/job_id): prune orders bundles by it
    meta: Dict[str, Any] = json.loads(out_json.read_text()) if out_json.is_file() else {}
    meta.update({"name": args.name, "kind": args.kind, "status": "succeeded"})
    meta.setdefault("created_at", _now_iso())
    stage = exports_dir / f".stage_{args.name}"
    try:
        if args.kind == "map":
            sources = [Path(s) for s in args.sources.split(":") if s]
            meta["entries"] = _write_tar(tar_path, [(build_map_dir(run_dir, sources, stage), "")])
        elif args.kind == "report":
            report = Path(args.eval_dir or "")
            if not report.is_dir():
                raise RuntimeError(f"report dir not found: {report}")
            meta["entries"] = _write_tar(tar_path, [(report, "")])
        else:
            steps = [int(s) for s in args.steps.split(",")] if args.steps else None
            step_dirs = [d for d in _step_dirs(run_dir, steps) if (d / "preds").is_dir()]
            meta["entries"] = _write_tar(tar_path, [(d / "preds", f"{d.name}/preds/") for d in step_dirs])
            meta["steps"] = [_step_index(d) for d in step_dirs]
        meta["size"] = tar_path.stat().st_size
        meta["sha256"] = _sha256(tar_path)
        if args.verify:
            meta["verified"] = _verify(tar_path, exports_dir / f".verify_{args.name}")
        print(f"exported {tar_path} ({meta['size']} bytes, {meta['entries']} entries)", flush=True)
    except Exception as exc:  # noqa: BLE001 - a failed export must still leave metadata on disk
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        print(f"export failed: {meta['error']}", flush=True)
        out_json.write_text(json.dumps(meta))
        return 1
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    out_json.write_text(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
