"""Export jobs and the exports/ directory lifecycle (create/list/path/delete/prune)."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import Settings
from ..jobs.runner import JobRunner
from ..models import Job
from ..readers.ate import final_eval_dir
from .runs import RunStore

EXPORT_SCRIPT = Path(__file__).resolve().parents[1] / "jobs" / "export_job.py"
KEEP_RECENT = 3


class ExportService:
    def __init__(self, settings: Settings, runner: JobRunner, runs: RunStore) -> None:
        self.settings, self.runner, self.runs = settings, runner, runs

    def exports_dir(self, rid: str, run_id: str) -> Path:
        return self.runs.run_dir(rid, run_id) / "exports"

    def create(self, rid: str, run_id: str, kind: str, steps: Optional[Sequence[int]] = None,
               image_sources: Optional[Sequence[Path]] = None) -> Job:
        self.runs.get(rid, run_id)  # KeyError -> 404
        run_dir = self.runs.run_dir(rid, run_id)
        name = f"{kind}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        out_json = self.exports_dir(rid, run_id) / f"{name}.json"
        argv = [str(self.settings.python), str(EXPORT_SCRIPT), "--run_dir", str(run_dir),
                "--kind", kind, "--name", name, "--out_json", str(out_json)]
        if kind == "map" and image_sources:
            argv += ["--sources", ":".join(str(p) for p in image_sources)]
        if kind == "report":
            argv += ["--eval_dir", str(final_eval_dir(run_dir) / "report")]
        if kind == "preds" and steps:
            argv += ["--steps", ",".join(str(s) for s in sorted(steps))]
        job = self.runner.new_job("export", "cpu", argv, region_id=rid, run_id=run_id)
        # metadata written before submit so list/DELETE see the bundle while it is still packing
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"name": name, "kind": kind, "status": "queued", "job_id": job.id,
                                        "created_at": job.created_at, "steps": sorted(steps) if steps else None}))
        return self.runner.submit(job)

    def _job_status(self, job_id: Optional[str]) -> Optional[str]:
        if not job_id:
            return None
        try:
            return self.runner.store.get(job_id).status
        except KeyError:
            return None

    def list(self, rid: str, run_id: str) -> List[Dict[str, Any]]:
        self.runs.get(rid, run_id)
        out: List[Dict[str, Any]] = []
        for p in sorted(self.exports_dir(rid, run_id).glob("*.json"), reverse=True):
            data = json.loads(p.read_text())
            data.setdefault("name", p.stem)
            status = self._job_status(data.get("job_id"))
            if status in ("queued", "running") or (status in ("failed", "cancelled") and data["status"] == "queued"):
                data["status"] = status  # the job's state supersedes the placeholder written at create()
            out.append(data)
        return out

    def path(self, rid: str, run_id: str, name: str) -> Path:
        tar = self.exports_dir(rid, run_id) / f"{name}.tar.gz"
        if "/" in name or "\\" in name or not tar.is_file():
            raise KeyError(name)
        return tar

    def delete(self, rid: str, run_id: str, name: str) -> None:
        d = self.exports_dir(rid, run_id)
        tar, meta = d / f"{name}.tar.gz", d / f"{name}.json"
        if "/" in name or "\\" in name or not (tar.is_file() or meta.is_file()):
            raise KeyError(name)
        if meta.is_file() and self._job_status(json.loads(meta.read_text()).get("job_id")) in ("queued", "running"):
            raise RuntimeError(f"export {name} is still packing")
        for p in (tar, meta):
            if p.is_file():
                p.unlink()

    def prune(self, rid: str, run_id: str, keep: int = KEEP_RECENT) -> None:
        """Drop the oldest bundles beyond `keep` (by created_at, newest first); in-flight ones are skipped."""
        items = sorted(self.list(rid, run_id), key=lambda d: d.get("created_at", ""), reverse=True)
        for item in items[keep:]:
            try:
                self.delete(rid, run_id, item["name"])
            except (KeyError, RuntimeError):
                continue
