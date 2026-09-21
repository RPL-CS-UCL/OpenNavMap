"""Evaluation jobs: one per-step ATE evaluation per completed step, one final report per run.

Both run jobs/eval_job.py through the traj_evaluation toolchain (cpu queue); the console
never computes an alignment or RMSE itself.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..jobs.env import build_env
from ..jobs.runner import JobRunner
from ..models import Job
from ..readers.ate import final_eval_dir, step_eval_path, step_has_gt
from .runs import RunStore
EVAL_ALG = "opennavmap"
EVAL_SCRIPT = Path(__file__).resolve().parents[1] / "jobs" / "eval_job.py"


class EvalService:
    def __init__(self, settings: Settings, runner: JobRunner, runs: RunStore) -> None:
        self.settings, self.runner, self.runs = settings, runner, runs

    def _eval_env(self) -> Dict[str, str]:
        # the merge env's LD_PRELOAD points at the merge conda's libstdc++, which is not the eval env's
        return {k: v for k, v in build_env(self.settings).items() if k != "LD_PRELOAD"}

    def eval_argv(self, step_dir: Path, eval_dir: Path, dataset: str) -> List[str]:
        argv = [str(self.settings.eval_python), str(EVAL_SCRIPT), "--step_dir", str(step_dir),
                "--eval_dir", str(eval_dir), "--dataset", dataset, "--alg", EVAL_ALG]
        if self.settings.fake_pipeline:
            argv.append("--fake-report")
        return argv

    def _submit(self, kind: str, rid: str, run_id: str, step_dir: Path, eval_dir: Path, dataset: str) -> Job:
        job = self.runner.new_job(kind, "cpu", self.eval_argv(step_dir, eval_dir, dataset),  # type: ignore[arg-type]
                                  region_id=rid, run_id=run_id, env=self._eval_env())
        return self.runner.submit(job)

    def enqueue_step_eval(self, rid: str, run_id: str, step_index: int) -> Optional[Job]:
        """One per-step ATE job for a completed step; None when the step carries no GT."""
        run_dir = self.runs.run_dir(rid, run_id)
        rec = next((s for s in self.runs.read_steps(rid, run_id) if s.index == step_index), None)
        if rec is None or not rec.dir_name:
            return None
        step_dir = run_dir / "output" / rec.dir_name
        if not step_has_gt(step_dir):
            return None
        return self._submit("per_step_eval", rid, run_id, step_dir, step_eval_path(run_dir, step_index).parent,
                            f"step_{step_index:02d}")

    def enqueue_final_eval(self, rid: str, run_id: str) -> Optional[Job]:
        """The official report for the run's final map; falls back to the last done step (imported runs)."""
        run_dir = self.runs.run_dir(rid, run_id)
        step_dir = run_dir / "final"
        if not (step_dir / "poses.txt").is_file():
            done = [s for s in self.runs.read_steps(rid, run_id) if s.status == "done" and s.dir_name]
            if not done:
                return None
            step_dir = run_dir / "output" / max(done, key=lambda s: s.index).dir_name
        if not step_has_gt(step_dir):
            return None
        return self._submit("official_eval", rid, run_id, step_dir, final_eval_dir(run_dir), "final")

    def enqueue_all_steps(self, rid: str, run_id: str) -> List[Job]:
        jobs = [self.enqueue_step_eval(rid, run_id, s.index)
                for s in self.runs.read_steps(rid, run_id) if s.status == "done" and s.dir_name]
        return [j for j in jobs if j is not None]

    # ---- reads (the single "final" report) --------------------------------
    def _final_job(self, run_id: str) -> Optional[Job]:
        jobs = [j for j in self.runner.store.list() if j.kind == "official_eval" and j.run_id == run_id]
        return max(jobs, key=lambda j: j.created_at) if jobs else None

    def list_reports(self, rid: str, run_id: str) -> List[Dict[str, Any]]:
        """eval.json of the final report (+ its job), or just the job while it is still queued/running."""
        self.runs.get(rid, run_id)  # KeyError -> 404
        eval_json = final_eval_dir(self.runs.run_dir(rid, run_id)) / "eval.json"
        job = self._final_job(run_id)
        if not eval_json.is_file():
            return [] if job is None else [{"eid": "final", "status": job.status, "job": job.model_dump()}]
        data = json.loads(eval_json.read_text())
        if job is not None and job.status in ("queued", "running"):
            data["status"] = job.status  # a re-run in flight supersedes the stale file
        data.update(eid="final", job=job.model_dump() if job else None)
        return [data]

    def report_detail(self, rid: str, run_id: str, eid: str) -> Dict[str, Any]:
        if eid != "final":
            raise KeyError(eid)
        items = self.list_reports(rid, run_id)
        if not items or "ate_trans" not in items[0]:
            raise FileNotFoundError(f"evaluation {eid} has no report yet")
        report = self._report_dir(rid, run_id)
        items[0]["report_files"] = sorted(str(p.relative_to(report)) for p in report.rglob("*") if p.is_file()) \
            if report.is_dir() else []
        return items[0]

    def _report_dir(self, rid: str, run_id: str) -> Path:
        return final_eval_dir(self.runs.run_dir(rid, run_id)) / "report"

    def report_file(self, rid: str, run_id: str, eid: str, name: str) -> Path:
        """A file below evaluations/final/report/ (nested paths allowed, escaping it is not)."""
        self.runs.get(rid, run_id)
        if eid != "final":
            raise KeyError(eid)
        report = self._report_dir(rid, run_id).resolve()
        target = (report / name).resolve()
        if report != target and report not in target.parents:
            raise ValueError("path escapes report/")
        if not target.is_file():
            raise FileNotFoundError(name)
        return target
