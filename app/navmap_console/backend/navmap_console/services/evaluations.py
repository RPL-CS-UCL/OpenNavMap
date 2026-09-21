"""Evaluation jobs: one per-step ATE evaluation per completed step, one final report per run.

Both run jobs/eval_job.py through the traj_evaluation toolchain (cpu queue); the console
never computes an alignment or RMSE itself.
"""
from pathlib import Path
from typing import Dict, List, Optional

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
