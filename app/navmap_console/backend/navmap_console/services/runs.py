"""Run records under regions/<rid>/runs/<run_id>/ (spec §5.3) and the run service."""
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..jobs.runner import JobHooks, JobRunner
from ..jobs.specs import merge_argv, validate_params
from ..models import Job, RegionHead, Run, RunCreate, RunParent, StepRecord, now_iso
from ..store import list_records, new_id, read_json, write_json_atomic
from .catalog import RegionStore, SessionStore

log = logging.getLogger(__name__)
EVAL_KINDS = ("per_step_eval", "official_eval")  # job kinds owned by services/evaluations.py
IMAGE_SIZE = (512, 288)


class RunStore:
    def __init__(self, regions_root: Path) -> None:
        self.root = regions_root

    def run_dir(self, rid: str, run_id: str) -> Path:
        return self.root / rid / "runs" / run_id

    def list(self, rid: str) -> List[Run]:
        return [Run.model_validate(r) for r in list_records(self.root / rid / "runs", "run.json")]

    def get(self, rid: str, run_id: str) -> Run:
        p = self.run_dir(rid, run_id) / "run.json"
        if not p.is_file():
            raise KeyError(run_id)
        return Run.model_validate(read_json(p))

    def save(self, run: Run) -> Run:
        write_json_atomic(self.run_dir(run.region_id, run.id) / "run.json", run.model_dump())
        return run

    def read_steps(self, rid: str, run_id: str) -> List[StepRecord]:
        p = self.run_dir(rid, run_id) / "steps.json"
        if not p.is_file():
            return []
        return [StepRecord.model_validate(s) for s in read_json(p)["steps"]]

    def write_steps(self, rid: str, run_id: str, steps: List[StepRecord]) -> None:
        write_json_atomic(self.run_dir(rid, run_id) / "steps.json", {"steps": [s.model_dump() for s in steps]})


class RegionBusyError(RuntimeError):
    """A merge/append is already queued or running for the region."""


def _pack():
    """Lazy import: python/map_merge_pack.py lives outside the package (numpy only, no torch)."""
    import map_merge_pack

    return map_merge_pack


def sessions_up_to(runs: RunStore, run: Run, step_index: int) -> List[str]:
    """Sessions merged into `run` at global step `step_index`, following the parent chain."""
    own = run.session_ids[: max(0, step_index - run.start_step + 1)]
    if run.parent is None:
        return own
    parent = runs.get(run.region_id, run.parent.run_id)
    return sessions_up_to(runs, parent, run.parent.step_index) + own


def _lineage(runs: RunStore, run: Run) -> List[str]:
    chain = [run.id]
    cur = run
    while cur.parent is not None:
        cur = runs.get(cur.region_id, cur.parent.run_id)
        chain.append(cur.id)
    return list(reversed(chain))


class RunService:
    def __init__(self, settings: Settings, runner: JobRunner, regions: RegionStore, sessions: SessionStore,
                 runs: RunStore) -> None:
        self.settings, self.runner, self.regions, self.sessions, self.runs = settings, runner, regions, sessions, runs
        # lazy: both modules import RunStore from this module
        from .evaluations import EvalService
        from .exports import ExportService

        self.evals = EvalService(settings, runner, runs)
        self.exports = ExportService(settings, runner, runs)

    # ---- queries ---------------------------------------------------------
    def job_of(self, run: Run) -> Optional[Job]:
        if not run.job_id:
            return None
        try:
            return self.runner.store.get(run.job_id)
        except KeyError:
            return None

    def detail(self, rid: str, run_id: str) -> Dict[str, Any]:
        run = self.runs.get(rid, run_id)
        job = self.job_of(run)
        return {"run": run.model_dump(), "steps": [s.model_dump() for s in self.runs.read_steps(rid, run_id)],
                "job": job.model_dump() if job else None}

    def _busy(self, rid: str) -> bool:
        return any(j.region_id == rid and j.kind in ("merge", "append") and j.status in ("queued", "running")
                   for j in self.runner.store.list())

    # ---- commands --------------------------------------------------------
    def create(self, rid: str, data: RunCreate) -> Run:
        region = self.regions.get(rid)
        known = {s.id: s for s in self.sessions.list(rid)}
        missing = [s for s in data.session_ids if s not in known]
        if missing:
            raise ValueError(f"unknown session ids: {', '.join(missing)}")
        params = validate_params(data.params)
        parent: Optional[RunParent] = data.parent
        if data.kind == "append":
            if parent is None:
                if region.head is None:
                    raise ValueError("append needs a parent step and the region has no final map yet")
                parent = RunParent(run_id=region.head.run_id, step_index=region.head.step_index)
            self.runs.get(rid, parent.run_id)  # KeyError if missing
        if self._busy(rid):  # after input validation so callers get the more specific error first
            raise RegionBusyError(f"region {rid} already has a merge queued or running")
        run_id = new_id("run")
        run_dir = self.runs.run_dir(rid, run_id)
        for sub in ("inputs", "output", "cache", "evaluations"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        links = []
        for sid in data.session_ids:
            link = run_dir / "inputs" / sid
            link.symlink_to(Path(known[sid].path).resolve(), target_is_directory=True)
            links.append(str(link))
        (run_dir / "inputs.txt").write_text("\n".join(links) + "\n")
        start_step = 0
        append_from: Optional[Path] = None
        if parent is not None:
            append_from = self._prepare_base(rid, parent, run_dir / "base")
            start_step = parent.step_index + 1
        name = data.name or f"{data.kind} {now_iso()[:16].replace('T', ' ')}"
        argv = merge_argv(self.settings, submap_list=run_dir / "inputs.txt", result_dir=run_dir / "output",
                          image_size=IMAGE_SIZE, params=params, append_from=append_from, start_step=start_step,
                          rerun_viz_dir=run_dir / "output" / "rerun_viz")
        job = self.runner.new_job(data.kind, "gpu", argv, region_id=rid, run_id=run_id,
                                  total_steps=len(data.session_ids))
        run = Run(id=run_id, region_id=rid, name=name, kind=data.kind, parent=parent, start_step=start_step,
                  session_ids=list(data.session_ids), params=params, meta=dict(data.meta), job_id=job.id,
                  git_commit=_pack().git_commit_of(self.settings.repo_root), num_steps_expected=len(data.session_ids))
        self.runs.save(run)
        self.runs.write_steps(rid, run_id, [])
        self.runner.submit(job)
        return run

    def _prepare_base(self, rid: str, parent: RunParent, base_dir: Path) -> Path:
        """Consolidate the parent step into a complete, loadable map for --append_from (spec §5.5.5)."""
        parent_run = self.runs.get(rid, parent.run_id)
        steps = {s.index: s for s in self.runs.read_steps(rid, parent.run_id) if s.status == "done"}
        if parent.step_index not in steps:
            raise ValueError(f"parent step {parent.step_index} of run {parent.run_id} is not a completed step")
        parent_dir = self.runs.run_dir(rid, parent.run_id)
        step_dir = parent_dir / "output" / steps[parent.step_index].dir_name
        sources = self._image_sources(rid, parent_run, parent.step_index)
        _pack().consolidate_map(step_dir, sources, base_dir,
                                meta={"run_id": parent.run_id, "step_index": parent.step_index})
        return base_dir

    def _image_sources(self, rid: str, run: Run, up_to: int) -> List[Path]:
        run_dir = self.runs.run_dir(rid, run.id)
        sources: List[Path] = []
        if (run_dir / "base").is_dir():
            sources.append(run_dir / "base")
        for s in sorted(self.runs.read_steps(rid, run.id), key=lambda s: s.index):
            if s.status == "done" and s.index <= up_to and s.dir_name:
                sources.append(run_dir / "output" / s.dir_name)
        return sources

    def image_sources(self, rid: str, run_id: str) -> List[Path]:
        """base/ (append runs) plus every completed step directory, in order; later dirs override earlier."""
        run = self.runs.get(rid, run_id)
        return self._image_sources(rid, run, up_to=10 ** 9)

    async def cancel(self, rid: str, run_id: str) -> Run:
        """Ask the runner to stop the job; the run itself is finalised by the on_finished hook."""
        run = self.runs.get(rid, run_id)
        if run.job_id and run.status in ("queued", "running"):
            await self.runner.cancel(run.job_id)
        return run

    # ---- called by hooks -------------------------------------------------
    def apply_progress(self, job: Job, ev: Dict[str, Any]) -> None:
        rid, run_id = job.region_id or "", job.run_id or ""
        run = self.runs.get(rid, run_id)
        steps = self.runs.read_steps(rid, run_id)
        kind = ev["kind"]
        if kind == "step_start":
            steps = [s for s in steps if s.index != ev["index"]]
            steps.append(StepRecord(index=ev["index"], session_id=ev["sid"], started_at=now_iso()))
            if run.status != "running":
                run.status = "running"
                self.runs.save(run)
                self.runner.bus.publish(f"run:{run_id}", "run.state", run.model_dump())
        elif steps and kind == "pgo_initial":
            steps[-1].pgo_error_initial = ev["value"]
        elif steps and kind == "pgo_final":
            steps[-1].pgo_error_final = ev["value"]
        elif steps and kind == "step_done":
            s = steps[-1]
            s.status, s.finished_at, s.dir_name = "done", now_iso(), Path(ev["dir"]).name
            s.id_offset, s.odom_nodes, s.covis_nodes = ev.get("id_offset"), ev.get("odom_nodes"), ev.get("covis_nodes")
            s.components, s.registry_edges = ev.get("components"), ev.get("registry")
            run.last_step_index = s.index
            self.runs.save(run)
        else:
            return
        self.runs.write_steps(rid, run_id, steps)
        if kind == "step_done":
            self.runner.bus.publish(f"run:{run_id}", "run.step_completed",
                                    {"run_id": run_id, "step": steps[-1].model_dump()})
            self.evals.enqueue_step_eval(rid, run_id, steps[-1].index)

    async def finish(self, job: Job) -> None:
        rid, run_id = job.region_id or "", job.run_id or ""
        run = self.runs.get(rid, run_id)
        steps = self.runs.read_steps(rid, run_id)
        run.status, run.finished_at = job.status, now_iso()
        if job.status != "succeeded":
            for s in steps:
                if s.status == "running":
                    s.status, s.finished_at = "failed", now_iso()
            self.runs.write_steps(rid, run_id, steps)
        else:
            await self._consolidate_final(rid, run, steps)
        self.runs.save(run)
        self.runner.bus.publish(f"run:{run_id}", "run.state", run.model_dump())
        if job.status == "succeeded":
            self.evals.enqueue_final_eval(rid, run_id)

    async def _consolidate_final(self, rid: str, run: Run, steps: List[StepRecord]) -> None:
        done = [s for s in steps if s.status == "done"]
        if not done:
            run.final_error = "no completed step"
            return
        last = max(done, key=lambda s: s.index)
        run_dir = self.runs.run_dir(rid, run.id)
        step_dir = run_dir / "output" / last.dir_name
        sources = self._image_sources(rid, run, last.index)
        meta = {"run_id": run.id, "region_id": rid, "step_index": last.index, "params": run.params,
                "session_ids": sessions_up_to(self.runs, run, last.index)}
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, lambda: _pack().consolidate_map(step_dir, sources, run_dir / "final", meta=meta))
            run.final_dir = str(run_dir / "final")
        except Exception as exc:  # consolidation failure must not hide a successful merge
            log.exception("consolidation failed for run %s", run.id)
            run.final_error = f"{type(exc).__name__}: {exc}"
            return
        region = self.regions.get(rid)
        region.head = RegionHead(run_id=run.id, step_index=last.index,
                                 session_ids=sessions_up_to(self.runs, run, last.index),
                                 lineage=_lineage(self.runs, run))
        self.regions.save(region)
        map_link = self.regions.region_dir(rid) / "map"
        if map_link.is_symlink() or map_link.exists():
            map_link.unlink()
        map_link.symlink_to(run_dir / "final", target_is_directory=True)


class RunJobHooks(JobHooks):
    def __init__(self, service: RunService) -> None:
        self.service = service

    async def on_progress(self, job: Job, event: Dict[str, Any]) -> None:
        if job.run_id and job.kind in ("merge", "append"):
            self.service.apply_progress(job, event)

    async def on_finished(self, job: Job) -> None:
        if job.run_id and job.kind in ("merge", "append"):
            await self.service.finish(job)
        elif job.run_id and job.kind in EVAL_KINDS:
            # summaries/evaluations read eval.json lazily; tell subscribers it changed
            self.service.runner.bus.publish(f"run:{job.run_id}", "run.evaluated",
                                            {"run_id": job.run_id, "job_id": job.id, "kind": job.kind,
                                             "status": job.status})
        elif job.kind == "export" and job.region_id and job.run_id:
            self.service.exports.prune(job.region_id, job.run_id)
