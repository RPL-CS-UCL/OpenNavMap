"""Subprocess job runner: two queues (gpu/cpu), file-backed logs, restart-safe adoption (spec §5.4)."""
import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import psutil

from ..config import Settings
from ..models import Job, JobKind, QueueName, now_iso
from ..store import new_id
from .bus import EventBus
from .env import build_env, command_prefix, resolve_cpu_list
from .progress import STAGE_NAMES, LineTailer, classify_crash, parse_progress, read_log_lines
from .store import JobStore

log = logging.getLogger(__name__)
POLL_SECONDS = 0.5
KILL_GRACE_SECONDS = 10.0
ERROR_TAIL_LINES = 60


class JobHooks:
    """Callbacks run inside the event loop; keep them quick (file writes) or offload to an executor."""

    async def on_progress(self, job: Job, event: Dict[str, Any]) -> None:
        return None

    async def on_finished(self, job: Job) -> None:
        return None


class JobRunner:
    def __init__(self, settings: Settings, store: JobStore, bus: EventBus, hooks: Optional[JobHooks] = None) -> None:
        self.settings = settings
        self.store = store
        self.bus = bus
        self.hooks = hooks or JobHooks()
        self._queues: Dict[str, "asyncio.Queue[str]"] = {}
        self._workers: List["asyncio.Task[None]"] = []
        self._watchers: Dict[str, "asyncio.Task[None]"] = {}
        self._escalations: List["asyncio.Task[None]"] = []
        self._procs: Dict[str, "asyncio.subprocess.Process"] = {}
        self._active: Dict[str, str] = {}  # queue name -> job id
        self._cancel_requested: Set[str] = set()
        self._line_counts: Dict[str, int] = {}

    # ---- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        self._queues = {"gpu": asyncio.Queue(), "cpu": asyncio.Queue()}
        for job in reversed(self.store.list()):  # oldest first
            if job.status == "running":
                await self._adopt(job)
            elif job.status == "queued":
                self._queues[job.queue].put_nowait(job.id)
        self._workers = [asyncio.ensure_future(self._worker(name)) for name in self._queues]

    async def stop(self) -> None:
        """Stop scheduling; running subprocesses are left alive and re-adopted on the next start."""
        tasks = self._workers + list(self._watchers.values()) + self._escalations
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._workers, self._watchers, self._escalations = [], {}, []

    # ---- public API ------------------------------------------------------
    def new_job(self, kind: JobKind, queue: QueueName, argv: List[str], *, region_id: Optional[str] = None,
                run_id: Optional[str] = None, total_steps: Optional[int] = None) -> Job:
        jid = new_id("job")
        job = Job(id=jid, kind=kind, queue=queue, argv=list(argv), cwd=str(self.settings.repo_root),
                  env=build_env(self.settings), cpu_list=resolve_cpu_list(self.settings) if queue == "gpu" else None,
                  log_path=str(self.store.log_path(jid)), region_id=region_id, run_id=run_id)
        job.progress.total = total_steps
        self.store.save(job)
        return job

    def submit(self, job: Job) -> Job:
        job.status = "queued"
        self.store.save(job)
        self._publish_state(job)
        self._queues[job.queue].put_nowait(job.id)
        return job

    def queue_lengths(self) -> Dict[str, int]:
        return {name: q.qsize() + (1 if name in self._active else 0) for name, q in self._queues.items()}

    def line_count(self, job_id: str) -> int:
        if job_id not in self._line_counts:
            self._line_counts[job_id] = len(read_log_lines(self.store.log_path(job_id)))
        return self._line_counts[job_id]

    async def cancel(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job.status == "queued":
            self._cancel_requested.add(job_id)
            job.status, job.finished_at = "cancelled", now_iso()
            self.store.save(job)
            self._publish_state(job)
            await self.hooks.on_finished(job)
            return job
        if job.status == "running" and job.pid:
            self._cancel_requested.add(job_id)
            self._signal_group(job.pid, signal.SIGTERM)
            self._escalations.append(asyncio.ensure_future(self._escalate(job.pid)))
        return job

    # ---- internals -------------------------------------------------------
    def _publish_state(self, job: Job) -> None:
        data = job.model_dump()
        self.bus.publish("jobs", "job.state", data)
        self.bus.publish(f"job:{job.id}", "job.state", data)

    def _publish_progress(self, job: Job) -> None:
        data = {"id": job.id, "run_id": job.run_id, "progress": job.progress.model_dump()}
        self.bus.publish("jobs", "job.progress", data)
        self.bus.publish(f"job:{job.id}", "job.progress", data)

    @staticmethod
    def _signal_group(pid: int, sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except ProcessLookupError:
            pass

    async def _escalate(self, pid: int) -> None:
        await asyncio.sleep(KILL_GRACE_SECONDS)
        if psutil.pid_exists(pid):
            self._signal_group(pid, signal.SIGKILL)

    async def _worker(self, queue_name: str) -> None:
        q = self._queues[queue_name]
        while True:
            jid = await q.get()
            if jid in self._cancel_requested:
                self._cancel_requested.discard(jid)
                continue
            try:
                job = self.store.get(jid)
            except KeyError:
                continue
            if job.status != "queued":
                continue
            self._active[queue_name] = jid
            try:
                await self._run(job)
            except Exception:  # never let one job kill the worker
                log.exception("job %s crashed the runner loop", jid)
                job = self.store.get(jid)
                if job.status in ("queued", "running"):
                    await self._finish(job, returncode=None, error="runner exception; see backend log")
            finally:
                self._active.pop(queue_name, None)

    async def _run(self, job: Job) -> None:
        argv = command_prefix(job.cpu_list) + job.argv
        env = dict(os.environ)
        env.update(job.env)
        Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(job.log_path, "ab") as log_f:
            log_f.write(("$ " + " ".join(argv) + "\n").encode())
            log_f.flush()
            proc = await asyncio.create_subprocess_exec(*argv, cwd=job.cwd, env=env, stdout=log_f,
                                                        stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        self._procs[job.id] = proc
        job.status, job.started_at, job.pid = "running", now_iso(), proc.pid
        try:
            job.process_create_time = psutil.Process(proc.pid).create_time()
        except psutil.Error:
            job.process_create_time = None
        self.store.save(job)
        self._publish_state(job)
        self._line_counts[job.id] = 0
        await self._tail(job, lambda: proc.returncode is not None)
        returncode = await proc.wait()
        self._procs.pop(job.id, None)
        await self._finish(self.store.get(job.id), returncode=returncode)

    async def _adopt(self, job: Job) -> None:
        """Re-attach to a subprocess that outlived the previous backend process.

        Whatever the process wrote while no backend was watching is replayed from ``job.log_offset``
        first, so step completions are not lost; a process that has already exited is then judged
        by its log tail instead of being written off as orphaned.
        """
        self._line_counts[job.id] = self._lines_before(Path(job.log_path), job.log_offset)
        if not self._process_alive(job):
            await self._tail(job, lambda: True, start_offset=job.log_offset)
            await self._finish_adopted(self.store.get(job.id), gone_at_start=True)
            return
        self._active[job.queue] = job.id
        self._watchers[job.id] = asyncio.ensure_future(self._watch_adopted(job))

    @staticmethod
    def _process_alive(job: Job) -> bool:
        if not job.pid or job.process_create_time is None:
            return False
        try:
            return abs(psutil.Process(job.pid).create_time() - job.process_create_time) < 1.0
        except psutil.Error:
            return False

    @staticmethod
    def _lines_before(path: Path, offset: int) -> int:
        """Number of log lines in the first ``offset`` bytes (``offset`` always ends on a newline)."""
        if offset <= 0 or not path.is_file():
            return 0
        with open(path, "rb") as f:
            return f.read(offset).count(b"\n")

    async def _watch_adopted(self, job: Job) -> None:
        pid = job.pid or 0
        path = Path(job.log_path)

        def gone() -> bool:
            return not psutil.pid_exists(pid)

        try:
            await self._tail(job, gone, start_offset=job.log_offset)
        finally:
            self._active.pop(job.queue, None)
            self._watchers.pop(job.id, None)
        await self._finish_adopted(self.store.get(job.id), gone_at_start=False)

    async def _finish_adopted(self, job: Job, *, gone_at_start: bool) -> None:
        """Exit status is unknowable for an adopted process; infer success from the log tail."""
        tail = read_log_lines(Path(job.log_path))[-3:]
        if any("merge_finalmap ->" in l or "STEP_DONE" in l for l in tail):
            await self._finish(job, returncode=0)
        elif gone_at_start:
            await self._finish(job, returncode=None, error="backend restarted and the process was gone",
                               status="orphaned")
        else:
            await self._finish(job, returncode=None, error="adopted process ended without a completion marker")

    async def _tail(self, job: Job, is_done: Callable[[], bool], start_offset: int = 0) -> None:
        tailer = LineTailer()
        offset = start_offset
        path = Path(job.log_path)
        while True:
            done = is_done()
            chunk = b""
            if path.exists():
                with open(path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read()
                offset += len(chunk)
            lines = tailer.feed(chunk) if chunk else []
            if done:
                last = tailer.flush()
                if last is not None:
                    lines.append(last)
            job.log_offset = offset - tailer.pending  # saved with the next progress change
            if lines:
                await self._handle_lines(job, lines)
            if done:
                return
            await asyncio.sleep(POLL_SECONDS)

    async def _handle_lines(self, job: Job, lines: List[str]) -> None:
        first_seq = self._line_counts.get(job.id, 0)
        self._line_counts[job.id] = first_seq + len(lines)
        self.bus.publish(f"job:{job.id}", "job.log", {"id": job.id, "first_seq": first_seq, "lines": lines})
        changed = False
        for line in lines:
            ev = parse_progress(line)
            if ev is None:
                continue
            self._apply_progress(job, ev)
            changed = True
            try:
                await self.hooks.on_progress(job, ev)
            except Exception:
                log.exception("on_progress hook failed for %s", job.id)
        if changed:
            self.store.save(job)
            self._publish_progress(job)

    @staticmethod
    def _apply_progress(job: Job, ev: Dict[str, Any]) -> None:
        p = job.progress
        kind = ev["kind"]
        if kind == "step_start":
            p.step, p.stage_index, p.detail = ev["index"], 2, f"submap {ev['sid']}"
        elif kind == "stage":
            p.stage_index = ev["stage_index"]
        elif kind == "pgo_initial":
            p.stage_index, p.detail = 7, f"PGO initial error {ev['value']:.3f}"
        elif kind == "pgo_final":
            p.stage_index, p.detail = 8, f"PGO final error {ev['value']:.3f}"
        elif kind == "step_saved":
            p.detail = "saved"
        elif kind == "step_done":
            p.completed_steps += 1
        p.stage = STAGE_NAMES.get(p.stage_index or 0)

    async def _finish(self, job: Job, *, returncode: Optional[int], error: Optional[str] = None,
                      status: Optional[str] = None) -> None:
        log_path = Path(job.log_path)
        tail = read_log_lines(log_path)[-ERROR_TAIL_LINES:]
        job.returncode, job.finished_at = returncode, now_iso()
        job.log_offset = log_path.stat().st_size if log_path.is_file() else job.log_offset
        if status:
            job.status = status
        elif job.id in self._cancel_requested:
            job.status = "cancelled"
        elif returncode == 0:
            job.status = "succeeded"
        else:
            job.status = "failed"
        self._cancel_requested.discard(job.id)
        if job.status == "failed":
            job.crash_kind = classify_crash(returncode, tail)
            job.error = error or "\n".join(tail)
        elif error:
            job.error = error
        self.store.save(job)
        self._publish_state(job)
        try:
            await self.hooks.on_finished(job)
        except Exception:
            log.exception("on_finished hook failed for %s", job.id)
