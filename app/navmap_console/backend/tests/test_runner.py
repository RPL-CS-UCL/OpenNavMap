import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from conftest import REPO_ROOT


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _settings(tmp_path: Path):
    from navmap_console.config import load_settings

    return load_settings({"NAVMAP_CONSOLE_DATA_ROOT": str(tmp_path), "NAVMAP_CONSOLE_REPO": str(REPO_ROOT),
                          "NAVMAP_CONSOLE_PYTHON": sys.executable, "MERGE_CPU_LIST": ""})


def _make(tmp_path: Path, hooks=None):
    from navmap_console.jobs.bus import EventBus
    from navmap_console.jobs.runner import JobRunner
    from navmap_console.jobs.store import JobStore

    settings = _settings(tmp_path)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    bus = EventBus()
    runner = JobRunner(settings, JobStore(settings.jobs_dir), bus, hooks)
    return settings, bus, runner


async def _wait_status(runner, jid: str, wanted, timeout: float = 10.0):
    for _ in range(int(timeout / 0.05)):
        job = runner.store.get(jid)
        if job.status in wanted:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {jid} stuck in {runner.store.get(jid).status}")


def test_event_bus_seq_and_fanout():
    from navmap_console.jobs.bus import EventBus

    async def go():
        bus = EventBus()
        q1, q2 = bus.subscribe("t"), bus.subscribe("t")
        m = bus.publish("t", "job.state", {"x": 1})
        assert m["seq"] == 1 and m["topic"] == "t" and m["ts"]
        assert (await q1.get())["data"] == {"x": 1} and (await q2.get())["seq"] == 1
        bus.unsubscribe("t", q1)
        assert bus.publish("t", "job.state", {})["seq"] == 2
        assert q1.empty() and q2.qsize() == 1

    run(go())


def test_runner_runs_script_and_streams(tmp_path: Path):
    events: List[Dict[str, Any]] = []
    finished: List[str] = []

    from navmap_console.jobs.runner import JobHooks

    class Hooks(JobHooks):
        async def on_progress(self, job, event):
            events.append(event)

        async def on_finished(self, job):
            finished.append(job.status)

    settings, bus, runner = _make(tmp_path, Hooks())
    script = tmp_path / "s.py"
    script.write_text("import sys,time\nprint('--- Merging submap 0: a ---')\nsys.stdout.write('p 1%\\rp 99%\\n')\n"
                      "time.sleep(0.2)\nprint('STEP_DONE index=0 sid=a dir=/x id_offset=0 odom_nodes=3 covis_nodes=3 components=1 registry=0')\n"
                      "sys.stderr.write('warn\\n')\n")

    async def go():
        await runner.start()
        q = bus.subscribe("jobs")
        job = runner.new_job("merge", "gpu", [sys.executable, str(script)], region_id="r", run_id="run", total_steps=1)
        assert job.env["MKL_THREADING_LAYER"] == "GNU" and job.progress.total == 1
        runner.submit(job)
        qj = bus.subscribe(f"job:{job.id}")
        assert runner.queue_lengths()["gpu"] >= 1
        done = await _wait_status(runner, job.id, ("succeeded", "failed"))
        await runner.stop()
        return job, done, q, qj

    job, done, q, qj = run(go())
    assert done.status == "succeeded" and done.returncode == 0 and done.pid and done.process_create_time
    assert done.progress.completed_steps == 1 and done.progress.step == 0
    lines = Path(done.log_path).read_bytes().decode().split("\n")  # text mode would translate the tqdm "\r"
    assert "p 1%\rp 99%" in lines and "warn" in lines
    assert [e["kind"] for e in events] == ["step_start", "step_done"]
    assert finished == ["succeeded"]
    types = [m["type"] for m in list(q._queue)]
    assert types[0] == "job.state" and "job.progress" in types and types[-1] == "job.state"
    log_msgs = [m for m in list(qj._queue) if m["type"] == "job.log"]
    assert log_msgs and log_msgs[0]["data"]["first_seq"] == 0 and "p 99%" in sum((m["data"]["lines"] for m in log_msgs), [])
    assert runner.line_count(job.id) == 5  # "$ argv" echo + 4 script lines


def test_runner_failure_classified(tmp_path: Path):
    settings, bus, runner = _make(tmp_path)
    script = tmp_path / "f.py"
    script.write_text("print('RuntimeError: CUDA out of memory')\nraise SystemExit(1)\n")

    async def go():
        await runner.start()
        job = runner.submit(runner.new_job("merge", "gpu", [sys.executable, str(script)]))
        done = await _wait_status(runner, job.id, ("succeeded", "failed"))
        await runner.stop()
        return done

    done = run(go())
    assert done.status == "failed" and done.returncode == 1 and done.crash_kind == "cuda_oom"
    assert "CUDA out of memory" in (done.error or "")


def test_runner_cancel_kills_process_group(tmp_path: Path):
    settings, bus, runner = _make(tmp_path)
    marker = tmp_path / "child.pid"
    script = tmp_path / "c.py"
    script.write_text(
        "import subprocess,sys,time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(marker)!r}, 'w').write(str(p.pid))\n"
        "print('started', flush=True)\ntime.sleep(60)\n")

    async def go():
        await runner.start()
        job = runner.submit(runner.new_job("merge", "gpu", [sys.executable, str(script)]))
        await _wait_status(runner, job.id, ("running",))
        for _ in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.05)
        child = int(marker.read_text())
        await runner.cancel(job.id)
        done = await _wait_status(runner, job.id, ("cancelled",))
        await asyncio.sleep(0.2)
        await runner.stop()
        return done, child

    done, child = run(go())
    assert done.status == "cancelled" and done.finished_at
    try:
        os.kill(child, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive, "grandchild survived: process group was not killed"


def test_runner_cancel_queued_job(tmp_path: Path):
    settings, bus, runner = _make(tmp_path)
    slow = tmp_path / "slow.py"
    slow.write_text("import time; time.sleep(3)\n")

    async def go():
        await runner.start()
        first = runner.submit(runner.new_job("merge", "gpu", [sys.executable, str(slow)]))
        second = runner.submit(runner.new_job("merge", "gpu", [sys.executable, str(slow)]))
        assert runner.queue_lengths()["gpu"] == 2
        await runner.cancel(second.id)
        assert runner.store.get(second.id).status == "cancelled"
        await runner.cancel(first.id)
        await _wait_status(runner, first.id, ("cancelled",))
        await runner.stop()

    run(go())


def test_runner_adopts_or_orphans_on_start(tmp_path: Path):
    import psutil

    from navmap_console.models import Job

    settings, bus, runner = _make(tmp_path)
    dead = Job(id="job_20260918_000000_dead", kind="merge", queue="gpu", status="running", argv=["x"], cwd=".",
               log_path=str(runner.store.log_path("job_20260918_000000_dead")), pid=2 ** 22 - 1, process_create_time=1.0)
    runner.store.save(dead)
    me = psutil.Process(os.getpid())
    alive = Job(id="job_20260918_000001_live", kind="merge", queue="gpu", status="running", argv=["x"], cwd=".",
                log_path=str(runner.store.log_path("job_20260918_000001_live")), pid=me.pid,
                process_create_time=me.create_time())
    runner.store.save(alive)
    Path(alive.log_path).write_text("--- Merging submap 4: z ---\n")

    async def go():
        await runner.start()
        await asyncio.sleep(0.1)
        got = runner.store.get(dead.id), runner.store.get(alive.id), runner.line_count(alive.id)
        await runner.stop()
        return got

    d, a, n = run(go())
    assert d.status == "orphaned" and d.finished_at
    assert a.status == "running" and n == 1
    assert a.progress.step == 4  # lines before the offset were replayed on adoption


def test_runner_adoption_replays_log_written_while_down(tmp_path: Path):
    """Lines the subprocess wrote after the old backend died must still drive progress and completion."""
    from navmap_console.models import Job, JobProgress

    events: List[Dict[str, Any]] = []

    class Hooks:
        async def on_progress(self, job, event):
            events.append(event)

        async def on_finished(self, job):
            pass

    settings, bus, runner = _make(tmp_path, Hooks())
    jid = "job_20260918_000002_gone"
    log_path = Path(runner.store.log_path(jid))
    seen = b"$ x\n--- Merging submap 0: a ---\n"
    unseen = (b"PGO: final error: 0.5\n"
              b"STEP_DONE index=0 sid=a dir=/tmp/m0 id_offset=0 odom_nodes=1 covis_nodes=1 components=1 registry=0\n"
              b"--- Merging submap 1: b ---\n"
              b"STEP_DONE index=1 sid=b dir=/tmp/m1 id_offset=1 odom_nodes=2 covis_nodes=2 components=1 registry=0\n"
              b"merge_finalmap -> /tmp/m1\n")
    log_path.write_bytes(seen + unseen)
    job = Job(id=jid, kind="merge", queue="gpu", status="running", argv=["x"], cwd=".", log_path=str(log_path),
              pid=2 ** 22 - 2, process_create_time=1.0, log_offset=len(seen), progress=JobProgress(step=0))
    runner.store.save(job)

    async def go():
        await runner.start()
        await asyncio.sleep(0.1)
        got = runner.store.get(jid), runner.line_count(jid)
        await runner.stop()
        return got

    done, n = run(go())
    assert done.status == "succeeded" and done.returncode == 0
    assert done.progress.completed_steps == 2 and done.log_offset == len(seen) + len(unseen)
    assert [e["index"] for e in events if e["kind"] == "step_done"] == [0, 1]
    assert n == 7  # seq numbering continues from the lines before log_offset
