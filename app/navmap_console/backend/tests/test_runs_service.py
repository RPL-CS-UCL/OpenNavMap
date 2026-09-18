import asyncio
import shutil
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, SYNTHETIC_MAP


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def world(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NAVMAP_CONSOLE_FAKE_STEP_SECONDS", "0.05")
    from navmap_console.config import load_settings
    from navmap_console.jobs.bus import EventBus
    from navmap_console.jobs.runner import JobRunner
    from navmap_console.jobs.store import JobStore
    from navmap_console.models import RegionCreate
    from navmap_console.services.catalog import RegionStore, SessionStore
    from navmap_console.services.ingest import session_from_dir
    from navmap_console.services.runs import RunJobHooks, RunService, RunStore

    settings = load_settings({"NAVMAP_CONSOLE_DATA_ROOT": str(tmp_path / "data"), "NAVMAP_CONSOLE_REPO": str(REPO_ROOT),
                              "NAVMAP_CONSOLE_PYTHON": sys.executable, "NAVMAP_CONSOLE_FAKE_PIPELINE": "1",
                              "MERGE_CPU_LIST": ""})
    for d in (settings.regions_dir, settings.jobs_dir):
        d.mkdir(parents=True, exist_ok=True)
    regions, sessions, runs = RegionStore(settings.regions_dir), SessionStore(settings.regions_dir), RunStore(settings.regions_dir)
    region = regions.create(RegionCreate(name="r"))
    sids = []
    for name in ("a", "b"):
        d = tmp_path / "ext" / name
        shutil.copytree(SYNTHETIC_MAP, d)
        s = session_from_dir(region.id, name, "path", d)
        sessions.save(s)
        sids.append(s.id)
    bus = EventBus()
    runner = JobRunner(settings, JobStore(settings.jobs_dir), bus)
    service = RunService(settings, runner, regions, sessions, runs)
    runner.hooks = RunJobHooks(service)
    return settings, bus, runner, service, region, sids


async def _wait_run(service, rid, run_id, timeout=15.0):
    for _ in range(int(timeout / 0.05)):
        r = service.runs.get(rid, run_id)
        if r.status in ("succeeded", "failed", "cancelled", "orphaned"):
            return r
        await asyncio.sleep(0.05)
    raise AssertionError("run did not finish")


def test_create_merge_run_end_to_end(world):
    from navmap_console.models import RunCreate

    settings, bus, runner, service, region, sids = world

    async def go():
        await runner.start()
        q = bus.subscribe("run:pending")
        run_ = service.create(region.id, RunCreate(name="first", session_ids=sids, params={"pgo_loop_sigma_trans": 0.2}))
        bus.unsubscribe("run:pending", q)
        q = bus.subscribe(f"run:{run_.id}")
        assert run_.status == "queued" and run_.job_id and run_.num_steps_expected == 2
        assert run_.params["pgo_loop_sigma_trans"] == 0.2 and run_.params["pgo_robust"] == "gnc_gm"
        done = await _wait_run(service, region.id, run_.id)
        await asyncio.sleep(0.2)
        await runner.stop()
        return run_, done, list(q._queue)

    run_, done, msgs = run(go())
    rdir = service.runs.run_dir(region.id, run_.id)
    assert (rdir / "inputs" / sids[0]).is_symlink() and (rdir / "inputs.txt").read_text().splitlines() == [
        str(rdir / "inputs" / sids[0]), str(rdir / "inputs" / sids[1])]
    assert done.status == "succeeded" and done.last_step_index == 1 and done.finished_at
    steps = service.runs.read_steps(region.id, run_.id)
    assert [s.status for s in steps] == ["done", "done"] and steps[1].dir_name == f"merge_001_{sids[1]}"
    assert steps[0].pgo_error_initial == pytest.approx(1.234) and steps[0].pgo_error_final == pytest.approx(0.456)
    assert steps[1].odom_nodes == 24 and steps[1].registry_edges == 1
    assert (rdir / "output" / f"merge_001_{sids[1]}" / "edges_trav.txt").is_file()
    assert done.final_dir == str(rdir / "final") and (rdir / "final" / "merge_meta.json").is_file()
    region_after = service.regions.get(region.id)
    assert region_after.head and region_after.head.run_id == run_.id and region_after.head.step_index == 1
    assert region_after.head.session_ids == sids
    types = [m["type"] for m in msgs]
    assert types.count("run.step_completed") == 2 and types[-1] == "run.state"
    detail = service.detail(region.id, run_.id)
    assert detail["job"]["status"] == "succeeded" and len(detail["steps"]) == 2


def test_region_busy_and_bad_inputs(world):
    from navmap_console.models import RunCreate
    from navmap_console.services.runs import RegionBusyError

    settings, bus, runner, service, region, sids = world

    async def go():
        await runner.start()
        first = service.create(region.id, RunCreate(session_ids=sids))
        with pytest.raises(RegionBusyError):
            service.create(region.id, RunCreate(session_ids=sids))
        with pytest.raises(ValueError, match="session"):
            service.create(region.id, RunCreate(session_ids=["ses_nope"]))
        with pytest.raises(ValueError, match="unknown"):
            service.create(region.id, RunCreate(session_ids=sids, params={"bogus": 1}))
        with pytest.raises(ValueError, match="parent"):
            service.create(region.id, RunCreate(kind="append", session_ids=sids))
        with pytest.raises(KeyError):
            service.create("reg_missing", RunCreate(session_ids=sids))
        cancelled = service.cancel(region.id, first.id)
        done = await _wait_run(service, region.id, first.id)
        await runner.stop()
        return cancelled, done

    cancelled, done = run(go())
    assert done.status == "cancelled"


def test_append_run_uses_base_and_global_steps(world):
    from navmap_console.models import RunCreate

    settings, bus, runner, service, region, sids = world

    async def go():
        await runner.start()
        first = service.create(region.id, RunCreate(session_ids=[sids[0]]))
        await _wait_run(service, region.id, first.id)
        await asyncio.sleep(0.2)
        second = service.create(region.id, RunCreate(kind="append", session_ids=[sids[1]]))  # parent defaults to head
        assert second.parent.run_id == first.id and second.parent.step_index == 0 and second.start_step == 1
        done = await _wait_run(service, region.id, second.id)
        await asyncio.sleep(0.2)
        await runner.stop()
        return first, second, done

    first, second, done = run(go())
    rdir = service.runs.run_dir(region.id, second.id)
    assert (rdir / "base" / "poses.txt").is_file()
    job = service.job_of(done)
    assert "--append_from" in job.argv and job.argv[job.argv.index("--start_step") + 1] == "1"
    steps = service.runs.read_steps(region.id, second.id)
    assert [s.index for s in steps] == [1] and steps[0].dir_name == f"merge_001_{sids[1]}"
    head = service.regions.get(region.id).head
    assert head.run_id == second.id and head.session_ids == sids and head.lineage == [first.id, second.id]


def test_failed_run_marks_running_step(world, monkeypatch):
    from navmap_console.models import RunCreate

    settings, bus, runner, service, region, sids = world
    monkeypatch.setenv("NAVMAP_CONSOLE_FAKE_FAIL_AT", "1")

    async def go():
        await runner.start()
        r = service.create(region.id, RunCreate(session_ids=sids))
        done = await _wait_run(service, region.id, r.id)
        await runner.stop()
        return done

    done = run(go())
    steps = service.runs.read_steps(region.id, done.id)
    assert done.status == "failed" and [s.status for s in steps] == ["done", "failed"]
    assert service.regions.get(region.id).head is None
