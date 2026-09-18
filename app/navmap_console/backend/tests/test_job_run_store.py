from pathlib import Path

import pytest


def test_job_store_roundtrip(tmp_path: Path):
    from navmap_console.jobs.store import JobStore
    from navmap_console.models import Job

    store = JobStore(tmp_path / "jobs")
    job = Job(id="job_20260918_200000_ab12", kind="merge", queue="gpu", argv=["python", "x.py"],
              cwd="/repo", log_path=str(store.log_path("job_20260918_200000_ab12")))
    store.save(job)
    assert store.get(job.id).argv == ["python", "x.py"]
    assert store.path(job.id).name == "job_20260918_200000_ab12.json"
    assert store.log_path(job.id).suffix == ".log"
    later = job.model_copy(update={"id": "job_20260918_200100_cd34"})
    store.save(later)
    assert [j.id for j in store.list()] == [later.id, job.id]  # newest first
    with pytest.raises(KeyError):
        store.get("job_missing")


def test_run_store_and_steps(tmp_path: Path):
    from navmap_console.models import Run, StepRecord
    from navmap_console.services.runs import RunStore

    runs = RunStore(tmp_path / "regions")
    run = Run(id="run_1", region_id="reg_1", name="first", session_ids=["ses_a", "ses_b"], num_steps_expected=2)
    runs.save(run)
    assert runs.get("reg_1", "run_1").session_ids == ["ses_a", "ses_b"]
    assert runs.read_steps("reg_1", "run_1") == []
    runs.write_steps("reg_1", "run_1", [StepRecord(index=0, session_id="ses_a", status="done")])
    steps = runs.read_steps("reg_1", "run_1")
    assert steps[0].status == "done" and steps[0].pgo_error_final is None
    assert [r.id for r in runs.list("reg_1")] == ["run_1"]
    with pytest.raises(KeyError):
        runs.get("reg_1", "run_x")


def test_run_create_requires_sessions():
    from pydantic import ValidationError

    from navmap_console.models import RunCreate

    with pytest.raises(ValidationError):
        RunCreate(session_ids=[])
    assert RunCreate(session_ids=["s"]).kind == "merge"


def test_settings_fake_pipeline_flag(tmp_path: Path):
    from navmap_console.config import load_settings

    assert load_settings({"NAVMAP_CONSOLE_DATA_ROOT": str(tmp_path)}).fake_pipeline is False
    assert load_settings({"NAVMAP_CONSOLE_DATA_ROOT": str(tmp_path), "NAVMAP_CONSOLE_FAKE_PIPELINE": "1"}).fake_pipeline
