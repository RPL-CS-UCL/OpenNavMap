"""EvalService: per-step and final evaluation jobs, queued on the cpu queue."""
import time
from pathlib import Path
from typing import Tuple

from fastapi.testclient import TestClient

from navmap_console.services.evaluations import EvalService


def _service(client: TestClient) -> EvalService:
    return client.app.state.run_service.evals


def make_gt_result(tmp_path: Path, n_steps: int = 2) -> Path:
    """A fake merge result whose step dirs carry GT + timestamps (2 identity frames each)."""
    result = tmp_path / "result"
    for i in range(n_steps):
        step = result / ("merge_" + "_".join(str(k) for k in range(i + 1)))  # cumulative names: merge_0, merge_0_1
        (step / "seq").mkdir(parents=True)
        (step / "preds").mkdir()
        lines = [f"seq/{k:06d}.color.jpg 1 0 0 0 {k}.0 0 0" for k in range(2)]
        (step / "poses.txt").write_text("\n".join(lines) + "\n")
        (step / "poses_abs_gt.txt").write_text("\n".join(lines) + "\n")
        (step / "timestamps.txt").write_text("\n".join(f"seq/{k:06d}.color.jpg {1000 + k}.0" for k in range(2)) + "\n")
        (step / "intrinsics.txt").write_text("\n".join(f"seq/{k:06d}.color.jpg 1 1 1 1 1 1" for k in range(2)) + "\n")
        for name in ("edges_odom.txt", "edges_covis.txt", "edges_trav.txt"):
            (step / name).write_text("")
    return result


def _make_imported_run(client: TestClient, tmp_path: Path) -> Tuple[str, str, Path]:
    result = make_gt_result(tmp_path)
    rid = client.post("/api/regions", json={"name": "r"}).json()["id"]
    run = client.post(f"/api/regions/{rid}/runs/import", json={"result_dir": str(result)}).json()
    return rid, run["id"], result


def _wait_jobs_done(client: TestClient, run_id: str, timeout: float = 10.0) -> list:
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = [j for j in client.get("/api/jobs").json() if j["run_id"] == run_id]
        if jobs and all(j["status"] in ("succeeded", "failed") for j in jobs):
            return jobs
        time.sleep(0.1)
    raise AssertionError("evaluation jobs did not finish")


def test_enqueue_step_eval_creates_cpu_job(client: TestClient, tmp_path: Path) -> None:
    rid, run_id, _ = _make_imported_run(client, tmp_path)
    job = _service(client).enqueue_step_eval(rid, run_id, 0)
    assert job is not None
    assert job.kind == "per_step_eval"
    assert job.queue == "cpu"
    assert job.run_id == run_id
    assert "--fake-report" in job.argv  # settings fixture runs with FAKE_PIPELINE=1
    assert job.argv[0] == str(client.app.state.settings.eval_python)
    assert "LD_PRELOAD" not in job.env
    assert job.argv[job.argv.index("--dataset") + 1] == "step_00"


def test_enqueue_step_eval_skips_when_no_gt(client: TestClient, tmp_path: Path) -> None:
    rid, run_id, result = _make_imported_run(client, tmp_path)
    (result / "merge_0" / "poses_abs_gt.txt").unlink()
    assert _service(client).enqueue_step_eval(rid, run_id, 0) is None


def test_enqueue_final_eval_uses_last_step_when_no_final_dir(client: TestClient, tmp_path: Path) -> None:
    rid, run_id, _ = _make_imported_run(client, tmp_path)
    job = _service(client).enqueue_final_eval(rid, run_id)
    assert job is not None
    assert job.kind == "official_eval"
    assert job.argv[job.argv.index("--dataset") + 1] == "final"
    assert job.argv[job.argv.index("--step_dir") + 1].endswith("merge_0_1")


def test_import_queues_step_and_final_evals_and_fills_summaries(client: TestClient, tmp_path: Path) -> None:
    rid, run_id, _ = _make_imported_run(client, tmp_path)
    jobs = _wait_jobs_done(client, run_id)
    kinds = sorted(j["kind"] for j in jobs)
    assert kinds == ["official_eval", "per_step_eval", "per_step_eval"]
    assert all(j["status"] == "succeeded" for j in jobs)
    run_dir = client.app.state.run_service.runs.run_dir(rid, run_id)
    assert (run_dir / "evaluations" / "per_step" / "step_01" / "eval.json").is_file()
    assert (run_dir / "evaluations" / "final" / "eval.json").is_file()
    summaries = client.get(f"/api/regions/{rid}/runs/{run_id}/summaries").json()["steps"]
    assert [s["ate_trans_rmse"] for s in summaries] == [0.612, 0.612]
    assert all(s["ate_reason"] is None for s in summaries)


def test_eval_job_finish_publishes_run_evaluated(client: TestClient, tmp_path: Path) -> None:
    bus = client.app.state.runner.bus
    seen = []
    orig = bus.publish

    def spy(topic, type_, data):
        seen.append((topic, type_, data))
        return orig(topic, type_, data)

    bus.publish = spy
    rid, run_id, _ = _make_imported_run(client, tmp_path)
    _wait_jobs_done(client, run_id)
    events = [e for e in seen if e[0] == f"run:{run_id}" and e[1] == "run.evaluated"]
    assert len(events) == 3
    assert {e[2]["kind"] for e in events} == {"per_step_eval", "official_eval"}
