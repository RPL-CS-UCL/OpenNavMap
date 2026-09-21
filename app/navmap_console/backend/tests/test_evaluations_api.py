"""Evaluation endpoints: manual re-run, list/detail, report file serving."""
from pathlib import Path

from test_evaluations_service import _make_imported_run, _wait_jobs_done


def test_manual_evaluate_queues_job(client, tmp_path: Path) -> None:
    rid, run_id, _ = _make_imported_run(client, tmp_path)
    res = client.post(f"/api/regions/{rid}/runs/{run_id}/evaluate")
    assert res.status_code == 200
    assert res.json()["kind"] == "official_eval"
    assert client.post(f"/api/regions/{rid}/runs/nope/evaluate").status_code == 404


def test_manual_evaluate_404_without_gt(client, tmp_path: Path) -> None:
    rid, run_id, result = _make_imported_run(client, tmp_path)
    for step in result.glob("merge_*"):
        (step / "poses_abs_gt.txt").unlink()
    assert client.post(f"/api/regions/{rid}/runs/{run_id}/evaluate").status_code == 404


def test_list_and_detail_after_fake_eval(client, tmp_path: Path) -> None:
    rid, run_id, _ = _make_imported_run(client, tmp_path)
    _wait_jobs_done(client, run_id)
    res = client.get(f"/api/regions/{rid}/runs/{run_id}/evaluations")
    assert res.status_code == 200
    body = res.json()
    assert body["evaluating"] is False
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["eid"] == "final" and item["status"] == "succeeded"
    assert item["ate_trans"] == 0.612 and item["job"]["kind"] == "official_eval"
    detail = client.get(f"/api/regions/{rid}/runs/{run_id}/evaluations/final").json()
    assert detail["ate_trans"] == 0.612
    assert any("rmse" in f for f in detail["report_files"])


def test_list_is_empty_before_any_final_job(client, tmp_path: Path) -> None:
    rid, run_id, result = _make_imported_run(client, tmp_path)
    _wait_jobs_done(client, run_id)
    run_dir = client.app.state.run_service.runs.run_dir(rid, run_id)
    # a second region-less run dir with no eval jobs: list is empty, detail 404
    rid2 = client.post("/api/regions", json={"name": "r2"}).json()["id"]
    for step in result.glob("merge_*"):
        (step / "poses_abs_gt.txt").unlink()
    run2 = client.post(f"/api/regions/{rid2}/runs/import", json={"result_dir": str(result)}).json()
    assert client.get(f"/api/regions/{rid2}/runs/{run2['id']}/evaluations").json() == {"items": [], "evaluating": False}
    assert client.get(f"/api/regions/{rid2}/runs/{run2['id']}/evaluations/final").status_code == 404
    assert run_dir.is_dir()


def test_report_file_served_and_escape_refused(client, tmp_path: Path) -> None:
    rid, run_id, _ = _make_imported_run(client, tmp_path)
    _wait_jobs_done(client, run_id)
    run_dir = client.app.state.run_service.runs.run_dir(rid, run_id)
    report = run_dir / "evaluations" / "final" / "report"
    (report / "plot.pdf").write_bytes(b"%PDF-fake")
    res = client.get(f"/api/regions/{rid}/runs/{run_id}/evaluations/final/files/plot.pdf")
    assert res.status_code == 200 and res.content == b"%PDF-fake"
    assert res.headers["content-disposition"].startswith("inline")  # browsers render the preview instead of saving it
    # nested report files are addressed by their path below report/
    nested = next(report.rglob("laptop_translation_rmse_*.txt")).relative_to(report)
    assert client.get(f"/api/regions/{rid}/runs/{run_id}/evaluations/final/files/{nested}").status_code == 200
    assert client.get(f"/api/regions/{rid}/runs/{run_id}/evaluations/final/files/../eval.json").status_code in (400, 404)
    assert client.get(f"/api/regions/{rid}/runs/{run_id}/evaluations/final/files/..%2Feval.json").status_code in (400, 404)
    assert client.get(f"/api/regions/{rid}/runs/{run_id}/evaluations/final/files/missing.txt").status_code == 404


def test_unknown_eid_404(client, tmp_path: Path) -> None:
    rid, run_id, _ = _make_imported_run(client, tmp_path)
    assert client.get(f"/api/regions/{rid}/runs/{run_id}/evaluations/nope").status_code == 404
    assert client.get(f"/api/regions/{rid}/runs/{run_id}/evaluations/nope/files/x.txt").status_code == 404
