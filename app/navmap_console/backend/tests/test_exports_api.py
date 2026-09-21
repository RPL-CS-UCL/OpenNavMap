"""Export endpoints: create/list/download/delete lifecycle, 409 while packing, prune to 3."""
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from test_evaluations_service import _make_imported_run, _wait_jobs_done


def _imported_run_with_images(client: TestClient, tmp_path: Path):
    rid, run_id, result = _make_imported_run(client, tmp_path)
    for step in result.glob("merge_*"):
        for k in range(2):
            (step / "seq" / f"{k:06d}.color.jpg").write_bytes(b"\xff\xd8fakejpg")
    _wait_jobs_done(client, run_id)  # let the import's eval jobs drain first
    return rid, run_id


def _exports(client: TestClient, rid: str, run_id: str) -> list:
    return client.get(f"/api/regions/{rid}/runs/{run_id}/exports").json()["items"]


def test_create_list_download_delete(client: TestClient, tmp_path: Path) -> None:
    rid, run_id = _imported_run_with_images(client, tmp_path)
    res = client.post(f"/api/regions/{rid}/runs/{run_id}/exports", json={"kind": "map"})
    assert res.status_code == 201, res.text
    job = res.json()
    assert job["kind"] == "export" and job["queue"] == "cpu" and job["run_id"] == run_id
    items = _exports(client, rid, run_id)
    assert len(items) == 1 and items[0]["kind"] == "map" and items[0]["job_id"] == job["id"]
    name = items[0]["name"]
    assert name.startswith("map_")
    _wait_jobs_done(client, run_id)
    item = _exports(client, rid, run_id)[0]
    assert item["status"] == "succeeded", item
    assert item["size"] > 0 and len(item["sha256"]) == 64
    assert item["verified"] is True  # map bundles are re-extracted and checked
    dl = client.get(f"/api/regions/{rid}/runs/{run_id}/exports/{name}/download")
    assert dl.status_code == 200 and dl.content[:2] == b"\x1f\x8b"
    assert client.delete(f"/api/regions/{rid}/runs/{run_id}/exports/{name}").status_code == 204
    assert _exports(client, rid, run_id) == []
    assert client.get(f"/api/regions/{rid}/runs/{run_id}/exports/{name}/download").status_code == 404
    assert client.delete(f"/api/regions/{rid}/runs/{run_id}/exports/{name}").status_code == 404


def test_create_404_and_400(client: TestClient, tmp_path: Path) -> None:
    rid, run_id = _imported_run_with_images(client, tmp_path)
    assert client.post(f"/api/regions/{rid}/runs/nope/exports", json={"kind": "map"}).status_code == 404
    assert client.post(f"/api/regions/{rid}/runs/{run_id}/exports", json={"kind": "zip"}).status_code == 422
    # report bundle needs a finished final evaluation report directory
    assert client.get(f"/api/regions/{rid}/runs/nope/exports").status_code == 404


def test_preds_bundle_passes_steps(client: TestClient, tmp_path: Path) -> None:
    rid, run_id = _imported_run_with_images(client, tmp_path)
    res = client.post(f"/api/regions/{rid}/runs/{run_id}/exports", json={"kind": "preds", "steps": [1]})
    assert res.status_code == 201
    argv = res.json()["argv"]
    assert argv[argv.index("--steps") + 1] == "1"
    assert _exports(client, rid, run_id)[0]["steps"] == [1]


def test_delete_409_while_packing(client: TestClient, tmp_path: Path) -> None:
    rid, run_id = _imported_run_with_images(client, tmp_path)
    svc = client.app.state.export_service
    job = svc.create(rid, run_id, "map")
    name = job.argv[job.argv.index("--name") + 1]
    job.status = "running"
    client.app.state.job_store.save(job)
    assert client.delete(f"/api/regions/{rid}/runs/{run_id}/exports/{name}").status_code == 409


def test_prune_keeps_newest_three(client: TestClient, tmp_path: Path) -> None:
    rid, run_id = _imported_run_with_images(client, tmp_path)
    svc = client.app.state.export_service
    exports_dir = svc.exports_dir(rid, run_id)
    exports_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (exports_dir / f"map_{i}.tar.gz").write_bytes(b"x")
        (exports_dir / f"map_{i}.json").write_text(
            json.dumps({"name": f"map_{i}", "kind": "map", "status": "succeeded",
                        "created_at": f"2026-09-20T0{i}:00:00+00:00"}))
    svc.prune(rid, run_id, keep=3)
    names = sorted(p.name for p in exports_dir.glob("map_*"))
    assert names == ["map_2.json", "map_2.tar.gz", "map_3.json", "map_3.tar.gz", "map_4.json", "map_4.tar.gz"]


def test_prune_runs_after_export_job(client: TestClient, tmp_path: Path) -> None:
    rid, run_id = _imported_run_with_images(client, tmp_path)
    for _ in range(4):
        assert client.post(f"/api/regions/{rid}/runs/{run_id}/exports", json={"kind": "preds"}).status_code == 201
        time.sleep(1.1)  # export names carry a per-second timestamp
    _wait_jobs_done(client, run_id, timeout=30)
    deadline = time.time() + 5  # prune runs in the on_finished hook, just after the job status is saved
    while time.time() < deadline and len(_exports(client, rid, run_id)) != 3:
        time.sleep(0.1)
    items = _exports(client, rid, run_id)
    assert len(items) == 3 and all(i["status"] == "succeeded" for i in items)
