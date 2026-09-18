import shutil
import time
from pathlib import Path

from conftest import SYNTHETIC_MAP


def _region_with_sessions(client, tmp_path: Path, n: int = 2):
    rid = client.post("/api/regions", json={"name": "r"}).json()["id"]
    sids = []
    for i in range(n):
        d = tmp_path / f"ext_{i}"
        shutil.copytree(SYNTHETIC_MAP, d)
        r = client.post(f"/api/regions/{rid}/sessions/register", json={"path": str(d), "name": f"s{i}"})
        assert r.status_code == 201, r.text
        sids.append(r.json()["id"])
    return rid, sids


def _wait(client, rid, run_id, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = client.get(f"/api/regions/{rid}/runs/{run_id}").json()
        if d["run"]["status"] in ("succeeded", "failed", "cancelled", "orphaned"):
            return d
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_params_schema_endpoint(client):
    r = client.get("/api/params/merge")
    assert r.status_code == 200 and any(p["name"] == "pgo_robust" for p in r.json())


def test_run_lifecycle_over_http(client, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NAVMAP_CONSOLE_FAKE_STEP_SECONDS", "0.05")
    rid, sids = _region_with_sessions(client, tmp_path)
    r = client.post(f"/api/regions/{rid}/runs", json={"name": "n", "session_ids": sids, "params": {"use_td": False}})
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]
    assert client.post(f"/api/regions/{rid}/runs", json={"session_ids": sids}).status_code == 409
    d = _wait(client, rid, run_id)
    assert d["run"]["status"] == "succeeded" and len(d["steps"]) == 2 and d["job"]["status"] == "succeeded"
    assert client.get(f"/api/regions/{rid}/runs/{run_id}/steps").json()[1]["index"] == 1
    assert [x["id"] for x in client.get(f"/api/regions/{rid}/runs").json()] == [run_id]
    region = client.get(f"/api/regions/{rid}").json()
    assert region["head"]["run_id"] == run_id and region["run_count"] == 1
    log = client.get(f"/api/jobs/{d['job']['id']}/log", params={"after": 1}).json()
    assert log["next"] == log["total"] and any("STEP_DONE" in l for l in log["lines"])
    assert client.get("/api/health").json()["queues"] == {"gpu": 0, "cpu": 0}


def test_run_errors(client, tmp_path: Path):
    rid, sids = _region_with_sessions(client, tmp_path, n=1)
    assert client.post(f"/api/regions/{rid}/runs", json={"session_ids": ["nope"]}).status_code == 400
    assert client.post(f"/api/regions/{rid}/runs", json={"session_ids": sids, "params": {"x": 1}}).status_code == 400
    assert client.post(f"/api/regions/{rid}/runs", json={"session_ids": []}).status_code == 422
    assert client.post("/api/regions/reg_missing/runs", json={"session_ids": sids}).status_code == 404
    assert client.get(f"/api/regions/{rid}/runs/run_missing").status_code == 404
    assert client.get("/api/jobs/job_missing").status_code == 404


def test_cancel_run(client, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NAVMAP_CONSOLE_FAKE_STEP_SECONDS", "5")
    rid, sids = _region_with_sessions(client, tmp_path)
    run_id = client.post(f"/api/regions/{rid}/runs", json={"session_ids": sids}).json()["id"]
    for _ in range(50):
        if client.get(f"/api/regions/{rid}/runs/{run_id}").json()["run"]["status"] == "running":
            break
        time.sleep(0.1)
    assert client.post(f"/api/regions/{rid}/runs/{run_id}/cancel").status_code == 200
    d = _wait(client, rid, run_id)
    assert d["run"]["status"] == "cancelled" and d["job"]["status"] == "cancelled"
    jobs = client.get("/api/jobs", params={"status": "cancelled"}).json()
    assert [j["id"] for j in jobs] == [d["job"]["id"]]
