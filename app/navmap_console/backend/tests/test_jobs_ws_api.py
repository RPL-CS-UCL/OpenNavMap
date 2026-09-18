import shutil
import time
from pathlib import Path

from conftest import SYNTHETIC_MAP


def test_ws_job_snapshot_log_and_run_events(client, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NAVMAP_CONSOLE_FAKE_STEP_SECONDS", "0.3")
    rid = client.post("/api/regions", json={"name": "r"}).json()["id"]
    d = tmp_path / "ext"
    shutil.copytree(SYNTHETIC_MAP, d)
    sid = client.post(f"/api/regions/{rid}/sessions/register", json={"path": str(d), "name": "s"}).json()["id"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"op": "sub", "topic": "jobs"})
        run = client.post(f"/api/regions/{rid}/runs", json={"session_ids": [sid]}).json()
        first = ws.receive_json()
        assert first["topic"] == "jobs" and first["type"] == "job.state" and first["data"]["run_id"] == run["id"]
        jid = first["data"]["id"]
        ws.send_json({"op": "sub", "topic": f"job:{jid}"})
        ws.send_json({"op": "sub", "topic": f"run:{run['id']}"})
        seen = {"job.snapshot": 0, "job.log": 0, "job.progress": 0, "run.step_completed": 0, "run.state": 0}
        final_state = None
        t0 = time.time()
        while time.time() - t0 < 20:
            m = ws.receive_json()
            if m["type"] in seen:
                seen[m["type"]] += 1
            if m["type"] == "job.snapshot":
                assert "job" in m["data"] and isinstance(m["data"]["lines"], list) and "next_seq" in m["data"]
            if m["topic"] == f"run:{run['id']}" and m["type"] == "run.state":
                final_state = m["data"]["status"]
                if final_state in ("succeeded", "failed"):
                    break
        assert final_state == "succeeded", seen
        assert seen["job.snapshot"] == 1 and seen["job.log"] >= 1 and seen["run.step_completed"] == 1
        ws.send_json({"op": "unsub", "topic": "jobs"})
        ws.send_json({"op": "bogus"})
        err = ws.receive_json()
        assert err["type"] == "error"
