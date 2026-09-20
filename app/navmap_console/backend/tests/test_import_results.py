# tests/test_import_results.py
import shutil
from pathlib import Path

import pytest
from conftest import SYNTHETIC_MAP


def _result_dir(tmp_path: Path, style: str = "cumulative") -> Path:
    """Three cumulative steps built with the fake pipeline helpers, plus a merge_finalmap symlink and a junk dir."""
    from navmap_console.jobs.fake_pipeline import build_step_dir
    from navmap_console.jobs.fake_preds import write_fake_preds

    root = tmp_path / "s00000_results_x"
    root.mkdir(parents=True, exist_ok=True)
    names = {"cumulative": ["merge_0", "merge_0_1", "merge_0_1_2"],
             "indexed": ["merge_000_0", "merge_001_1", "merge_002_2"]}[style]
    prev, offset, hist = None, 0, []
    for k, name in enumerate(names):
        d = root / name
        n = build_step_dir(prev, SYNTHETIC_MAP, d, offset)
        db, q = write_fake_preds(d, offset, n, hist, seed=k)
        if db >= 0:
            hist.append((db, q))
        prev, offset = d, n
    (root / "merge_finalmap").symlink_to(root / names[-1], target_is_directory=True)
    (root / "merge_junk").mkdir()  # no poses.txt: ignored
    sessions = tmp_path / "s00000_aria_data_390"
    for tag in ("0", "1", "2"):
        shutil.copytree(SYNTHETIC_MAP, sessions / tag)
    return root


def test_scan_result_dir_cumulative_and_indexed(tmp_path: Path):
    from navmap_console.services.import_results import scan_result_dir

    root = _result_dir(tmp_path)
    assert scan_result_dir(root) == [(0, "0", "merge_0"), (1, "1", "merge_0_1"), (2, "2", "merge_0_1_2")]
    root2 = _result_dir(tmp_path / "b", "indexed")
    assert [x[:2] for x in scan_result_dir(root2)] == [(0, "0"), (1, "1"), (2, "2")]
    shutil.rmtree(root2 / "merge_001_1")
    with pytest.raises(ValueError, match="contiguous"):
        scan_result_dir(root2)
    with pytest.raises(ValueError, match="no merge_"):
        scan_result_dir(tmp_path / "b")


def test_import_results_over_http(client, settings, tmp_path: Path):
    root = _result_dir(tmp_path)
    rid = client.post("/api/regions", json={"name": "r"}).json()["id"]
    r = client.post(f"/api/regions/{rid}/runs/import",
                    json={"result_dir": str(root), "sessions_root": str(tmp_path / "s00000_aria_data_390"),
                          "promote": True})
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["kind"] == "imported" and run["status"] == "succeeded" and run["num_steps_expected"] == 3
    assert run["last_step_index"] == 2 and run["name"] == "s00000_results_x"
    d = client.get(f"/api/regions/{rid}/runs/{run['id']}").json()
    steps = d["steps"]
    assert [s["index"] for s in steps] == [0, 1, 2] and [s["id_offset"] for s in steps] == [0, 12, 24]
    assert steps[2]["odom_nodes"] == 36 and steps[2]["components"] == 1 and steps[2]["registry_edges"] == 2
    assert steps[0]["dir_name"] == "merge_0" and steps[0]["registry_edges"] == 0
    sessions = client.get(f"/api/regions/{rid}/sessions").json()
    assert sorted(s["name"] for s in sessions) == ["0", "1", "2"] and all(s["source"] == "imported" for s in sessions)
    assert [s["session_id"] for s in steps] == [s2["id"] for s2 in sorted(sessions, key=lambda s: s["name"])]
    region = client.get(f"/api/regions/{rid}").json()
    assert region["head"]["run_id"] == run["id"] and region["head"]["step_index"] == 2
    out = settings.regions_dir / rid / "runs" / run["id"] / "output"
    assert out.is_symlink() and (out / "merge_0_1_2" / "poses.txt").is_file()
    # the viewer endpoints work on the imported run
    assert len(client.get(f"/api/regions/{rid}/runs/{run['id']}/summaries").json()["steps"]) == 3
    assert client.get(f"/api/regions/{rid}/runs/{run['id']}/steps/2/scene.bin").status_code == 200


def test_import_rejects_outside_and_missing(client, tmp_path: Path):
    rid = client.post("/api/regions", json={"name": "r"}).json()["id"]
    assert client.post(f"/api/regions/{rid}/runs/import", json={"result_dir": "/etc"}).status_code == 403
    assert client.post(f"/api/regions/{rid}/runs/import", json={"result_dir": str(tmp_path / "nope")}).status_code == 404
    (tmp_path / "empty").mkdir()
    assert client.post(f"/api/regions/{rid}/runs/import", json={"result_dir": str(tmp_path / "empty")}).status_code == 400
