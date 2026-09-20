# tests/test_results_api.py
import io
import json
from pathlib import Path

from conftest import SYNTHETIC_MAP
from PIL import Image

T0, T1 = "2026-09-18T12:00:00+00:00", "2026-09-18T12:00:03+00:00"


def _make_run(client, settings):
    from navmap_console.jobs.fake_pipeline import build_step_dir
    from navmap_console.jobs.fake_preds import write_fake_preds
    from navmap_console.models import Run, StepRecord
    from navmap_console.services.runs import RunStore

    rid = client.post("/api/regions", json={"name": "r"}).json()["id"]
    store = RunStore(settings.regions_dir)
    run_id = "run_20260918_120000_test"
    out = store.run_dir(rid, run_id) / "output"
    s0, s1 = out / "merge_000_a", out / "merge_001_b"
    n0 = build_step_dir(None, SYNTHETIC_MAP, s0, 0)
    write_fake_preds(s0, 0, n0, [], seed=0)
    n1 = build_step_dir(s0, SYNTHETIC_MAP, s1, n0)
    write_fake_preds(s1, n0, n1, [], seed=1)
    store.save(Run(id=run_id, region_id=rid, name="t", status="succeeded", session_ids=["a", "b"],
                   num_steps_expected=2, last_step_index=1))
    store.write_steps(rid, run_id, [
        StepRecord(index=0, session_id="a", dir_name="merge_000_a", status="done", id_offset=0, odom_nodes=n0,
                   pgo_error_initial=1.0, pgo_error_final=0.5, started_at=T0, finished_at=T1),
        StepRecord(index=1, session_id="b", dir_name="merge_001_b", status="done", id_offset=n0, odom_nodes=n1,
                   pgo_error_initial=2.0, pgo_error_final=0.7, started_at=T0, finished_at=T1),
        StepRecord(index=2, session_id="c", status="running", started_at=T1),
    ])
    viz = out / "rerun_viz"
    viz.mkdir()
    rows = [{"demo_step": s, "merge_step": s, "stage": "vpr", "event_type": t, "submap_id": s, "keyframe_id": None,
             "payload": {"i": i}, "artifacts": {}} for i, (s, t) in enumerate([(0, "vpr_candidate"),
                                                                                 (1, "vpr_candidate"),
                                                                                 (1, "map_committed")])]
    (viz / "demo_events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rid, run_id


def test_summaries_and_scene(client, settings):
    from navmap_console.readers.scene_bundle import decode_scene_bundle

    rid, run_id = _make_run(client, settings)
    base = f"/api/regions/{rid}/runs/{run_id}"
    steps = client.get(f"{base}/summaries").json()["steps"]
    assert [s["index"] for s in steps] == [0, 1] and steps[1]["num_nodes"] == 24 and steps[1]["loops"]["total"] == 3
    r = client.get(f"{base}/steps/1/scene.bin")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/octet-stream")
    assert decode_scene_bundle(r.content)["node_pos"].shape == (24, 3)
    assert client.get(f"{base}/steps/2/scene.bin").status_code == 404  # running step has no directory yet
    assert client.get(f"{base}/steps/9/scene.bin").status_code == 404
    assert client.get(f"/api/regions/{rid}/runs/run_nope/summaries").status_code == 404


def test_dmatrix_endpoints(client, settings):
    rid, run_id = _make_run(client, settings)
    base = f"/api/regions/{rid}/runs/{run_id}"
    j = client.get(f"{base}/steps/1/dmatrix.json").json()
    assert j["rows"] == "db" and len(j["row_node_ids"]) == 12 and len(j["col_node_ids"]) == 12
    assert len(j["candidates"]) == 6 and len(j["factors"]) == 3 and j["vmin"] < j["vmax"]
    assert all(c["query"] >= 12 for c in j["candidates"]) and sum(f["accepted"] for f in j["factors"]) == 1
    png = client.get(f"{base}/steps/1/dmatrix.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(png.content)) as im:
        assert im.size == (12, 12) and im.mode == "L"
    assert client.get(f"{base}/steps/0/dmatrix.json").status_code == 404


def test_culling_and_node_detail(client, settings):
    rid, run_id = _make_run(client, settings)
    base = f"/api/regions/{rid}/runs/{run_id}"
    c = client.get(f"{base}/steps/1/culling.json").json()
    assert len(c["culled"]) == 1 and c["culled"][0]["method"] == "culled_by_forward"
    assert c["culled"][0]["node_id"] == 23 and c["culled"][0]["image_url"].endswith("/nodes/23/image")
    assert len(c["kept"]) == 1 and c["kept"][0]["node_id"] == 12
    d = client.get(f"{base}/steps/1/nodes/13").json()
    assert d["step"] == 1 and d["session_id"] == "b" and d["frame"] == "seq/000013.color.jpg"
    assert d["degree"]["odom"] == 2 and len(d["pos"]) == 3 and d["image_url"].endswith("/nodes/13/image")
    assert client.get(f"{base}/steps/1/nodes/5").json()["session_id"] == "a"
    assert client.get(f"{base}/steps/1/nodes/99").status_code == 404
    assert client.get(f"{base}/steps/0/nodes/13").status_code == 404


def test_images_preds_and_events(client, settings):
    rid, run_id = _make_run(client, settings)
    base = f"/api/regions/{rid}/runs/{run_id}"
    r = client.get(f"{base}/nodes/13/image", params={"w": 64})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(r.content)) as im:
        assert im.width == 64
    assert client.get(f"{base}/nodes/13/image", params={"w": 8}).status_code == 422
    assert client.get(f"{base}/nodes/999/image").status_code == 404
    r = client.get(f"{base}/pairs/3/13/image", params={"w": 64})
    with Image.open(io.BytesIO(r.content)) as im:
        assert im.width == 132
    assert client.get(f"{base}/steps/1/preds/D_matrix_axes.json").status_code == 200
    assert client.get(f"{base}/steps/1/preds/../poses.txt").status_code in (400, 404)
    assert client.get(f"{base}/steps/1/preds/missing.txt").status_code == 404
    ev = client.get(f"{base}/events", params={"step": 1}).json()
    assert [e["event_type"] for e in ev] == ["vpr_candidate", "map_committed"] and ev[1]["payload"] == {"omitted": True}
    assert len(client.get(f"{base}/events", params={"types": "vpr_candidate"}).json()) == 2


def test_dmatrix_jpg_fallback_for_legacy_results(client, settings):
    """旧结果没有 D_matrix_*.npy，只有 matplotlib 带坐标轴的 jpg：dmatrix.png 直接回退 serve jpg。"""
    from navmap_console.services.runs import RunStore

    rid, run_id = _make_run(client, settings)
    step_dir = RunStore(settings.regions_dir).run_dir(rid, run_id) / "output" / "merge_001_b" / "preds"
    for p in step_dir.glob("D_matrix*.npy"):
        p.unlink()
    Image.new("L", (12, 12), 128).save(step_dir / "D_matrix_vpr.jpg")
    base = f"/api/regions/{rid}/runs/{run_id}"
    r = client.get(f"{base}/steps/1/dmatrix.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(r.content)) as im:
        assert im.size == (12, 12)
    # json 仍是 404：前端以此隐藏候选叠加
    assert client.get(f"{base}/steps/1/dmatrix.json").status_code == 404
