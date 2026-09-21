# tests/test_step_reader.py
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
from conftest import SYNTHETIC_MAP


def _two_steps(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """output/merge_000_a (12 nodes) and output/merge_001_b (24 nodes) built the way the fake pipeline does."""
    from navmap_console.jobs.fake_pipeline import build_step_dir
    from navmap_console.jobs.fake_preds import write_fake_preds

    run_dir = tmp_path / "run"
    out = run_dir / "output"
    s0, s1 = out / "merge_000_a", out / "merge_001_b"
    n0 = build_step_dir(None, SYNTHETIC_MAP, s0, 0)
    write_fake_preds(s0, 0, n0, [], seed=0)
    n1 = build_step_dir(s0, SYNTHETIC_MAP, s1, n0)
    write_fake_preds(s1, n0, n1, [], seed=1)
    return run_dir, s0, s1


def _record(index: int, sid: str, dir_name: str, id_offset: int):
    from navmap_console.models import StepRecord

    return StepRecord(index=index, session_id=sid, dir_name=dir_name, status="done", id_offset=id_offset,
                      pgo_error_initial=1.5, pgo_error_final=0.5,
                      started_at="2026-09-18T12:00:00+00:00", finished_at="2026-09-18T12:00:03+00:00")


def test_build_step_dir_is_cumulative(tmp_path: Path):
    _, s0, s1 = _two_steps(tmp_path)
    names0 = [l.split()[0] for l in (s0 / "poses.txt").read_text().splitlines()]
    names1 = [l.split()[0] for l in (s1 / "poses.txt").read_text().splitlines()]
    assert names0 == [f"seq/{i:06d}.color.jpg" for i in range(12)]
    assert names1 == [f"seq/{i:06d}.color.jpg" for i in range(24)]
    assert sorted(p.name for p in (s1 / "seq").iterdir()) == [f"{i:06d}.color.jpg" for i in range(12, 24)]
    odom = np.loadtxt(s1 / "edges_odom.txt", ndmin=2)
    assert odom.shape == (23, 3) and [11.0, 12.0] in odom[:, :2].tolist()  # 11 + 11 + one bridging edge
    assert len((s1 / "intrinsics.txt").read_text().splitlines()) == 24
    for name in ("gnc_weights.txt", "edge_history.txt", "cull_node_info.txt", "not_cull_node_info.txt",
                 "lloc_history.txt", "initial_pose_graph.g2o", "D_matrix.npy", "D_matrix_axes.json",
                 "loop_registry.txt"):
        assert (s1 / "preds" / name).is_file(), name
    assert sorted(p.name for p in (s0 / "preds").iterdir()) == ["initial_pose_graph.g2o"]


def test_build_step_summary_and_arrays(tmp_path: Path):
    from navmap_console.readers import step_reader as sr

    _, s0, s1 = _two_steps(tmp_path)
    summary, arrays = sr.build_step(s1, _record(1, "b", s1.name, 12), [(0, 0), (12, 1)])
    assert summary.num_nodes == 24 and summary.num_new == 12 and summary.id_offset == 12
    assert summary.component_sizes == [24] and summary.edge_counts == {"odom": 23, "covis": 60, "trav": 22}
    assert summary.loops.total == 3 and summary.loops.new == 3 and summary.loops.accepted == 1
    assert summary.loops.rejected_new == 2 and summary.loops.hist == 0
    assert summary.history["vpr"] == 6 and summary.precision == [0.5, 0.5, 1.0]
    assert summary.num_culled == 1 and summary.has_dmatrix and summary.has_pre_pgo
    assert 0 < summary.max_displacement < 0.1 and summary.duration_s == 3.0
    assert summary.pgo_error_initial == 1.5 and summary.status == "done" and summary.session_id == "b"
    assert arrays["node_pos"].shape == (24, 3) and arrays["node_pos_pre"].shape == (24, 3)
    assert arrays["node_step"].tolist() == [0] * 12 + [1] * 12
    flags = arrays["node_flags"]
    assert (flags[:12] & sr.NODE_NEW).sum() == 0 and (flags[12:] & sr.NODE_NEW).astype(bool).all()
    assert (flags & sr.NODE_CULLED).astype(bool).sum() == 1
    assert arrays["edge_odom"].dtype == np.uint32 and arrays["edge_odom"].shape == (23, 2)
    assert arrays["loop_idx"].shape == (3, 2) and arrays["loop_flags"].tolist().count(sr.LOOP_ACCEPTED) == 1
    assert (arrays["loop_flags"] & sr.LOOP_REJECTED_NEW).astype(bool).sum() == 2
    assert (arrays["loop_idx"][:, 0] < 12).all() and (arrays["loop_idx"][:, 1] >= 12).all()
    assert not np.isnan(arrays["loop_conf"]).any()


def test_build_step_first_step_without_preds(tmp_path: Path):
    from navmap_console.readers import step_reader as sr

    _, s0, _ = _two_steps(tmp_path)
    summary, arrays = sr.build_step(s0, _record(0, "a", s0.name, 0), [(0, 0)])
    assert summary.num_nodes == 12 and summary.num_new == 12 and summary.loops.total == 0
    assert summary.has_pre_pgo and not summary.has_dmatrix and summary.history["vpr"] == 0
    assert arrays["loop_idx"].shape == (0, 2) and (arrays["node_flags"] & sr.NODE_NEW).astype(bool).all()


def test_hist_loops_flagged(tmp_path: Path):
    from navmap_console.jobs.fake_pipeline import build_step_dir
    from navmap_console.jobs.fake_preds import write_fake_preds
    from navmap_console.readers import step_reader as sr

    out = tmp_path / "output"
    s0 = out / "merge_000_a"
    n0 = build_step_dir(None, SYNTHETIC_MAP, s0, 0)
    s1 = out / "merge_001_b"
    n1 = build_step_dir(s0, SYNTHETIC_MAP, s1, n0)
    acc = write_fake_preds(s1, n0, n1, [], seed=1)
    s2 = out / "merge_002_c"
    n2 = build_step_dir(s1, SYNTHETIC_MAP, s2, n1)
    write_fake_preds(s2, n1, n2, [acc], seed=2)
    summary, arrays = sr.build_step(s2, _record(2, "c", s2.name, 24), [(0, 0), (12, 1), (24, 2)])
    assert summary.loops.total == 4 and summary.loops.hist == 1 and summary.loops.accepted == 2
    hist_rows = arrays["loop_flags"] & sr.LOOP_HIST
    assert hist_rows.astype(bool).sum() == 1
    assert arrays["node_step"].tolist() == [0] * 12 + [1] * 12 + [2] * 12
    assert summary.num_nodes == 36 and (arrays["node_comp"] == 0).all()


def test_step_cache_reuses_and_invalidates(tmp_path: Path):
    from navmap_console.readers.scene_bundle import decode_scene_bundle
    from navmap_console.readers.step_reader import StepCache
    from navmap_console.store import read_json

    run_dir, _, s1 = _two_steps(tmp_path)
    cache = StepCache(run_dir)
    rec = _record(1, "b", s1.name, 12)
    summary, data = cache.get(s1, rec, [(0, 0), (12, 1)])
    sj = run_dir / "cache" / "steps" / "merge_001_b.summary.json"
    sb = run_dir / "cache" / "steps" / "merge_001_b.scene.bin"
    assert sj.is_file() and sb.is_file() and sb.read_bytes() == data
    assert decode_scene_bundle(data)["node_pos"].shape == (24, 3)
    first_source = read_json(sj)["source"]
    summary2, data2 = cache.get(s1, rec, [(0, 0), (12, 1)])
    assert data2 == data and summary2 == summary and read_json(sj)["source"] == first_source
    poses = s1 / "poses.txt"
    st = poses.stat()
    os.utime(poses, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    cache.get(s1, rec, [(0, 0), (12, 1)])
    assert read_json(sj)["source"] != first_source


def test_fake_pipeline_prints_cumulative_step_done(tmp_path: Path, capsys):
    from navmap_console.jobs import fake_pipeline

    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(SYNTHETIC_MAP, a)
    shutil.copytree(SYNTHETIC_MAP, b)
    lst = tmp_path / "list.txt"
    lst.write_text(f"{a}\n{b}\n")
    os.environ["NAVMAP_CONSOLE_FAKE_STEP_SECONDS"] = "0"
    assert fake_pipeline.main(["--submap_list", str(lst), "--result_dir", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "STEP_DONE index=0 sid=a" in out and "id_offset=0 odom_nodes=12" in out
    assert "id_offset=12 odom_nodes=24 covis_nodes=24 components=1 registry=1" in out
    assert (tmp_path / "out" / "merge_finalmap").resolve() == (tmp_path / "out" / "merge_001_b").resolve()


# --- per-step ATE fields (M5) -------------------------------------------------

def _make_step(tmp_path: Path, with_gt: bool) -> Path:
    """A minimal merge_* directory under a fake run dir; no preds/ files needed."""
    run_dir = tmp_path / "run"
    step = run_dir / "output" / "merge_0"
    (step / "seq").mkdir(parents=True)
    (step / "preds").mkdir()
    lines = [f"seq/{i:06d}.color.jpg 1 0 0 0 {i}.0 0 0" for i in range(3)]
    (step / "poses.txt").write_text("\n".join(lines) + "\n")
    (step / "intrinsics.txt").write_text("\n".join(f"seq/{i:06d}.color.jpg 1 1 1 1 1 1" for i in range(3)) + "\n")
    for name in ("edges_odom.txt", "edges_covis.txt", "edges_trav.txt"):
        (step / name).write_text("")
    if with_gt:
        (step / "poses_abs_gt.txt").write_text("\n".join(lines) + "\n")
        (step / "timestamps.txt").write_text(
            "\n".join(f"seq/{i:06d}.color.jpg {1000 + i}.0" for i in range(3)) + "\n")
    return step


def test_build_step_ate_fields_no_gt(tmp_path: Path):
    from navmap_console.models import StepRecord
    from navmap_console.readers import step_reader as sr

    step = _make_step(tmp_path, with_gt=False)
    rec = StepRecord(index=0, session_id="s", status="done")
    summary, _ = sr.build_step(step, rec, [(0, 0)])
    assert summary.ate_trans_rmse is None
    assert summary.ate_reason == "no gt"


def test_build_step_ate_fields_pending(tmp_path: Path):
    from navmap_console.models import StepRecord
    from navmap_console.readers import step_reader as sr

    step = _make_step(tmp_path, with_gt=True)
    rec = StepRecord(index=0, session_id="s", status="done")
    summary, _ = sr.build_step(step, rec, [(0, 0)])
    assert summary.ate_trans_rmse is None
    assert summary.ate_reason == "pending"


def test_build_step_ate_fields_from_eval_json(tmp_path: Path):
    import json

    from navmap_console.models import StepRecord
    from navmap_console.readers import step_reader as sr
    from navmap_console.readers.ate import step_eval_path

    step = _make_step(tmp_path, with_gt=True)
    p = step_eval_path(step.parent.parent, 0)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"status": "succeeded", "ate_trans": 0.612, "ate_rot": 1.23, "frames": 3}))
    rec = StepRecord(index=0, session_id="s", status="done")
    summary, _ = sr.build_step(step, rec, [(0, 0)])
    assert summary.ate_trans_rmse == 0.612
    assert summary.ate_rot_rmse == 1.23
    assert summary.ate_frames == 3
    assert summary.ate_reason is None


def test_step_cache_invalidates_when_eval_json_appears(tmp_path: Path):
    import json

    from navmap_console.models import StepRecord
    from navmap_console.readers.ate import step_eval_path
    from navmap_console.readers.step_reader import StepCache

    step = _make_step(tmp_path, with_gt=True)
    run_dir = step.parent.parent
    rec = StepRecord(index=0, session_id="s", status="done")
    cache = StepCache(run_dir)
    first, _ = cache.get(step, rec, [(0, 0)])
    assert first.ate_reason == "pending"
    p = step_eval_path(run_dir, 0)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"status": "succeeded", "ate_trans": 0.612, "ate_rot": 1.23, "frames": 3}))
    second, _ = cache.get(step, rec, [(0, 0)])
    assert second.ate_trans_rmse == 0.612
