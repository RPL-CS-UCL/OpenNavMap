import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from map_merge_pack import (  # noqa: E402
    REGISTRY_HEADER, pose_to_vec, read_loop_registry, vec_to_pose, write_loop_registry,
)


def _pose(yaw_deg: float, t) -> np.ndarray:
    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


def test_pose_vec_roundtrip():
    T = _pose(30.0, [1.0, -2.0, 0.5])
    t, q = pose_to_vec(T)
    assert q.shape == (4,) and abs(np.linalg.norm(q) - 1) < 1e-9
    assert np.allclose(vec_to_pose(t, q), T, atol=1e-9)


def test_registry_13_column_roundtrip(tmp_path: Path):
    registry = {
        (4, 48): {"T_AB": _pose(10.0, [0.1, 0.2, 0.3]), "conf": 2.181, "first_step": 1,
                  "reject_count": 0, "last_weight": 1.0},
        (5, 49): {"T_AB": _pose(-5.0, [1, 2, 3]), "conf": 1.94, "first_step": 2,
                  "reject_count": 1, "last_weight": 0.25},
    }
    path = tmp_path / "loop_registry.txt"
    write_loop_registry(path, registry)
    lines = path.read_text().splitlines()
    assert lines[0] == REGISTRY_HEADER
    assert len(lines[1].split(",")) == 13
    back = read_loop_registry(path)
    assert set(back) == set(registry)
    for key, rec in registry.items():
        assert np.allclose(back[key]["T_AB"], rec["T_AB"], atol=1e-8)
        assert back[key]["first_step"] == rec["first_step"]
        assert back[key]["reject_count"] == rec["reject_count"]
        assert abs(back[key]["last_weight"] - rec["last_weight"]) < 1e-6


def test_registry_reads_legacy_6_columns(tmp_path: Path):
    path = tmp_path / "loop_registry.txt"
    path.write_text("# a_id,b_id,conf,first_step,reject_count,last_weight\n4,48,2.181,1,0,1.000000\n")
    back = read_loop_registry(path)
    assert back[(4, 48)]["T_AB"] is None and back[(4, 48)]["conf"] == pytest.approx(2.181)


def test_write_rejects_missing_pose(tmp_path: Path):
    with pytest.raises(ValueError):
        write_loop_registry(tmp_path / "x.txt", {(1, 2): {"T_AB": None, "conf": 1.0, "first_step": 0,
                                                            "reject_count": 0, "last_weight": 1.0}})


def test_empty_registry(tmp_path: Path):
    path = tmp_path / "loop_registry.txt"
    write_loop_registry(path, {})
    assert read_loop_registry(path) == {}


def test_pipeline_save_and_load_registry(tmp_path: Path):
    pytest.importorskip("gtsam")
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "third_party" / "litevloc_code" / "python"))
    from map_merge_pipeline import MergePipeline

    merger = MergePipeline.__new__(MergePipeline)
    merger.loop_edge_registry = {
        (1, 9): {"T_AB": _pose(3.0, [0.5, 0, 0]), "conf": 1.5, "first_step": 0, "reject_count": 0, "last_weight": 1.0}
    }
    path = tmp_path / "loop_registry.txt"
    merger.save_loop_registry(str(path))
    loaded = merger.load_loop_registry(str(path))
    assert np.allclose(loaded[(1, 9)]["T_AB"], merger.loop_edge_registry[(1, 9)]["T_AB"], atol=1e-8)

    path.write_text("# legacy\n1,9,1.5,0,0,1.0\n")
    with pytest.raises(ValueError, match="recover-registry"):
        merger.load_loop_registry(str(path))


# --- image index / consolidation / g2o recovery (Task 14) ------------------

from map_merge_pack import (  # noqa: E402
    build_image_index, consolidate_map, recover_registry_from_g2o,
)

SYNTHETIC = REPO / "python" / "visualization" / "example_data" / "synthetic_map"


def _fake_step(tmp_path: Path, name: str, image_ids) -> Path:
    """Copy synthetic_map's text files into a step dir holding only some images."""
    step = tmp_path / name
    (step / "seq").mkdir(parents=True)
    (step / "preds").mkdir()
    for f in SYNTHETIC.iterdir():
        if f.is_file():
            shutil.copy(f, step / f.name)
    for i in image_ids:
        shutil.copy(SYNTHETIC / "seq" / f"{i:06d}.color.jpg", step / "seq" / f"{i:06d}.color.jpg")
    return step


def test_build_image_index_later_dir_wins(tmp_path: Path):
    a = _fake_step(tmp_path, "s0", [0, 1, 2])
    b = _fake_step(tmp_path, "s1", [2, 3])
    idx = build_image_index([a, b, tmp_path / "missing"])
    assert sorted(idx) == [0, 1, 2, 3]
    assert idx[2] == b / "seq" / "000002.color.jpg"


def test_recover_registry_from_g2o(tmp_path: Path):
    T = _pose(12.0, [0.3, -0.1, 0.7])
    t, q = pose_to_vec(T)
    g2o = tmp_path / "initial_pose_graph.g2o"
    g2o.write_text(
        "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
        f"EDGE_SE3:QUAT 4 48 {' '.join(f'{v:.9f}' for v in (*t, *q))} 1 0 0 0 0 0 1 0 0 0 0 1\n"
    )
    registry = {(4, 48): {"T_AB": None, "conf": 1.0, "first_step": 0, "reject_count": 0, "last_weight": 1.0}}
    fixed = recover_registry_from_g2o(registry, g2o)
    assert np.allclose(fixed[(4, 48)]["T_AB"], T, atol=1e-7)
    assert registry[(4, 48)]["T_AB"] is None  # input untouched
    with pytest.raises(ValueError, match=r"\(5, 49\)"):
        recover_registry_from_g2o({(5, 49): dict(registry[(4, 48)])}, g2o)


def test_consolidate_gathers_images_and_registry(tmp_path: Path):
    n = len([l for l in (SYNTHETIC / "poses.txt").read_text().splitlines() if l.strip()])
    ids = list(range(n))
    s0 = _fake_step(tmp_path, "m0", ids[: n // 3])
    s1 = _fake_step(tmp_path, "m1", ids[n // 3: 2 * n // 3])
    s2 = _fake_step(tmp_path, "m2", ids[2 * n // 3:])
    T = _pose(1.0, [0, 0, 1])
    t, q = pose_to_vec(T)
    (s2 / "preds" / "loop_registry.txt").write_text(
        "# a_id,b_id,conf,first_step,reject_count,last_weight\n1,7,2.0,1,0,1.0\n")
    (s2 / "preds" / "initial_pose_graph.g2o").write_text(
        f"EDGE_SE3:QUAT 1 7 {' '.join(f'{v:.9f}' for v in (*t, *q))} 1 0 0 0 0 0 1 0 0 0 0 1\n")

    out = tmp_path / "final"
    meta = consolidate_map(s2, [s0, s1, s2], out, meta={"run_id": "run_x"})
    assert meta["num_images"] == n and meta["registry_edges"] == 1 and meta["run_id"] == "run_x"
    assert len(list((out / "seq").glob("*.color.jpg"))) == n
    assert (out / "poses.txt").read_text() == (s2 / "poses.txt").read_text()
    assert not (out / "preds" / "initial_pose_graph.g2o").exists()  # preds not copied wholesale
    back = read_loop_registry(out / "preds" / "loop_registry.txt")
    assert np.allclose(back[(1, 7)]["T_AB"], T, atol=1e-7)
    assert json.loads((out / "merge_meta.json").read_text())["num_nodes"] == n
    # hard link (same inode) when on the same filesystem
    src = s0 / "seq" / "000000.color.jpg"
    assert (out / "seq" / "000000.color.jpg").stat().st_ino == src.stat().st_ino


def test_consolidate_missing_image(tmp_path: Path):
    s0 = _fake_step(tmp_path, "m0", [0, 1])
    with pytest.raises(FileNotFoundError, match="2"):
        consolidate_map(s0, [s0], tmp_path / "final")


def test_consolidate_rejects_non_consecutive(tmp_path: Path):
    s0 = _fake_step(tmp_path, "m0", range(12))
    lines = (s0 / "poses.txt").read_text().splitlines()
    lines[1] = lines[1].replace("000001", "000009")
    (s0 / "poses.txt").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        consolidate_map(s0, [s0], tmp_path / "final")
