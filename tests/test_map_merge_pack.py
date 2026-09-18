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
