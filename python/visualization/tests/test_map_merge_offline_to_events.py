from pathlib import Path
from visualization.map_merge_offline_to_events import detect_merge_dirs


def test_detect_merge_dirs_finds_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "merge_0").mkdir()
    (tmp_path / "merge_0_1_2").mkdir()
    (tmp_path / "merge_0_1").mkdir()
    (tmp_path / "merge_finalmap").write_text("link")  # file, not dir
    (tmp_path / "not_merge").mkdir()

    result = detect_merge_dirs(tmp_path)
    names = [p.name for p in result]
    assert names == ["merge_0", "merge_0_1", "merge_0_1_2"]


def test_detect_merge_dirs_empty_when_no_merge_dirs(tmp_path: Path) -> None:
    (tmp_path / "other").mkdir()
    assert detect_merge_dirs(tmp_path) == []


import numpy as np
from visualization.map_merge_offline_to_events import (
    PoseEntry,
    EdgeEntry,
    IntrinsicsEntry,
    load_poses,
    load_edges,
    load_intrinsics,
    load_descriptors,
)


def test_load_poses_parses_quat_xyzw_and_translation(tmp_path: Path) -> None:
    poses_file = tmp_path / "poses.txt"
    poses_file.write_text(
        "seq/000000.color.jpg 0.0 0.0 0.0 1.0 1.0 2.0 3.0\n"
        "seq/000001.color.jpg 0.1 0.2 0.3 0.4 4.0 5.0 6.0\n"
    )
    result = load_poses(poses_file)
    assert len(result) == 2
    assert result[0].img_name == "seq/000000.color.jpg"
    assert result[0].quat_xyzw == [0.0, 0.0, 0.0, 1.0]
    assert result[0].position == [1.0, 2.0, 3.0]
    assert result[1].img_name == "seq/000001.color.jpg"
    assert result[1].quat_xyzw == [0.1, 0.2, 0.3, 0.4]
    assert result[1].position == [4.0, 5.0, 6.0]


def test_load_edges_parses_src_dst_weight(tmp_path: Path) -> None:
    edges_file = tmp_path / "edges_covis.txt"
    edges_file.write_text("0 1 0.310245\n1 2 0.402091\n")
    result = load_edges(edges_file)
    assert len(result) == 2
    assert result[0].src == 0
    assert result[0].dst == 1
    assert abs(result[0].weight - 0.310245) < 1e-6


def test_load_intrinsics_parses_fx_fy_cx_cy_w_h(tmp_path: Path) -> None:
    intr_file = tmp_path / "intrinsics.txt"
    intr_file.write_text(
        "seq/000000.color.jpg 444.49 444.49 511.5 287.5 1024 576\n"
    )
    result = load_intrinsics(intr_file)
    assert "seq/000000.color.jpg" in result
    entry = result["seq/000000.color.jpg"]
    assert entry.K[0][0] == 444.49
    assert entry.K[1][2] == 287.5
    assert entry.img_size == [1024, 576]


def test_load_descriptors_parses_256_dim(tmp_path: Path) -> None:
    desc_file = tmp_path / "database_descriptors.txt"
    vals = " ".join(str(float(i)) for i in range(256))
    desc_file.write_text(f"seq/000000.color.jpg {vals}\n")
    result = load_descriptors(desc_file)
    assert "seq/000000.color.jpg" in result
    assert result["seq/000000.color.jpg"].shape == (256,)
    assert result["seq/000000.color.jpg"][0] == 0.0
    assert result["seq/000000.color.jpg"][255] == 255.0
