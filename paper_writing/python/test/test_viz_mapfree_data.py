#!/usr/bin/env python

import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from paper_writing.python import viz_mapfree_data


def test_load_poses_parses_query_and_refs(tmp_path: Path) -> None:
    poses_file = tmp_path / "poses.txt"
    poses_file.write_text(textwrap.dedent("""\
        #seq0/frame_00000.jpg 0.9525 0.0620 -0.2925 -0.0564 0.6753 0.0501 0.0500
        seq0/frame_00000.jpg 1.0 0.0 0.0 0.0 0.0 0.0 0.0
        seq1/frame_00000.jpg 0.9682 0.0956 0.2145 0.0851 -0.9034 -0.1792 0.3971
        seq1/frame_00001.jpg 0.9683 0.0988 0.2128 0.0849 -0.8941 -0.1803 0.3898
    """), encoding="utf-8")

    poses = viz_mapfree_data.load_mapfree_poses(poses_file)

    assert "seq0/frame_00000.jpg" in poses
    assert "seq1/frame_00000.jpg" in poses
    assert "seq1/frame_00001.jpg" in poses
    assert sum(1 for k in poses if k.startswith("#")) == 0
    T_c2w = poses["seq0/frame_00000.jpg"]
    np.testing.assert_allclose(T_c2w[:3, 3], [0.0, 0.0, 0.0], atol=1e-6)
    assert T_c2w.shape == (4, 4)


def test_load_poses_w2c_inverted_correctly(tmp_path: Path) -> None:
    poses_file = tmp_path / "poses.txt"
    # qw=1 (identity rotation), tx=1 ty=0 tz=0 → T_c2w translation = [-1,0,0]
    poses_file.write_text(
        "seq1/frame_00000.jpg 1.0 0.0 0.0 0.0 1.0 0.0 0.0\n",
        encoding="utf-8",
    )
    poses = viz_mapfree_data.load_mapfree_poses(poses_file)
    T_c2w = poses["seq1/frame_00000.jpg"]
    np.testing.assert_allclose(T_c2w[:3, 3], [-1.0, 0.0, 0.0], atol=1e-6)


def test_sample_ref_images_returns_exactly_n(tmp_path: Path) -> None:
    seq1 = tmp_path / "seq1"
    seq1.mkdir()
    for i in range(10):
        (seq1 / f"frame_{i:05d}.jpg").write_bytes(b"")

    selected = viz_mapfree_data.sample_ref_images(seq1, n=4, seed=42)

    assert len(selected) == 4
    assert all(p.suffix == ".jpg" for p in selected)
    # deterministic with same seed
    selected2 = viz_mapfree_data.sample_ref_images(seq1, n=4, seed=42)
    assert selected == selected2


def test_sample_ref_images_fewer_than_n(tmp_path: Path) -> None:
    seq1 = tmp_path / "seq1"
    seq1.mkdir()
    for i in range(2):
        (seq1 / f"frame_{i:05d}.jpg").write_bytes(b"")

    selected = viz_mapfree_data.sample_ref_images(seq1, n=4, seed=0)

    assert len(selected) == 2


def test_visualize_scene_writes_png(tmp_path: Path) -> None:
    scene_dir = tmp_path / "s00000"
    (scene_dir / "seq0").mkdir(parents=True)
    (scene_dir / "seq1").mkdir(parents=True)

    dummy = PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    dummy.save(scene_dir / "seq0" / "frame_00000.jpg")
    for i in range(4):
        dummy.save(scene_dir / "seq1" / f"frame_{i:05d}.jpg")

    lines = ["seq0/frame_00000.jpg 1.0 0.0 0.0 0.0 0.0 0.0 0.0\n"]
    for i in range(4):
        lines.append(f"seq1/frame_{i:05d}.jpg 1.0 0.0 0.0 0.0 {float(i)} 0.0 0.0\n")
    (scene_dir / "poses.txt").write_text("".join(lines), encoding="utf-8")

    out_path = tmp_path / "output" / "s00000.png"

    viz_mapfree_data.visualize_scene(
        scene_dir=scene_dir,
        output_path=out_path,
        n_refs=4,
        seed=0,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
