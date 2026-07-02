"""Tests for apply_ba_to_prebuilt_sfm."""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from benchmark_map_merge.scripts.apply_ba_to_prebuilt_sfm import (
    _configure_ba_options,
    apply_ba_single,
    apply_ba_to_prebuilt_sfm,
)


def _make_synthetic_model(pose_noises):
    """Build a small COLMAP Reconstruction with noisy poses and correct 2D observations.

    Args:
        pose_noises: per-camera x-axis offset added to the true pose.
    """
    model = pycolmap.Reconstruction()
    fx, fy, cx, cy = 100.0, 100.0, 400.0, 300.0
    cam = pycolmap.Camera(
        model="PINHOLE", width=800, height=600,
        params=[fx, fy, cx, cy],
    )
    cam.camera_id = 1
    model.add_camera_with_trivial_rig(cam)

    pts3d = np.array(
        [[0, 0, 5], [1, 0, 5], [0, 1, 5], [1, 1, 5],
         [0.5, 0.5, 4], [2, 0, 6], [-1, 0, 4]],
        dtype=np.float64,
    )
    cam_positions = [0.0, 0.5, 1.0]

    for i, cam_x in enumerate(cam_positions):
        img = pycolmap.Image()
        img.image_id = i + 1
        img.name = f"img{i}"
        img.camera_id = 1
        noisy_x = cam_x + pose_noises[i]
        T = np.eye(4)
        T[0, 3] = -noisy_x
        cfw = pycolmap.Rigid3d(
            np.hstack([T[:3, :3], T[:3, 3:].reshape(3, 1)])
        )
        pts2d = []
        for p in pts3d:
            p_cam = p.copy()
            p_cam[0] -= cam_x
            x2 = fx * p_cam[0] / p_cam[2] + cx
            y2 = fy * p_cam[1] / p_cam[2] + cy
            p2d = pycolmap.Point2D()
            p2d.xy = np.array([x2, y2], dtype=np.float64)
            pts2d.append(p2d)
        img.points2D = pts2d
        model.add_image_with_trivial_frame(img, cfw)

    for j, p in enumerate(pts3d):
        pid = model.add_point3D(p.reshape(3, 1), pycolmap.Track())
        for img_id in [1, 2, 3]:
            te = pycolmap.TrackElement()
            te.image_id = img_id
            te.point2D_idx = j
            model.add_observation(pid, te)

    return model


def _write_model(model, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.write_binary(str(out_dir))
    return out_dir


def _pose_translations(model) -> dict:
    result = {}
    for iid, img in model.images.items():
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        result[iid] = cfw.translation.copy()
    return result


# ── unit tests ──────────────────────────────────────────────────────────────


def test_configure_ba_options_refine_extrinsics_true():
    opts = _configure_ba_options(ba_iter=10, refine_extrinsics=True)
    assert opts.refine_rig_from_world is True
    assert opts.refine_sensor_from_rig is False
    assert opts.refine_points3D is True
    assert opts.refine_focal_length is False
    assert opts.refine_principal_point is False
    assert opts.refine_extra_params is False
    assert opts.ceres.solver_options.max_num_iterations == 10


def test_configure_ba_options_refine_extrinsics_false():
    opts = _configure_ba_options(ba_iter=20, refine_extrinsics=False)
    assert opts.refine_rig_from_world is False
    assert opts.refine_points3D is True
    assert opts.ceres.solver_options.max_num_iterations == 20


# ── integration tests ───────────────────────────────────────────────────────


def test_apply_ba_single_refines_poses_when_enabled():
    pose_noises = [0.15, -0.1, 0.2]
    model = _make_synthetic_model(pose_noises)
    pre_poses = _pose_translations(model)

    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in" / "sfm"
        out_dir = Path(tmp) / "out" / "sfm"
        _write_model(model, in_dir)

        stats = apply_ba_single(
            str(in_dir), str(out_dir), ba_iter=50,
            refine_extrinsics=True, submap_id="0",
        )

        assert stats["submap_id"] == "0"
        assert stats["ba_iter"] == 50
        assert stats["refine_extrinsics"] is True
        assert stats["after"]["num_images"] == stats["before"]["num_images"]

        reloaded = pycolmap.Reconstruction()
        reloaded.read_binary(str(out_dir))
    post_poses = _pose_translations(reloaded)

    max_diff = max(
        np.linalg.norm(post_poses[iid] - pre_poses[iid])
        for iid in pre_poses
    )
    assert max_diff > 1e-4, "At least one pose should change with refine_extrinsics=True"


def test_apply_ba_single_keeps_poses_fixed_when_disabled():
    pose_noises = [0.15, -0.1, 0.2]
    model = _make_synthetic_model(pose_noises)
    pre_poses = _pose_translations(model)

    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in" / "sfm"
        out_dir = Path(tmp) / "out" / "sfm"
        _write_model(model, in_dir)

        apply_ba_single(
            str(in_dir), str(out_dir), ba_iter=50,
            refine_extrinsics=False, submap_id="0",
        )

        reloaded = pycolmap.Reconstruction()
        reloaded.read_binary(str(out_dir))
    post_poses = _pose_translations(reloaded)

    for iid in pre_poses:
        diff = np.linalg.norm(post_poses[iid] - pre_poses[iid])
        assert diff < 1e-8, f"Pose {iid} should not change with refine_extrinsics=False"


def test_apply_ba_single_writes_valid_bin():
    model = _make_synthetic_model([0.0, 0.0, 0.0])

    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in" / "sfm"
        out_dir = Path(tmp) / "out" / "sfm"
        _write_model(model, in_dir)

        apply_ba_single(
            str(in_dir), str(out_dir), ba_iter=5,
            refine_extrinsics=True, submap_id="test",
        )

        assert (out_dir / "cameras.bin").is_file()
        assert (out_dir / "images.bin").is_file()
        assert (out_dir / "points3D.bin").is_file()

        reloaded = pycolmap.Reconstruction()
        reloaded.read_binary(str(out_dir))
        assert len(reloaded.images) == 3
        assert len(reloaded.points3D) == 7


def test_apply_ba_to_prebuilt_sfm_end_to_end():
    model = _make_synthetic_model([0.1, -0.05, 0.15])

    with tempfile.TemporaryDirectory() as tmp:
        in_root = Path(tmp) / "input"
        out_root = Path(tmp) / "output"
        sub_dir = in_root / "submaps_sfm" / "0"
        _write_model(model, sub_dir / "sfm")

        results = apply_ba_to_prebuilt_sfm(
            input_root=in_root,
            output_root=out_root,
            ba_iter=10,
            refine_extrinsics=True,
            jobs=1,
        )

        assert len(results) == 1
        assert results[0]["submap_id"] == "0"

        out_sfm = out_root / "submaps_sfm" / "0" / "sfm"
        assert (out_sfm / "cameras.bin").is_file()
        assert (out_sfm / "images.bin").is_file()
        assert (out_sfm / "points3D.bin").is_file()
        assert (out_root / "submaps_sfm" / "0" / "sfm_summary.json").is_file()
        assert (out_root / "logs" / "ba_applied.log").is_file()
