import numpy as np
from scipy.spatial.transform import Rotation

from benchmark_map_merge.hloc_sfm_merger import (
    _build_vio_reference_model,
    _build_pnp_per_frame_log,
    _copy_points2d_without_tracks,
    _estimate_se3_umeyama,
    _image_cam_from_world,
    _run_light_bundle_adjustment,
)


def test_estimate_se3_umeyama_recovers_rigid_transform_without_scale() -> None:
    src_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotation = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dst_pts = (rotation @ src_pts.T + translation.reshape(3, 1)).T

    transform, inliers = _estimate_se3_umeyama(src_pts, dst_pts)

    assert transform is not None
    assert len(inliers) == len(src_pts)
    np.testing.assert_allclose(transform[:3, :3], rotation, atol=1e-6)
    np.testing.assert_allclose(transform[:3, 3], translation, atol=1e-6)


def test_build_vio_reference_model_uses_w2c_pose_directly() -> None:
    rotation = Rotation.from_euler("zyx", [15.0, -5.0, 3.0], degrees=True).as_matrix()
    q_xyzw = Rotation.from_matrix(rotation).as_quat()
    vio_pose = np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2], 0.2, -0.3, 1.4],
        dtype=np.float64,
    )

    model = _build_vio_reference_model(
        ["seq/000000.color.jpg", "seq/000001.color.jpg"],
        {"seq/000000.color.jpg": vio_pose},
        (444.0, 445.0, 511.5, 287.5, 1024, 576),
    )

    assert model.num_reg_images() == 1
    image = next(iter(model.images.values()))
    assert image.name == "seq/000000.color.jpg"
    cam_from_world = _image_cam_from_world(image)
    np.testing.assert_allclose(cam_from_world.rotation.matrix(), rotation, atol=1e-6)
    np.testing.assert_allclose(cam_from_world.translation, vio_pose[4:7], atol=1e-6)


def test_light_bundle_adjustment_keeps_rig_pose_constant(monkeypatch) -> None:
    captured = {}

    class FakeOptions:
        def __init__(self) -> None:
            self.refine_focal_length = True
            self.refine_principal_point = True
            self.refine_extra_params = True
            self.refine_points3D = False
            self.refine_rig_from_world = True
            self.refine_sensor_from_rig = True
            self.refine_extrinsics = True
            self.print_summary = True
            self.ceres = type("Ceres", (), {"solver_options": type("Solver", (), {"max_num_iterations": 0})()})()

    def fake_bundle_adjustment(model, options) -> None:
        captured["options"] = options

    monkeypatch.setattr(
        "benchmark_map_merge.hloc_sfm_merger.pycolmap.BundleAdjustmentOptions",
        FakeOptions,
    )
    monkeypatch.setattr(
        "benchmark_map_merge.hloc_sfm_merger.pycolmap.bundle_adjustment",
        fake_bundle_adjustment,
    )

    _run_light_bundle_adjustment(object(), 5)

    options = captured["options"]
    assert options.refine_points3D is True
    assert options.refine_rig_from_world is False
    assert options.refine_sensor_from_rig is False
    assert options.refine_extrinsics is False
    assert options.ceres.solver_options.max_num_iterations == 5


def test_build_pnp_per_frame_log_records_success_and_failure_inliers() -> None:
    pnp_per_frame = _build_pnp_per_frame_log(
        submap_idx=1,
        sfm_sampled_i=["seq/000000.color.jpg", "seq/000001.color.jpg"],
        pnp_results={"seq/000000.color.jpg": np.eye(4)},
        pnp_ref_frames={"seq/000000.color.jpg": "seq/ref.color.jpg"},
        pnp_logs={
            "seq/000000.color.jpg": {
                "num_inliers": 72,
                "points3D_ids": [1, 2, 3],
            },
        },
        failure_samples=[
            {
                "frame": "seq/000001.color.jpg",
                "reason": "insufficient_inliers",
                "num_inliers": 41,
                "num_db": 10,
            }
        ],
    )

    assert pnp_per_frame == [
        {
            "frame": "inc1/seq/000000.color.jpg",
            "best_db": "seq/ref.color.jpg",
            "num_2d3d": 3,
            "num_inliers": 72,
            "status": "SUCCESS",
            "trans_error_m": None,
        },
        {
            "frame": "inc1/seq/000001.color.jpg",
            "num_db": 10,
            "num_inliers": 41,
            "status": "FAIL(insufficient_inliers)",
        },
    ]


def test_copy_points2d_without_tracks_preserves_empty_list() -> None:
    copied_points2d = _copy_points2d_without_tracks([])

    assert copied_points2d == []
