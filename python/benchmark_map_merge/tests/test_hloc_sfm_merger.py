import numpy as np
from scipy.spatial.transform import Rotation

from benchmark_map_merge.hloc_sfm_merger import (
    _build_vio_reference_model,
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
    assert options.ceres.solver_options.max_num_iterations == 5
