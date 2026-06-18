import numpy as np
from scipy.spatial.transform import Rotation

from benchmark_map_merge.hloc_sfm_merger import (
    _c2w_to_rigid3d,
    _extract_w2c_vec_from_image,
    sample_frames_by_vio_distance,
)
from benchmark_map_merge.vis_utils import _vec_to_cam_pos


def _w2c_vec_from_camera_pose(camera_position: np.ndarray, yaw_degrees: float) -> np.ndarray:
    """Build W2C vec7 from a C2W camera center and yaw.

    C2W maps camera coordinates to world coordinates. W2C maps world points into
    the camera. TUM stores camera position/C2W rotation, while this dataset's
    poses.txt stores W2C [qw,qx,qy,qz,tx,ty,tz].
    """
    rotation_c2w = Rotation.from_euler("z", yaw_degrees, degrees=True).as_matrix()
    rotation_w2c = rotation_c2w.T
    translation_w2c = -rotation_w2c @ camera_position
    quat_xyzw = Rotation.from_matrix(rotation_w2c).as_quat()
    return np.array(
        [
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
            translation_w2c[0],
            translation_w2c[1],
            translation_w2c[2],
        ],
        dtype=np.float64,
    )


def test_vec_to_cam_pos_recovers_w2c_camera_center() -> None:
    camera_position = np.array([1.0, 2.0, 3.0])
    pose_vec = _w2c_vec_from_camera_pose(camera_position, yaw_degrees=90.0)

    np.testing.assert_allclose(_vec_to_cam_pos(pose_vec), camera_position)
    assert not np.allclose(pose_vec[4:7], camera_position)


def test_vio_distance_sampling_uses_camera_center_not_w2c_translation() -> None:
    image_list = ["seq/000000.color.jpg", "seq/000001.color.jpg", "seq/000002.color.jpg"]
    vio_poses = {
        image_list[0]: _w2c_vec_from_camera_pose(np.array([1.0, 0.0, 0.0]), 0.0),
        image_list[1]: _w2c_vec_from_camera_pose(np.array([1.0, 0.0, 0.0]), 180.0),
        image_list[2]: _w2c_vec_from_camera_pose(np.array([1.2, 0.0, 0.0]), 180.0),
    }

    selected = sample_frames_by_vio_distance(vio_poses, image_list, min_dist=0.5)

    assert selected == [image_list[0], image_list[2]]


def test_c2w_to_rigid3d_sets_pycolmap_cam_from_world() -> None:
    camera_position = np.array([1.0, 2.0, 3.0])
    rotation_c2w = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    transform_c2w = np.eye(4)
    transform_c2w[:3, :3] = rotation_c2w
    transform_c2w[:3, 3] = camera_position

    rigid = _c2w_to_rigid3d(transform_c2w)

    np.testing.assert_allclose(rigid.rotation.matrix(), rotation_c2w.T, atol=1e-8)
    np.testing.assert_allclose(rigid.translation, -rotation_c2w.T @ camera_position, atol=1e-8)


def test_extract_w2c_vec_from_image_returns_dataset_format() -> None:
    camera_position = np.array([1.0, 2.0, 3.0])
    rotation_c2w = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    transform_c2w = np.eye(4)
    transform_c2w[:3, :3] = rotation_c2w
    transform_c2w[:3, 3] = camera_position

    class _Image:
        cam_from_world = _c2w_to_rigid3d(transform_c2w)

    vec = _extract_w2c_vec_from_image(_Image())
    expected = _w2c_vec_from_camera_pose(camera_position, yaw_degrees=90.0)

    np.testing.assert_allclose(vec, expected, atol=1e-8)
