from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rosbag_convert.bag_reader import Intrinsics
from rosbag_convert.map_writer import (Frame, frame_name, write_image, write_orders_file,
                                       write_submap_metadata)
from utils.utils_geom import convert_vec_to_matrix, read_poses

INTR = Intrinsics(700.0, 701.0, 800.0, 648.0, 1600, 1296)


def _pose(x: float, yaw_deg: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    T[:3, 3] = (x, 0.5, 1.0)
    return T


def _frames(n: int = 5):
    # body pose along x; camera = body rotated so optical z looks along body x
    R_bc = Rotation.from_euler("xyz", [-90, 0, -90], degrees=True).as_matrix()
    T_bc = np.eye(4)
    T_bc[:3, :3] = R_bc
    T_bc[:3, 3] = (0.1, 0.0, 0.2)
    frames = []
    for i in range(n):
        T_wb = _pose(2.0 * i, 10.0 * i)
        frames.append(Frame(frame_name(i), 1_700_000_000_000_000_000 + i * 69_000_000,
                            T_wb @ T_bc, T_wb))
    return frames, T_bc


def _read_T(path: Path, name: str) -> np.ndarray:
    row = read_poses(str(path))[name]
    return convert_vec_to_matrix(row[4:], row[:4], mode="wxyz")


def test_metadata_files_round_trip_through_litevloc_readers(tmp_path):
    frames, _ = _frames()
    write_submap_metadata(tmp_path, frames, INTR, anchor_to_first_body=True)
    for fn in ("timestamps.txt", "intrinsics.txt", "poses.txt", "poses_abs_gt.txt", "gps_data.txt",
               "edges_odom.txt", "edges_covis.txt", "edges_trav.txt"):
        assert (tmp_path / fn).exists(), fn
    # poses_abs_gt.txt stores world-to-camera in the global frame
    T_file = _read_T(tmp_path / "poses_abs_gt.txt", frames[2].name)
    np.testing.assert_allclose(T_file, np.linalg.inv(frames[2].T_w_c), atol=1e-5)
    # poses.txt is anchored to the first body pose: camera 0 sits at T_bc, z stays up
    T_local = np.linalg.inv(_read_T(tmp_path / "poses.txt", frames[0].name))
    np.testing.assert_allclose(T_local, np.linalg.inv(frames[0].T_w_b) @ frames[0].T_w_c, atol=1e-5)
    T_local_2 = np.linalg.inv(_read_T(tmp_path / "poses.txt", frames[2].name))
    np.testing.assert_allclose(T_local_2, np.linalg.inv(frames[0].T_w_b) @ frames[2].T_w_c, atol=1e-5)


def test_edges_are_consecutive_chain_with_camera_distances(tmp_path):
    frames, _ = _frames()
    write_submap_metadata(tmp_path, frames, INTR, anchor_to_first_body=False)
    edges = np.loadtxt(tmp_path / "edges_odom.txt", ndmin=2)
    assert edges.shape == (4, 3)
    np.testing.assert_array_equal(edges[:, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(edges[:, 1], [1, 2, 3, 4])
    expected = [np.linalg.norm(frames[i + 1].T_w_c[:3, 3] - frames[i].T_w_c[:3, 3]) for i in range(4)]
    np.testing.assert_allclose(edges[:, 2], expected, atol=1e-5)
    assert (tmp_path / "edges_covis.txt").read_text() == (tmp_path / "edges_odom.txt").read_text()


def test_loaders_accept_written_submap(tmp_path):
    from point_graph import PointGraphLoader
    frames, _ = _frames()
    write_submap_metadata(tmp_path, frames, INTR, anchor_to_first_body=True)
    graph = PointGraphLoader.load_data(tmp_path, edge_type="odom")
    assert graph.get_num_node() == 5
    node = graph.get_node(3)
    T_local_3 = np.linalg.inv(frames[0].T_w_b) @ frames[3].T_w_c
    np.testing.assert_allclose(node.trans, T_local_3[:3, 3], atol=1e-5)
    gps = np.loadtxt(tmp_path / "gps_data.txt", usecols=range(1, 6))
    assert gps.shape == (5, 5) and np.all(np.isnan(gps))


def test_write_image_and_orders(tmp_path):
    img = np.zeros((8, 16, 3), dtype=np.uint8)
    write_image(tmp_path, frame_name(0), img, jpeg_quality=90)
    assert (tmp_path / "seq" / "000000.color.jpg").stat().st_size > 0
    orders = write_orders_file(tmp_path, "s00000", [0, 1, 2])
    assert orders.read_text() == "0 1 2\n"


def test_too_few_frames_rejected(tmp_path):
    frames, _ = _frames(2)
    with pytest.raises(ValueError, match="at least 3"):
        write_submap_metadata(tmp_path, frames, INTR, anchor_to_first_body=True)
