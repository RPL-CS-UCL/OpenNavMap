# tests/test_map_files.py
from pathlib import Path

import numpy as np
from conftest import SYNTHETIC_MAP
from scipy.spatial.transform import Rotation


def test_frame_index():
    from navmap_console.readers.map_files import frame_index

    assert frame_index("seq/004869.color.jpg") == 4869
    assert frame_index("frame_00012.jpg") == 12
    assert frame_index("000000.color.jpg") == 0


def test_read_poses_c2w_inverts_w2c(tmp_path: Path):
    from navmap_console.readers.map_files import read_poses_c2w

    rot = Rotation.from_euler("z", 90, degrees=True)  # world-to-camera rotation
    centre = np.array([1.0, 2.0, 3.0])
    t = -rot.apply(centre)  # w2c translation for a camera at `centre`
    q = rot.as_quat()  # xyzw
    (tmp_path / "poses.txt").write_text(
        f"seq/000000.color.jpg {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]}\n"
        "seq/000001.color.jpg 0 0 0 0 0 0 0\n")
    poses = read_poses_c2w(tmp_path / "poses.txt")
    assert poses.names == ["seq/000000.color.jpg", "seq/000001.color.jpg"]
    assert poses.pos.dtype == np.float32 and poses.quat.dtype == np.float32
    np.testing.assert_allclose(poses.pos[0], centre, atol=1e-5)
    np.testing.assert_allclose(np.abs(poses.quat[0]), np.abs(rot.inv().as_quat()), atol=1e-5)
    assert poses.valid.tolist() == [True, False]
    np.testing.assert_array_equal(poses.pos[1], 0)


def test_read_poses_synthetic():
    from navmap_console.readers.map_files import read_poses_c2w

    poses = read_poses_c2w(SYNTHETIC_MAP / "poses.txt")
    assert poses.pos.shape == (12, 3) and poses.quat.shape == (12, 4) and poses.valid.all()


def test_read_edges_shapes(tmp_path: Path):
    from navmap_console.readers.map_files import read_edges

    assert read_edges(SYNTHETIC_MAP / "edges_covis.txt").shape == (30, 3)
    assert read_edges(tmp_path / "missing.txt").shape == (0, 3)
    (tmp_path / "empty.txt").write_text("")
    assert read_edges(tmp_path / "empty.txt").shape == (0, 3)
    (tmp_path / "two.txt").write_text("0 1\n1 2\n")
    two = read_edges(tmp_path / "two.txt")
    assert two.shape == (2, 3) and two[:, 2].tolist() == [1.0, 1.0]
    (tmp_path / "one.txt").write_text("3 4 0.5\n")
    assert read_edges(tmp_path / "one.txt").shape == (1, 3)


def test_read_timestamps_and_node_ids():
    from navmap_console.readers.map_files import read_node_ids, read_timestamps

    ts = read_timestamps(SYNTHETIC_MAP / "timestamps.txt")
    assert len(ts) == 12 and all(isinstance(v, float) for v in ts.values())
    assert read_node_ids(SYNTHETIC_MAP / "intrinsics.txt").tolist() == list(range(12))


def test_read_g2o_vertices(tmp_path: Path):
    from navmap_console.readers.map_files import read_g2o_vertices

    (tmp_path / "g.g2o").write_text(
        "VERTEX_SE3:QUAT 3 1 2 3 0 0 0 1\n"
        "EDGE_SE3:QUAT 0 1 0 0 0 0 0 0 1 " + " ".join(["1"] * 21) + "\n"
        "VERTEX_SE3:QUAT 0 0.5 0 0 0 0 0.7071 0.7071\n")
    verts = read_g2o_vertices(tmp_path / "g.g2o")
    assert set(verts) == {0, 3}
    np.testing.assert_allclose(verts[3], [1, 2, 3, 0, 0, 0, 1])
    assert read_g2o_vertices(tmp_path / "missing.g2o") == {}


def test_connected_components_orders_by_size():
    from navmap_console.readers.map_files import connected_components

    labels = connected_components(5, np.array([[0, 1], [3, 4]]))
    assert labels.tolist() == [0, 0, 2, 1, 1]
    assert connected_components(3, np.zeros((0, 2), dtype=np.int64)).tolist() == [0, 1, 2]
    assert connected_components(0, np.zeros((0, 2))).shape == (0,)
