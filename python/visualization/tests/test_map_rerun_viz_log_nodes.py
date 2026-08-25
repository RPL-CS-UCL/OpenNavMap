"""log_map_nodes 的 images 开关(增量可视化第 4 步):逐帧那条路接管
camera/color / camera/depth 之后,这里必须能只关掉图像那部分,
关键帧本身的位姿/朝向三角形/位置点照常写。"""
from dataclasses import dataclass

import numpy as np

import visualization.map_rerun_viz as map_rerun_viz


class FakeRerun:
    """记下所有 rr.log/set_time_seconds 调用,不真的写 rerun。"""

    def __init__(self) -> None:
        self.logged = []
        self.times = []

    def set_time_seconds(self, timeline, value) -> None:
        self.times.append((timeline, value))

    @staticmethod
    def Transform3D(translation, mat3x3):
        return ("transform3d", translation, mat3x3)

    @staticmethod
    def LineStrips3D(strips, radii=None, colors=None):
        return ("linestrips3d", strips, radii, colors)

    @staticmethod
    def Points3D(positions, colors=None, radii=None):
        return ("points3d", positions, colors, radii)

    @staticmethod
    def Image(data):
        return ("image", data)

    @staticmethod
    def DepthImage(data, meter):
        return ("depthimage", data, meter)

    def log(self, entity_path, archetype) -> None:
        self.logged.append((entity_path, archetype))


@dataclass
class _Node:
    img_size: tuple = (4, 3)
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    trans: tuple = (0.0, 0.0, 0.0)
    K: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    rgb_img_name: str = "rgb.png"
    depth_img_name: str = "depth.png"
    time: float = 1.0


class _Covis:
    def __init__(self, nodes) -> None:
        self._nodes = nodes
        self.map_root = "/fake/root"

    @property
    def nodes(self):
        return list(self._nodes.keys())

    def get_node(self, nid):
        return self._nodes[nid]


def _setup(monkeypatch) -> FakeRerun:
    fake = FakeRerun()
    monkeypatch.setattr(map_rerun_viz, "rr", fake)
    monkeypatch.setattr(map_rerun_viz, "_load_rgb", lambda path: np.zeros((3, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(map_rerun_viz, "_load_depth", lambda path: np.zeros((3, 4), dtype=np.uint16))
    monkeypatch.setattr(map_rerun_viz, "_draw_obb_on_image", lambda rgb, objs, node: rgb)
    return fake


def test_log_map_nodes_images_true_writes_rgb_and_depth(monkeypatch) -> None:
    fake = _setup(monkeypatch)
    covis = _Covis({0: _Node()})

    map_rerun_viz.log_map_nodes(covis)

    paths = [entity for entity, _ in fake.logged]
    assert "camera/color" in paths
    assert "camera/depth" in paths


def test_log_map_nodes_images_false_skips_rgb_and_depth(monkeypatch) -> None:
    fake = _setup(monkeypatch)
    covis = _Covis({0: _Node()})

    map_rerun_viz.log_map_nodes(covis, images=False)

    paths = [entity for entity, _ in fake.logged]
    assert "camera/color" not in paths
    assert "camera/depth" not in paths
    # 关键帧本身的位姿/朝向三角形/位置点不受这个开关影响,照常写
    assert "map/nodes/0" in paths
    assert "map/nodes/0/heading" in paths
    assert "map/nodes/0/point" in paths
