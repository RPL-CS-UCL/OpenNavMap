from types import SimpleNamespace

import numpy as np
import torch
from matplotlib import rcParams

from map_merge_pipeline import (
    _node_payload,
    _plot_runtime_dmatrix_panels,
    _record_graph_edges,
    _scene_confidence_maps,
)


def test_scene_confidence_maps_uses_current_conf_interface() -> None:
    scene = SimpleNamespace(conf_i={"0_1": torch.tensor([2.0])}, conf_j={"0_1": torch.tensor([3.0])})

    conf_i, conf_j = _scene_confidence_maps(scene)

    assert conf_i["0_1"].item() == 2.0
    assert conf_j["0_1"].item() == 3.0


def test_scene_confidence_maps_keeps_legacy_fallback() -> None:
    scene = SimpleNamespace(weight_i={"0_1": torch.tensor([2.0])}, weight_j={"0_1": torch.tensor([3.0])})

    conf_i, conf_j = _scene_confidence_maps(scene)

    assert conf_i["0_1"].item() == 2.0
    assert conf_j["0_1"].item() == 3.0


class FakeRecorder:
    def __init__(self) -> None:
        self.events = []

    def record_event(self, **kwargs) -> None:
        self.events.append(kwargs)


class FakeNode:
    def __init__(self, node_id: int) -> None:
        self.id = node_id
        self.trans = np.array([float(node_id), 0.0, 0.0])
        self.edges = {}


def test_node_payload_records_camera_intrinsics_and_image_size(tmp_path) -> None:
    node = FakeNode(5)
    node.time = 123.4
    node.quat = np.array([0.0, 0.0, 0.0, 1.0])
    node.rgb_img_name = "seq/000005.color.jpg"
    node.raw_K = np.array([[100.0, 0.0, 50.0], [0.0, 110.0, 60.0], [0.0, 0.0, 1.0]])
    node.K = np.array([[90.0, 0.0, 45.0], [0.0, 95.0, 55.0], [0.0, 0.0, 1.0]])
    node.raw_img_size = np.array([2880, 2880])
    node.img_size = np.array([512, 288])
    graph = SimpleNamespace(map_root=tmp_path)

    payload = _node_payload(graph, node)

    assert payload["raw_K"] == [[100.0, 0.0, 50.0], [0.0, 110.0, 60.0], [0.0, 0.0, 1.0]]
    assert payload["K"] == [[90.0, 0.0, 45.0], [0.0, 95.0, 55.0], [0.0, 0.0, 1.0]]
    assert payload["raw_img_size"] == [2880, 2880]
    assert payload["img_size"] == [512, 288]
    assert payload["rgb_img_path"] == str(tmp_path / "seq/000005.color.jpg")


def test_record_graph_edges_writes_node_ids_weight_and_edge_type() -> None:
    node_a = FakeNode(3)
    node_b = FakeNode(8)
    node_a.edges[node_b.id] = (node_b, 0.75)
    node_b.edges[node_a.id] = (node_a, 0.75)
    graph = SimpleNamespace(nodes={node_a.id: node_a, node_b.id: node_b})
    recorder = FakeRecorder()

    _record_graph_edges(
        recorder=recorder,
        merge_step=1,
        submap_id=4,
        edge_type="covis",
        graph=graph,
    )

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event["event_type"] == "covis_edge_observed"
    assert event["stage"] == "graph_edge_observed"
    assert event["keyframe_id"] == 8
    assert event["payload"]["edge_type"] == "covis"
    assert event["payload"]["nodeAid"] == 3
    assert event["payload"]["nodeBid"] == 8
    assert event["payload"]["weight"] == 0.75


def test_plot_runtime_dmatrix_panels_uses_paper_style_and_palatino(tmp_path) -> None:
    output_path = tmp_path / "dmatrix.png"

    _plot_runtime_dmatrix_panels(
        D_all=np.eye(4),
        panels=[("Difference Matrix", [(1, 2)], (0.0, 152 / 255, 83 / 255))],
        output_path=output_path,
        figsize=(4, 3),
    )

    assert output_path.exists()
    assert rcParams["font.family"] == ["serif"]
    assert rcParams["font.serif"][0] == "Palatino"
