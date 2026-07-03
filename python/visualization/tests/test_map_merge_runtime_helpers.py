from types import SimpleNamespace

import numpy as np
import torch

from map_merge_pipeline import _record_graph_edges, _scene_confidence_maps


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
