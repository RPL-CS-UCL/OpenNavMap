#! /usr/bin/env python
"""Room layer acceptance (schema v2.2): RoomNode/RoomGraph round-trip, str-id
edge-list I/O, and map_manager detection of the rooms.json side-car.

Data here is schema-level (minimal-but-complete dicts), not a synthetic scene:
the real ep0 rooms.json is produced later by the consumer repo's build_rooms.

Run: pytest tests/test_room_graph.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "third_party" / "litevloc_code" / "python"))

from map_manager import load_map  # noqa: E402
from object_node import SCHEMA_VERSION  # noqa: E402
from room_graph import RoomGraph, RoomGraphLoader  # noqa: E402
from room_node import RoomNode  # noqa: E402


def _room(room_id: str, label: str = "bedroom", centroid=(1.0, 2.0, 0.0)) -> RoomNode:
    return RoomNode(
        id=room_id,
        label=label,
        centroid=np.asarray(centroid, float),
        member_keyframes=[0, 1, 2],
        member_objects=["obj_0", "obj_3"],
        embeddings={"clip": np.array([0.6, 0.8])},
        label_scores={"bedroom": 0.7, "living room": 0.2},
        confidence=0.7,
        floor_z=0.02,
        footprint={"resolution": 0.05, "polygons": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]},
        quality={"n_kf": 3, "n_door_edges": 1},
    )


def test_schema_version_is_2_2():
    assert SCHEMA_VERSION == "2.2"
    assert SCHEMA_VERSION.startswith("2.")  # apexnav-side compat assertion


def test_room_node_roundtrip():
    node = _room("room_0")
    data = node.to_dict()
    reloaded = RoomNode.from_dict(data)
    assert reloaded.id == "room_0"
    assert reloaded.label == "bedroom"
    assert reloaded.label_scores == pytest.approx({"bedroom": 0.7, "living room": 0.2})
    assert reloaded.member_keyframes == [0, 1, 2]
    assert reloaded.member_objects == ["obj_0", "obj_3"]
    np.testing.assert_allclose(reloaded.embeddings["clip"], [0.6, 0.8])
    np.testing.assert_allclose(reloaded.centroid, [1.0, 2.0, 0.0])
    assert reloaded.confidence == pytest.approx(0.7)
    assert reloaded.floor_z == pytest.approx(0.02)
    assert reloaded.footprint == node.footprint
    assert reloaded.quality == {"n_kf": 3, "n_door_edges": 1}


def test_room_node_optional_fields_default():
    node = RoomNode(id="room_1", label="unknown", centroid=np.zeros(3))
    reloaded = RoomNode.from_dict(node.to_dict())
    assert reloaded.member_keyframes == [] and reloaded.member_objects == []
    assert reloaded.embeddings == {} and reloaded.label_scores == {}
    assert reloaded.footprint is None and reloaded.quality == {}


def test_room_graph_roundtrip_with_str_id_edges(tmp_path):
    graph = RoomGraph(tmp_path)
    for i, label in enumerate(["bedroom", "kitchen", "hallway"]):
        graph.add_node(_room(f"room_{i}", label, centroid=(float(i), 0.0, 0.0)))
    graph.add_edge_undirected(graph.get_node("room_0"), graph.get_node("room_1"), 3.0)
    graph.add_edge_undirected(graph.get_node("room_1"), graph.get_node("room_2"), 1.0)
    graph.save_to_file()

    payload = (tmp_path / "rooms.json").read_text()
    assert '"schema_version": "%s"' % SCHEMA_VERSION in payload
    edge_text = (tmp_path / "edges_room.txt").read_text().strip().splitlines()
    assert len(edge_text) == 2 and edge_text[0].split()[0].startswith("room_")

    loaded = RoomGraphLoader.load_data(tmp_path)
    assert loaded.get_num_node() == 3
    assert loaded.embedding_dims() == {"clip": 2}
    for rid, original in graph.nodes.items():
        r = loaded.get_node(rid)
        assert r.label == original.label
        np.testing.assert_allclose(r.centroid, original.centroid)
    n1 = loaded.get_node("room_1")
    assert set(n1.edges) == {"room_0", "room_2"}
    assert n1.edges["room_0"][1] == pytest.approx(3.0)


def test_room_graph_empty_edge_file_ok(tmp_path):
    graph = RoomGraph(tmp_path)
    graph.add_node(_room("room_0"))
    graph.save_to_file()
    loaded = RoomGraphLoader.load_data(tmp_path)
    assert loaded.get_num_node() == 1 and not loaded.get_node("room_0").edges


def test_load_map_detects_rooms_json(tmp_path):
    graph = RoomGraph(tmp_path)
    graph.add_node(_room("room_0"))
    graph.add_node(_room("room_1", "kitchen"))
    graph.add_edge_undirected(graph.get_node("room_0"), graph.get_node("room_1"), 2.0)
    graph.save_to_file()

    manager = load_map(tmp_path)
    assert "room" in manager.graphs
    assert manager.room.get_num_node() == 2
    assert manager.room.get_node("room_0").label == "bedroom"


def test_load_map_without_rooms_json_unchanged(tmp_path):
    manager = load_map(tmp_path)  # empty dir: odom/trav attempted, no room key
    assert "room" not in manager.graphs
