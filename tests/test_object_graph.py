#! /usr/bin/env python
"""T1.2 acceptance: L4 object-graph schema round-trip + merge logic + provider.

Run: pytest third_party/opennavmap/tests/test_object_graph.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "third_party" / "litevloc_code" / "python"))

from object_graph import ObjectGraph, ObjectGraphLoader  # noqa: E402
from object_node import OBB, ObjectObservation  # noqa: E402
from object_provider import MockObjectProvider  # noqa: E402


def _obs(label, center, embedding, conf=0.8, kf_id=0, size=(1.0, 1.0, 1.0), provider="mock"):
    return ObjectObservation(
        label=label,
        obb=OBB(center=np.array(center, float), size=np.array(size, float), R=np.eye(3)),
        embedding=np.array(embedding, float),
        confidence=conf,
        provider=provider,
        keyframe_id=kf_id,
        visibility_score=1.0,
    )


def test_merge_same_object_double_observation():
    graph = ObjectGraph(Path("/tmp/ogn_objgraph_a"))
    _, created1 = graph.integrate_observation(_obs("chair", [0, 0, 0], [1, 0, 0, 0], kf_id=0), step=0)
    node, created2 = graph.integrate_observation(_obs("chair", [0.1, 0, 0], [1, 0, 0, 0], kf_id=1), step=1)
    assert created1 is True and created2 is False
    assert graph.get_num_node() == 1
    assert node.num_observations == 2
    assert node.confidence > 0.8  # accumulated
    assert node.observed_keyframes == [(0, 1.0), (1, 1.0)]


def test_different_objects_not_merged():
    graph = ObjectGraph(Path("/tmp/ogn_objgraph_b"))
    # far apart -> IoU 0
    graph.integrate_observation(_obs("chair", [0, 0, 0], [1, 0, 0, 0]), step=0)
    graph.integrate_observation(_obs("table", [9, 0, 0], [1, 0, 0, 0]), step=1)
    # same place but orthogonal embedding -> similarity below threshold
    graph.integrate_observation(_obs("plant", [0, 0, 0], [0, 1, 0, 0]), step=2)
    assert graph.get_num_node() == 3


def test_confidence_update_rule():
    graph = ObjectGraph(Path("/tmp/ogn_objgraph_c"))
    node, _ = graph.integrate_observation(_obs("chair", [0, 0, 0], [1, 0, 0, 0], conf=0.5), step=0)
    assert node.confidence == pytest.approx(0.5)
    graph.integrate_observation(_obs("chair", [0, 0, 0], [1, 0, 0, 0], conf=0.5), step=1)
    assert node.confidence == pytest.approx(1 - 0.5 * 0.5)  # 0.75


def test_schema_roundtrip(tmp_path):
    graph = ObjectGraph(tmp_path)
    graph.integrate_observation(_obs("chair", [0, 0, 0], [1, 0, 0, 0], kf_id=0), step=0)
    graph.integrate_observation(_obs("table", [9, 0, 0], [0, 1, 0, 0], kf_id=1), step=1)
    graph.save_to_file(edge_only=False)
    assert (tmp_path / "objects.json").is_file()

    loaded = ObjectGraphLoader.load_data(tmp_path)
    assert loaded.get_num_node() == graph.get_num_node()
    for node_id, original in graph.nodes.items():
        restored = loaded.get_node(node_id)
        assert restored.label == original.label
        assert restored.provider == original.provider
        assert restored.confidence == pytest.approx(original.confidence)
        np.testing.assert_allclose(restored.obb.center, original.obb.center)
        np.testing.assert_allclose(restored.obb.R, original.obb.R)
        np.testing.assert_allclose(restored.embedding, original.embedding)
        assert restored.observed_keyframes == original.observed_keyframes


def test_mock_provider_20_frames():
    # frames 0-9 see object A, 10-19 see object B (disjoint -> 2 merged nodes)
    scripted = []
    for i in range(20):
        if i < 10:
            scripted.append([_obs("chairA", [0, 0, 0], [1, 0, 0, 0], kf_id=i)])
        else:
            scripted.append([_obs("chairB", [5, 0, 0], [0, 1, 0, 0], kf_id=i)])
    provider = MockObjectProvider(scripted)

    graph = ObjectGraph(Path("/tmp/ogn_objgraph_d"))
    for step in range(20):
        graph.integrate_observations(provider.on_keyframe(kf=step, obs=None), step=step)

    assert graph.get_num_node() == 2
    counts = sorted(n.num_observations for n in graph.nodes.values())
    assert counts == [10, 10]
