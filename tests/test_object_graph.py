#! /usr/bin/env python
"""T1.2 acceptance (schema v2.0): dual-embedding schema round-trip, BOXER-aligned
IoU/merge, and provider integration.

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
from object_node import OBB, ObjectObservation, SCHEMA_VERSION  # noqa: E402
from object_provider import MockObjectProvider  # noqa: E402
from utils_object_geom import canonicalize_vertical_axis, iou_exact7, weighted_yaw_mean  # noqa: E402


def _obs(label, center, emb_msgnav, emb_boxer=None, conf=0.8, kf_id=0, size=(1.0, 1.0, 1.0), yaw=0.0):
    embs = {"msgnav": np.array(emb_msgnav, float)}
    if emb_boxer is not None:
        embs["boxer"] = np.array(emb_boxer, float)
    return ObjectObservation(
        label=label,
        obb=OBB.from_center_size_yaw(center, size, yaw),
        embeddings=embs,
        confidence=conf,
        provider="mock",
        keyframe_id=kf_id,
    )


def test_iou_exact7_sanity():
    c, s = [0, 0, 0], [2, 2, 2]
    assert iou_exact7(c, s, 0.0, c, s, 0.0) == pytest.approx(1.0, abs=1e-6)
    # translate by half-width along x -> 50% linear overlap in x -> IoU = 1/3
    assert iou_exact7(c, s, 0.0, [1, 0, 0], s, 0.0) == pytest.approx(1.0 / 3.0, abs=1e-3)
    # far apart -> 0
    assert iou_exact7(c, s, 0.0, [9, 0, 0], s, 0.0) == 0.0


def test_weighted_yaw_mean_pi_periodic():
    # yaw and yaw+pi are equivalent for a box -> mean of {0.1, 0.1+pi} ~ 0.1
    m = weighted_yaw_mean([0.1, 0.1 + np.pi], [1.0, 1.0])
    assert abs((m - 0.1 + np.pi / 2) % np.pi - np.pi / 2) < 1e-6


def test_merge_same_object_str_id():
    g = ObjectGraph(Path("/tmp/ogn_og_a"))
    n1, c1 = g.integrate_observation(_obs("chair", [0, 0, 0], [1, 0, 0, 0], kf_id=0), step=0)
    n2, c2 = g.integrate_observation(_obs("chair", [0.1, 0, 0], [1, 0, 0, 0], kf_id=1), step=1)
    assert c1 is True and c2 is False and n1 is n2
    assert g.get_num_node() == 1
    assert isinstance(n1.id, str) and n1.id.startswith("obj_")
    assert n1.num_observations == 2
    assert n1.observed_keyframes == [(0, 0.5), (1, 0.5)]  # default visibility_score


def test_different_objects_not_merged():
    g = ObjectGraph(Path("/tmp/ogn_og_b"))
    g.integrate_observation(_obs("chair", [0, 0, 0], [1, 0, 0, 0]), step=0)   # base
    g.integrate_observation(_obs("table", [9, 0, 0], [1, 0, 0, 0]), step=1)   # far -> IoU 0
    g.integrate_observation(_obs("plant", [0, 0, 0], [0, 1, 0, 0]), step=2)   # emb orthogonal
    assert g.get_num_node() == 3


def test_confidence_prob_weighting():
    g = ObjectGraph(Path("/tmp/ogn_og_c"))
    node, _ = g.integrate_observation(_obs("chair", [0, 0, 0], [1, 0, 0, 0], conf=0.5), step=0)
    g.integrate_observation(_obs("chair", [0, 0, 0], [1, 0, 0, 0], conf=0.9), step=1)
    # BOXER-style weighted mean of confidences (not probabilistic accumulation)
    assert node.confidence == pytest.approx(np.average([0.5, 0.9], weights=[0.5, 0.9]), abs=1e-6)


def test_dual_embedding_roundtrip(tmp_path):
    g = ObjectGraph(tmp_path)
    g.integrate_observation(
        _obs("chair", [0, 0, 0], [1, 0, 0, 0], emb_boxer=[0, 1, 0], kf_id=0), step=0)
    g.integrate_observation(_obs("table", [9, 0, 0], [0, 1, 0, 0], emb_boxer=[1, 0, 0], kf_id=1), step=1)
    g.save_to_file()
    payload = (tmp_path / "objects.json").read_text()
    assert '"schema_version": "%s"' % SCHEMA_VERSION in payload

    loaded = ObjectGraphLoader.load_data(tmp_path)
    assert loaded.get_num_node() == 2
    for nid, original in g.nodes.items():
        r = loaded.get_node(nid)
        assert set(r.embeddings) == {"msgnav", "boxer"}
        np.testing.assert_allclose(r.embeddings["msgnav"], original.embeddings["msgnav"])
        np.testing.assert_allclose(r.embeddings["boxer"], original.embeddings["boxer"])
        np.testing.assert_allclose(r.obb.center, original.obb.center)
        assert r.confidence == pytest.approx(original.confidence)


def test_mock_provider_20_frames():
    scripted = []
    for i in range(20):
        if i < 10:
            scripted.append([_obs("A", [0, 0, 0], [1, 0, 0, 0], kf_id=i)])
        else:
            scripted.append([_obs("B", [5, 0, 0], [0, 1, 0, 0], kf_id=i)])
    provider = MockObjectProvider(scripted)
    g = ObjectGraph(Path("/tmp/ogn_og_d"))
    for step in range(20):
        g.integrate_observations(provider.on_keyframe(kf=step, obs=None), step=step)
    assert g.get_num_node() == 2
    assert sorted(n.num_observations for n in g.nodes.values()) == [10, 10]


def test_canonicalize_vertical_axis_identity_rot():
    # local axis0 = vertical (height 1.2m); rot=I means world-up is also axis2.
    r_box = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    extent = np.array([1.2, 0.3, 0.4])
    r_fixed, extent_fixed = canonicalize_vertical_axis(r_box, extent, np.eye(3), up_axis=2)
    assert extent_fixed[2] == pytest.approx(1.2)
    composed = np.eye(3) @ r_fixed
    np.testing.assert_allclose(composed[:, 2], [0, 0, 1], atol=1e-9)


def test_canonicalize_vertical_axis_world_zup_rotation():
    # habitat y-up -> z-up world rotation (msgnav_provider._WORLD_ZUP).
    world_zup = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    # local axis0 = habitat-y = vertical (height 1.2m); axis1=habitat-z (0.3m); axis2=habitat-x (0.4m)
    r_box = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    extent = np.array([1.2, 0.3, 0.4])
    r_fixed, extent_fixed = canonicalize_vertical_axis(r_box, extent, world_zup, up_axis=2)
    assert extent_fixed[2] == pytest.approx(1.2)
    composed = world_zup @ r_fixed
    np.testing.assert_allclose(composed[:, 2], [0, 0, 1], atol=1e-9)
