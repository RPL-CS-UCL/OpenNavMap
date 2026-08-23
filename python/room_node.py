#! /usr/bin/env python
"""Room-layer graph node (schema v2.2).

A RoomNode is one detected room of the scene: a semantic grouping of covis
keyframes (and, through them, object nodes). It lives in ``rooms.json``, a
side-car file next to ``objects.json`` -- maps without it stay valid v2.x maps.

Changes in v2.2:
- new side-car ``rooms.json`` + ``edges_room.txt`` (room adjacency, str ids);
- ``objects.json`` objects may carry an optional ``room_id`` (absent = unknown).

Any field change must bump SCHEMA_VERSION (object_node.py) and update the
round-trip test (tests/test_room_graph.py).
"""
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.base_node import BaseNode  # litevloc read-only base


class RoomNode(BaseNode):
    """A detected room (str id, e.g. "room_0"); ``trans`` mirrors the centroid.

    Fields:
        label: room type (e.g. "bedroom"), "unknown" when classification abstains.
        label_scores: room-type -> probability from the classifier vote.
        member_keyframes: covis keyframe ids belonging to this room.
        member_objects: object node ids (e.g. "obj_3") assigned to this room.
        embeddings: {"clip": (D,)} aggregated room embedding(s).
        centroid: member-keyframe position mean, world frame (ros_zup).
        floor_z: estimated floor height of the room (m, world frame).
        footprint: {"resolution": m/cell, "polygons": [[[x, y], ...], ...]} --
            summary outline for viz only, NOT a navigation map.
        confidence: labeler confidence in ``label`` (0 when "unknown").
        quality: free-form diagnostics (e.g. {"n_kf": 12, "n_door_edges": 1}).
    """

    def __init__(
        self,
        id: str,
        label: str,
        centroid: np.ndarray,
        member_keyframes: Optional[List[int]] = None,
        member_objects: Optional[List[str]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        label_scores: Optional[Dict[str, float]] = None,
        confidence: float = 0.0,
        floor_z: float = 0.0,
        footprint: Optional[dict] = None,
        quality: Optional[dict] = None,
    ) -> None:
        super().__init__(id, trans=np.asarray(centroid, float).reshape(3))
        self.label = label
        self.label_scores: Dict[str, float] = {
            k: float(v) for k, v in (label_scores or {}).items()
        }
        self.member_keyframes: List[int] = [int(k) for k in (member_keyframes or [])]
        self.member_objects: List[str] = [str(o) for o in (member_objects or [])]
        self.embeddings: Dict[str, np.ndarray] = {
            k: np.asarray(v, float).reshape(-1) for k, v in (embeddings or {}).items()
        }
        self.confidence = float(confidence)
        self.floor_z = float(floor_z)
        self.footprint = footprint
        self.quality = dict(quality or {})

    @property
    def centroid(self) -> np.ndarray:
        return np.asarray(self.trans, float).reshape(3)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "label": self.label,
            "label_scores": {k: float(v) for k, v in self.label_scores.items()},
            "member_keyframes": [int(k) for k in self.member_keyframes],
            "member_objects": [str(o) for o in self.member_objects],
            "embeddings": {k: np.asarray(v, float).tolist() for k, v in self.embeddings.items()},
            "confidence": self.confidence,
            "centroid": np.asarray(self.centroid, float).reshape(3).tolist(),
            "floor_z": self.floor_z,
            "footprint": self.footprint,
            "quality": self.quality,
        }

    @staticmethod
    def from_dict(data: dict) -> "RoomNode":
        return RoomNode(
            id=str(data["id"]),
            label=data["label"],
            centroid=np.asarray(data["centroid"], float).reshape(3),
            member_keyframes=[int(k) for k in data.get("member_keyframes", [])],
            member_objects=[str(o) for o in data.get("member_objects", [])],
            embeddings={k: np.asarray(v, float) for k, v in data.get("embeddings", {}).items()},
            label_scores={k: float(v) for k, v in data.get("label_scores", {}).items()},
            confidence=float(data.get("confidence", 0.0)),
            floor_z=float(data.get("floor_z", 0.0)),
            footprint=data.get("footprint"),
            quality=data.get("quality") or {},
        )
