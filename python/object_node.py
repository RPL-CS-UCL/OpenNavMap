#! /usr/bin/env python
"""L4 object-graph node + observation schema (OpenGoalNav T1.2, schema v2.0).

Changes vs v1.0 (frozen after user review):
- ``id`` is now ``str`` (may contain digits, e.g. "obj_3").
- **dual embeddings**: ``embeddings: dict[str, np.ndarray]`` (e.g. "msgnav"=1024-d
  open_clip ViT-H-14 visual, "boxer"=512-d OWLv2 text) — dev-phase keeps both.
- OBB carries a yaw builder; 3D IoU + fusion align with BOXER (see utils_object_geom).

Any field change must bump SCHEMA_VERSION and update the round-trip test.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.base_node import BaseNode  # litevloc read-only base

SCHEMA_VERSION = "2.0"


def _R_about(up_axis: int, yaw: float) -> np.ndarray:
    """3x3 rotation of ``yaw`` radians about ``up_axis``."""
    c, s = np.cos(yaw), np.sin(yaw)
    a0, a1 = [i for i in range(3) if i != up_axis]
    rot = np.eye(3)
    rot[a0, a0], rot[a0, a1] = c, -s
    rot[a1, a0], rot[a1, a1] = s, c
    return rot


@dataclass
class OBB:
    """Oriented bounding box: center, full-extent size, 3x3 rotation (box->world)."""

    center: np.ndarray  # (3,)
    size: np.ndarray  # (3,) full extents
    R: np.ndarray  # (3, 3)

    @staticmethod
    def from_center_size_yaw(center, size, yaw: float, up_axis: int = 2) -> "OBB":
        return OBB(np.asarray(center, float).reshape(3),
                   np.asarray(size, float).reshape(3), _R_about(up_axis, yaw))

    def as_dict(self) -> dict:
        return {
            "center": np.asarray(self.center, float).reshape(3).tolist(),
            "size": np.asarray(self.size, float).reshape(3).tolist(),
            "R": np.asarray(self.R, float).reshape(9).tolist(),
        }

    @staticmethod
    def from_dict(data: dict) -> "OBB":
        return OBB(np.asarray(data["center"], float).reshape(3),
                   np.asarray(data["size"], float).reshape(3),
                   np.asarray(data["R"], float).reshape(3, 3))


@dataclass
class ObjectObservation:
    """One raw object observation from a provider (pre-association)."""

    label: str
    obb: OBB
    embeddings: Dict[str, np.ndarray]  # {"msgnav": (1024,), "boxer": (512,), ...}
    confidence: float
    provider: str
    keyframe_id: int
    visibility_score: float = 1.0
    pointcloud_ref: Optional[str] = None


class ObjectNode(BaseNode):
    """A merged scene object (str id); ``trans`` mirrors the OBB center."""

    def __init__(
        self,
        id: str,
        label: str,
        obb: OBB,
        embeddings: Dict[str, np.ndarray],
        confidence: float,
        last_verified_step: int,
        provider: str,
        pointcloud_ref: Optional[str] = None,
        observed_keyframes: Optional[List[Tuple[int, float]]] = None,
        num_observations: int = 1,
    ) -> None:
        super().__init__(id, trans=np.asarray(obb.center, float).reshape(3))
        self.label = label
        self.obb = obb
        self.embeddings: Dict[str, np.ndarray] = {
            k: np.asarray(v, float).reshape(-1) for k, v in embeddings.items()
        }
        self.confidence = float(confidence)
        self.last_verified_step = int(last_verified_step)
        self.provider = provider
        self.pointcloud_ref = pointcloud_ref
        self.observed_keyframes: List[Tuple[int, float]] = list(observed_keyframes or [])
        self.num_observations = int(num_observations)
        # In-memory fusion history (not serialized; seeded from fused state on load).
        self._centers: List[np.ndarray] = [np.asarray(obb.center, float).reshape(3)]
        self._sizes: List[np.ndarray] = [np.asarray(obb.size, float).reshape(3)]
        self._yaws: List[float] = [0.0]
        self._confidences: List[float] = [float(confidence)]
        self._emb_weight: Dict[str, float] = {k: float(confidence) for k in self.embeddings}

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "label": self.label,
            "obb": self.obb.as_dict(),
            "embeddings": {k: np.asarray(v, float).tolist() for k, v in self.embeddings.items()},
            "confidence": self.confidence,
            "last_verified_step": self.last_verified_step,
            "provider": self.provider,
            "pointcloud_ref": self.pointcloud_ref,
            "observed_keyframes": [[int(k), float(v)] for k, v in self.observed_keyframes],
            "num_observations": self.num_observations,
        }

    @staticmethod
    def from_dict(data: dict) -> "ObjectNode":
        node = ObjectNode(
            id=str(data["id"]),
            label=data["label"],
            obb=OBB.from_dict(data["obb"]),
            embeddings={k: np.asarray(v, float) for k, v in data["embeddings"].items()},
            confidence=float(data["confidence"]),
            last_verified_step=int(data["last_verified_step"]),
            provider=data["provider"],
            pointcloud_ref=data.get("pointcloud_ref"),
            observed_keyframes=[(int(k), float(v)) for k, v in data.get("observed_keyframes", [])],
            num_observations=int(data.get("num_observations", 1)),
        )
        return node

    @staticmethod
    def from_observation(node_id: str, obs: ObjectObservation, step: int) -> "ObjectNode":
        return ObjectNode(
            id=node_id,
            label=obs.label,
            obb=obs.obb,
            embeddings=obs.embeddings,
            confidence=obs.confidence,
            last_verified_step=step,
            provider=obs.provider,
            pointcloud_ref=obs.pointcloud_ref,
            observed_keyframes=[(obs.keyframe_id, obs.visibility_score)],
            num_observations=1,
        )
