#! /usr/bin/env python
"""L4 object-graph node + observation schema (OpenGoalNav T1.2).

Adds a fourth graph layer on top of litevloc's read-only BaseNode. ``ObjectNode``
is a merged, scene-stable object; ``ObjectObservation`` is a single raw detection
emitted by a provider (see object_provider.py) before cross-frame association.

Schema is frozen at SCHEMA_VERSION; any field change must bump it and update the
round-trip test (CLAUDE.md hard rule 3).
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.base_node import BaseNode  # litevloc read-only base

SCHEMA_VERSION = "1.0"


@dataclass
class OBB:
    """Oriented bounding box: center, full-extent size, and 3x3 rotation."""

    center: np.ndarray  # (3,)
    size: np.ndarray  # (3,) full extents
    R: np.ndarray  # (3, 3) rotation, box->world

    def as_dict(self) -> dict:
        return {
            "center": np.asarray(self.center, float).reshape(3).tolist(),
            "size": np.asarray(self.size, float).reshape(3).tolist(),
            "R": np.asarray(self.R, float).reshape(9).tolist(),
        }

    @staticmethod
    def from_dict(data: dict) -> "OBB":
        return OBB(
            center=np.asarray(data["center"], float).reshape(3),
            size=np.asarray(data["size"], float).reshape(3),
            R=np.asarray(data["R"], float).reshape(3, 3),
        )


@dataclass
class ObjectObservation:
    """One raw object observation from a provider (pre-association)."""

    label: str
    obb: OBB
    embedding: np.ndarray  # (D,) in a fixed CLIP/SigLIP space
    confidence: float
    provider: str
    keyframe_id: int
    visibility_score: float = 1.0
    pointcloud_ref: Optional[str] = None


class ObjectNode(BaseNode):
    """A merged scene object; ``trans`` mirrors the OBB center for graph reuse."""

    def __init__(
        self,
        id: int,
        label: str,
        obb: OBB,
        embedding: np.ndarray,
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
        self.embedding = np.asarray(embedding, float).reshape(-1)
        self.confidence = float(confidence)
        self.last_verified_step = int(last_verified_step)
        self.provider = provider
        self.pointcloud_ref = pointcloud_ref
        # object -> keyframe edges: (keyframe_id, visibility_score)
        self.observed_keyframes: List[Tuple[int, float]] = list(observed_keyframes or [])
        self.num_observations = int(num_observations)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (see objects.json schema)."""
        return {
            "id": int(self.id),
            "label": self.label,
            "obb": self.obb.as_dict(),
            "embedding": np.asarray(self.embedding, float).tolist(),
            "confidence": self.confidence,
            "last_verified_step": self.last_verified_step,
            "provider": self.provider,
            "pointcloud_ref": self.pointcloud_ref,
            "observed_keyframes": [[int(k), float(v)] for k, v in self.observed_keyframes],
            "num_observations": self.num_observations,
        }

    @staticmethod
    def from_dict(data: dict) -> "ObjectNode":
        return ObjectNode(
            id=int(data["id"]),
            label=data["label"],
            obb=OBB.from_dict(data["obb"]),
            embedding=np.asarray(data["embedding"], float),
            confidence=float(data["confidence"]),
            last_verified_step=int(data["last_verified_step"]),
            provider=data["provider"],
            pointcloud_ref=data.get("pointcloud_ref"),
            observed_keyframes=[(int(k), float(v)) for k, v in data.get("observed_keyframes", [])],
            num_observations=int(data.get("num_observations", 1)),
        )

    @staticmethod
    def from_observation(node_id: int, obs: ObjectObservation, step: int) -> "ObjectNode":
        """Create a fresh node from a first observation."""
        return ObjectNode(
            id=node_id,
            label=obs.label,
            obb=obs.obb,
            embedding=obs.embedding,
            confidence=obs.confidence,
            last_verified_step=step,
            provider=obs.provider,
            pointcloud_ref=obs.pointcloud_ref,
            observed_keyframes=[(obs.keyframe_id, obs.visibility_score)],
            num_observations=1,
        )
