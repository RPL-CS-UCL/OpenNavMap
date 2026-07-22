#! /usr/bin/env python
"""L4 object graph: registration, cross-frame association/merge, serialization.

Lives in the main fork (not litevloc). ``ObjectGraph`` extends litevloc's
BaseGraph and owns the "provider-agnostic" association promise: merging is done
here (3D IoU + embedding double threshold), so any ObjectProvider yields the same
merge behavior (CLAUDE.md T1.2 contract).
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.base_graph import BaseGraph  # litevloc read-only base

from object_node import OBB, ObjectNode, ObjectObservation, SCHEMA_VERSION
from utils_object_geom import cosine_similarity, iou_3d, normalize

DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_EMBEDDING_THRESHOLD = 0.7


class ObjectGraph(BaseGraph):
    """Graph of merged scene objects with object->keyframe visibility edges."""

    def __init__(self, map_root: Path, edge_type: str = "object") -> None:
        super().__init__(map_root, edge_type)

    def integrate_observation(
        self,
        obs: ObjectObservation,
        step: int,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
    ) -> Tuple[ObjectNode, bool]:
        """Associate one observation into the graph (merge) or add a new node.

        A candidate node matches only when BOTH thresholds hold (3D IoU and
        embedding cosine similarity); the best-scoring match is merged into.

        Returns:
            ``(node, created)`` where ``created`` is True iff a new node was added.
        """
        best_node, best_score = None, -1.0
        for node in self.nodes.values():
            iou = iou_3d(obs.obb, node.obb)
            sim = cosine_similarity(obs.embedding, node.embedding)
            if iou >= iou_threshold and sim >= embedding_threshold:
                score = iou + sim
                if score > best_score:
                    best_node, best_score = node, score

        if best_node is None:
            new_id = self.get_max_node_id() + 1 if self.get_num_node() > 0 else 0
            node = ObjectNode.from_observation(new_id, obs, step)
            self.add_node(node)
            return node, True

        self._merge_into(best_node, obs, step)
        return best_node, False

    def integrate_observations(self, observations: List[ObjectObservation], step: int) -> None:
        """Integrate a batch of observations from one keyframe."""
        for obs in observations:
            self.integrate_observation(obs, step)

    @staticmethod
    def _merge_into(node: ObjectNode, obs: ObjectObservation, step: int) -> None:
        """Fold an observation into an existing node (running stats + confidence)."""
        node.num_observations += 1
        weight = 1.0 / node.num_observations
        # Confidence: probabilistic accumulation — more sightings, more certain.
        node.confidence = 1.0 - (1.0 - node.confidence) * (1.0 - obs.confidence)
        # Embedding: running mean, renormalized.
        node.embedding = normalize((1.0 - weight) * node.embedding + weight * normalize(obs.embedding))
        # OBB: running-mean center/size, keep the latest orientation.
        node.obb = OBB(
            center=(1.0 - weight) * node.obb.center + weight * np.asarray(obs.obb.center, float).reshape(3),
            size=(1.0 - weight) * node.obb.size + weight * np.asarray(obs.obb.size, float).reshape(3),
            R=obs.obb.R,
        )
        node.trans = np.asarray(node.obb.center, float).reshape(3)
        node.last_verified_step = step
        node.observed_keyframes.append((obs.keyframe_id, obs.visibility_score))
        if obs.pointcloud_ref:
            node.pointcloud_ref = obs.pointcloud_ref

    def embedding_dim(self) -> int:
        for node in self.nodes.values():
            return int(np.asarray(node.embedding).reshape(-1).shape[0])
        return 0

    def save_to_file(self, edge_only: bool = False) -> None:
        """Write objects.json (schema_version + objects) and edges_object.txt."""
        if not edge_only:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "edge_type": self.edge_type,
                "embedding_dim": self.embedding_dim(),
                "objects": [node.to_dict() for node in self.nodes.values()],
            }
            with open(self.map_root / "objects.json", "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        self.write_edge_list(self.map_root / f"edges_{self.edge_type}.txt")


class ObjectGraphLoader:
    """Loads an ObjectGraph from objects.json + edges_object.txt."""

    @staticmethod
    def load_data(map_root: Path, edge_type: str = "object") -> ObjectGraph:
        graph = ObjectGraph(map_root, edge_type)
        objects_path = map_root / "objects.json"
        if objects_path.exists():
            payload = json.loads(objects_path.read_text(encoding="utf-8"))
            for object_dict in payload.get("objects", []):
                graph.add_node(ObjectNode.from_dict(object_dict))
        graph.read_edge_list(map_root / f"edges_{edge_type}.txt")
        return graph
