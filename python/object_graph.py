#! /usr/bin/env python
"""L4 object graph: registration, cross-frame association/merge, serialization.

Association + fusion are done here (provider-agnostic promise). Behavior is
aligned with BOXER (ports in utils_object_geom): 3D IoU via ``iou_exact7`` +
per-embedding cosine double threshold for association; robust confidence-weighted
OBB fusion with pi-periodic yaw circular mean and 90-degree size alignment.
Dual embeddings ("msgnav" visual 1024-d, "boxer" text 512-d) are each fused by a
confidence-weighted running mean.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.base_graph import BaseGraph  # litevloc read-only base

from object_node import OBB, ObjectNode, ObjectObservation, SCHEMA_VERSION
from utils_object_geom import (
    align_boxes_r90, cosine_similarity, iou_3d, normalize, robust_weights,
    weighted_yaw_mean, yaw_from_R,
)

DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_EMBEDDING_THRESHOLD = 0.7
_ID_RE = re.compile(r"obj_(\d+)")


class ObjectGraph(BaseGraph):
    """Graph of merged scene objects with object->keyframe visibility edges."""

    def __init__(self, map_root: Path, edge_type: str = "object", up_axis: int = 2) -> None:
        super().__init__(map_root, edge_type)
        self.up_axis = up_axis

    def _new_id(self) -> str:
        nums = [int(m.group(1)) for m in (_ID_RE.fullmatch(str(i)) for i in self.nodes) if m]
        return f"obj_{(max(nums) + 1) if nums else 0}"

    def _embedding_similarity(self, obs: ObjectObservation, node: ObjectNode) -> float:
        shared = set(obs.embeddings) & set(node.embeddings)
        if not shared:
            return 1.0  # no shared embedding space -> fall back to IoU-only gate
        return max(cosine_similarity(obs.embeddings[k], node.embeddings[k]) for k in shared)

    def integrate_observation(
        self,
        obs: ObjectObservation,
        step: int,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
    ) -> Tuple[ObjectNode, bool]:
        """Associate one observation (merge) or add a new node.

        Match requires BOTH ``iou_3d >= iou_threshold`` (yaw-only exact IoU) and a
        shared-embedding cosine ``>= embedding_threshold``; best score is merged.
        """
        obs_yaw = yaw_from_R(obs.obb.R, self.up_axis)
        best_node, best_score = None, -1.0
        for node in self.nodes.values():
            iou = iou_3d(obs.obb, node.obb, self.up_axis)
            if iou < iou_threshold:
                continue
            sim = self._embedding_similarity(obs, node)
            if sim < embedding_threshold:
                continue
            if iou + sim > best_score:
                best_node, best_score = node, iou + sim

        if best_node is None:
            node = ObjectNode.from_observation(self._new_id(), obs, step)
            node._yaws = [obs_yaw]
            self.add_node(node)
            return node, True

        self._merge_into(best_node, obs, obs_yaw, step)
        return best_node, False

    def integrate_observations(self, observations: List[ObjectObservation], step: int) -> None:
        for obs in observations:
            self.integrate_observation(obs, step)

    def _merge_into(self, node: ObjectNode, obs: ObjectObservation, obs_yaw: float, step: int) -> None:
        node.num_observations += 1
        node._centers.append(np.asarray(obs.obb.center, float).reshape(3))
        node._sizes.append(np.asarray(obs.obb.size, float).reshape(3))
        node._yaws.append(obs_yaw)
        node._confidences.append(float(obs.confidence))

        # Robust confidence-weighted OBB fusion (BOXER-style).
        weights = robust_weights(node._confidences, node._centers)
        center = np.average(np.array(node._centers), axis=0, weights=weights)
        sizes_aligned, yaws_aligned = align_boxes_r90(
            np.array(node._sizes), np.array(node._yaws), weights, self.up_axis)
        size = np.average(sizes_aligned, axis=0, weights=weights)
        yaw = weighted_yaw_mean(yaws_aligned, weights)
        node.obb = OBB.from_center_size_yaw(center, size, yaw, self.up_axis)
        node.trans = np.asarray(center, float).reshape(3)
        node.confidence = float(np.average(node._confidences, weights=weights))

        # Per-embedding confidence-weighted running mean (each space fused independently).
        for key, vec in obs.embeddings.items():
            unit = normalize(vec)
            if key in node.embeddings:
                w_old = node._emb_weight.get(key, node.confidence)
                w_new = float(obs.confidence)
                node.embeddings[key] = normalize(
                    (w_old * node.embeddings[key] + w_new * unit) / (w_old + w_new))
                node._emb_weight[key] = w_old + w_new
            else:
                node.embeddings[key] = unit
                node._emb_weight[key] = float(obs.confidence)

        node.last_verified_step = step
        node.observed_keyframes.append((obs.keyframe_id, obs.visibility_score))
        if obs.pointcloud_ref:
            node.pointcloud_ref = obs.pointcloud_ref

    def embedding_dims(self) -> dict:
        for node in self.nodes.values():
            return {k: int(np.asarray(v).reshape(-1).shape[0]) for k, v in node.embeddings.items()}
        return {}

    def save_to_file(self, edge_only: bool = False) -> None:
        """Write objects.json (schema_version + objects) and edges_object.txt."""
        if not edge_only:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "edge_type": self.edge_type,
                "embedding_dims": self.embedding_dims(),
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
