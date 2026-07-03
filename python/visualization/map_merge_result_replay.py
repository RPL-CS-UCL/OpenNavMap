from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from visualization.map_merge_viz_events import (
    ArtifactRefs,
    MapMergeVizEvent,
    STAGE_MERGED,
    STAGE_SEQUENCE_MATCHING,
    STAGE_STATUS,
    compute_axis_scale,
    w2c_vec_to_camera_position,
)


@dataclass(frozen=True)
class MergeStep:
    name: str
    path: Path
    submap_ids: Tuple[str, ...]
    merge_step: int


class MapMergeResultReplay:
    """Build best-effort visualization events from saved map-merge results."""

    def __init__(self, result_dir: Path) -> None:
        self.result_dir = Path(result_dir)

    def discover_steps(self) -> List[MergeStep]:
        if not self.result_dir.exists():
            raise FileNotFoundError(f"Result directory not found: {self.result_dir}")

        if (self.result_dir / "poses.txt").exists():
            submap_ids = self._parse_merge_name(self.result_dir.name) or (self.result_dir.name,)
            return [
                MergeStep(
                    name=self.result_dir.name,
                    path=self.result_dir,
                    submap_ids=submap_ids,
                    merge_step=max(len(submap_ids) - 1, 0),
                )
            ]

        steps = []
        for path in self.result_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("merge_"):
                continue
            submap_ids = self._parse_merge_name(path.name)
            if submap_ids is None:
                continue
            steps.append(
                MergeStep(
                    name=path.name,
                    path=path,
                    submap_ids=submap_ids,
                    merge_step=len(submap_ids) - 1,
                )
            )
        return sorted(steps, key=lambda step: (len(step.submap_ids), step.submap_ids))

    def build_events(self) -> List[MapMergeVizEvent]:
        events: List[MapMergeVizEvent] = []
        previous_edges: Dict[str, List[Tuple[int, int, float]]] = {
            "odom": [],
            "covis": [],
            "trav": [],
        }
        previous_node_count = 0
        for step in self.discover_steps():
            step_events, previous_node_count, previous_edges = self._build_step_events(
                step, previous_node_count, previous_edges
            )
            events.extend(step_events)
        return events

    @staticmethod
    def _parse_merge_name(name: str) -> Optional[Tuple[str, ...]]:
        suffix = name.removeprefix("merge_")
        if not suffix:
            return None
        tokens = suffix.split("_")
        if not all(token.isdigit() for token in tokens):
            return None
        return tuple(tokens)

    def _build_step_events(
        self,
        step: MergeStep,
        previous_node_count: int,
        previous_edges: Dict[str, List[Tuple[int, int, float]]],
    ) -> Tuple[
        List[MapMergeVizEvent], int, Dict[str, List[Tuple[int, int, float]]]
    ]:
        poses = self._read_poses(step.path / "poses.txt")
        positions = self._camera_positions(poses.values())
        axis_length, axis_radius = compute_axis_scale(positions)
        current_edges = {
            "odom": self._read_edges(step.path / "edges_odom.txt"),
            "covis": self._read_edges(step.path / "edges_covis.txt"),
            "trav": self._read_edges(step.path / "edges_trav.txt"),
        }
        edge_counts = {
            "odom": len(current_edges["odom"]),
            "covis": len(current_edges["covis"]),
            "trav": len(current_edges["trav"]),
        }
        pose_names = list(poses.keys())
        events = self._build_node_events(step, pose_names, positions, axis_length, axis_radius)
        events.extend(self._build_intrasubmap_edge_events(step, positions, current_edges["odom"]))
        events.extend(
            self._build_intersubmap_edge_events(
                step, positions, previous_node_count, previous_edges, current_edges
            )
        )
        events.append(
            MapMergeVizEvent(
                event_type="submap_merged",
                merge_step=step.merge_step,
                stage=STAGE_MERGED,
                submap_id=step.submap_ids[-1],
                keyframe_id=self._synthetic_keyframe_id(step.merge_step, 0),
                payload={
                    "merge_name": step.name,
                    "num_poses": len(poses),
                    "pose_names": pose_names,
                    "positions": positions,
                    "edge_counts": edge_counts,
                    "axis_length": axis_length,
                    "axis_radius": axis_radius,
                    "edges": current_edges,
                },
                artifact_refs=self._collect_pose_graph_refs(step.path),
            )
        )

        dmatrix = self._find_first(step.path / "preds", ["difference_matrix*", "D_matrix*"])
        if dmatrix is not None:
            events.append(
                MapMergeVizEvent(
                    event_type="dmatrix_ready",
                    merge_step=step.merge_step,
                    stage=STAGE_SEQUENCE_MATCHING,
                    submap_id=step.submap_ids[-1],
                    keyframe_id=self._synthetic_keyframe_id(step.merge_step, 1),
                    payload={"source": "read-only artifact"},
                    artifact_refs=ArtifactRefs(dmatrix=dmatrix),
                )
            )

        culling_image = self._find_first(step.path / "preds" / "kf_vis", ["*.jpg", "*.png"])
        if culling_image is not None:
            events.append(
                MapMergeVizEvent(
                    event_type="keyframe_culling_result",
                    merge_step=step.merge_step,
                    stage="culling",
                    submap_id=step.submap_ids[-1],
                    keyframe_id=self._synthetic_keyframe_id(step.merge_step, 2),
                    payload={"source": "read-only artifact"},
                    artifact_refs=ArtifactRefs(keyframe_culling=culling_image),
                )
            )

        events.extend(self._readonly_warning_events(step))
        return events, len(poses), current_edges

    def _build_node_events(
        self,
        step: MergeStep,
        pose_names: List[str],
        positions: np.ndarray,
        axis_length: float,
        axis_radius: float,
    ) -> List[MapMergeVizEvent]:
        events: List[MapMergeVizEvent] = []
        for node_id, pose_name in enumerate(pose_names):
            image_path = step.path / pose_name
            events.append(
                MapMergeVizEvent(
                    event_type="node_observed",
                    merge_step=step.merge_step,
                    stage="vio_pose",
                    submap_id=step.submap_ids[-1],
                    keyframe_id=self._keyframe_id(step.merge_step, node_id),
                    payload={
                        "node_id": node_id,
                        "pose_name": pose_name,
                        "position": positions[node_id],
                        "axis_length": axis_length,
                        "axis_radius": axis_radius,
                    },
                    artifact_refs=ArtifactRefs(
                        current_image=image_path if image_path.exists() else None
                    ),
                )
            )
        return events

    def _build_intrasubmap_edge_events(
        self,
        step: MergeStep,
        positions: np.ndarray,
        odom_edges: List[Tuple[int, int, float]],
    ) -> List[MapMergeVizEvent]:
        events: List[MapMergeVizEvent] = []
        for node_a, node_b, weight in odom_edges:
            if node_a >= len(positions) or node_b >= len(positions):
                continue
            endpoint_id = max(node_a, node_b)
            events.append(
                MapMergeVizEvent(
                    event_type="intrasubmap_edge_observed",
                    merge_step=step.merge_step,
                    stage="vio_pose",
                    submap_id=step.submap_ids[-1],
                    keyframe_id=self._keyframe_id(step.merge_step, endpoint_id),
                    payload={
                        "edge_type": "odom",
                        "node_a": node_a,
                        "node_b": node_b,
                        "position_a": positions[node_a],
                        "position_b": positions[node_b],
                        "weight": weight,
                    },
                )
            )
        return events

    def _build_intersubmap_edge_events(
        self,
        step: MergeStep,
        positions: np.ndarray,
        previous_node_count: int,
        previous_edges: Dict[str, List[Tuple[int, int, float]]],
        current_edges: Dict[str, List[Tuple[int, int, float]]],
    ) -> List[MapMergeVizEvent]:
        events: List[MapMergeVizEvent] = []
        if previous_node_count <= 0:
            return events
        for edge_type, edges in current_edges.items():
            previous_edge_keys = self._edge_key_set(previous_edges.get(edge_type, []))
            for node_a, node_b, weight in edges:
                if node_a >= len(positions) or node_b >= len(positions):
                    continue
                edge_key = tuple(sorted((node_a, node_b)))
                is_new_edge = edge_key not in previous_edge_keys
                crosses_boundary = (node_a < previous_node_count <= node_b) or (
                    node_b < previous_node_count <= node_a
                )
                if not is_new_edge or not crosses_boundary:
                    continue
                endpoint_id = max(node_a, node_b)
                events.append(
                    MapMergeVizEvent(
                        event_type="intersubmap_edge_observed",
                        merge_step=step.merge_step,
                        stage="metric_loc",
                        submap_id=step.submap_ids[-1],
                        keyframe_id=self._keyframe_id(step.merge_step, endpoint_id),
                        payload={
                            "edge_type": edge_type,
                            "node_a": node_a,
                            "node_b": node_b,
                            "position_a": positions[node_a],
                            "position_b": positions[node_b],
                            "weight": weight,
                            "source": "edge_diff",
                        },
                    )
                )
        return events

    @staticmethod
    def _read_poses(path: Path) -> Dict[str, np.ndarray]:
        poses: Dict[str, np.ndarray] = {}
        if not path.exists():
            return poses
        with path.open("r", encoding="utf-8") as pose_file:
            for line in pose_file:
                fields = line.strip().split()
                if len(fields) != 8 or fields[0].startswith("#"):
                    continue
                poses[fields[0]] = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
        return poses

    @staticmethod
    def _read_edges(path: Path) -> List[Tuple[int, int, float]]:
        edges: List[Tuple[int, int, float]] = []
        if not path.exists():
            return edges
        with path.open("r", encoding="utf-8") as edge_file:
            for line in edge_file:
                fields = line.strip().split()
                if len(fields) < 2 or fields[0].startswith("#"):
                    continue
                weight = float(fields[2]) if len(fields) > 2 else 1.0
                edges.append((int(fields[0]), int(fields[1]), weight))
        return edges

    @staticmethod
    def _camera_positions(pose_vectors: Iterable[np.ndarray]) -> np.ndarray:
        positions = [w2c_vec_to_camera_position(pose_vec) for pose_vec in pose_vectors]
        if not positions:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(positions)

    @staticmethod
    def _collect_pose_graph_refs(step_path: Path) -> ArtifactRefs:
        preds_dir = step_path / "preds"
        initial = preds_dir / "initial_pose_graph.g2o"
        refined = preds_dir / "refine_pose_graph.g2o"
        return ArtifactRefs(
            pose_graph_initial=initial if initial.exists() else None,
            pose_graph_refined=refined if refined.exists() else None,
        )

    @staticmethod
    def _find_first(root: Path, patterns: List[str]) -> Optional[Path]:
        if not root.exists():
            return None
        matches: List[Path] = []
        for pattern in patterns:
            matches.extend(root.glob(pattern))
        return sorted(matches)[0] if matches else None

    @staticmethod
    def _synthetic_keyframe_id(merge_step: int, stage_offset: int) -> int:
        return merge_step * 100000 + stage_offset

    @staticmethod
    def _keyframe_id(merge_step: int, node_id: int) -> int:
        return merge_step * 100000 + node_id

    @staticmethod
    def _edge_key_set(edges: List[Tuple[int, int, float]]) -> set[Tuple[int, int]]:
        return {tuple(sorted((node_a, node_b))) for node_a, node_b, _weight in edges}

    def _readonly_warning_events(self, step: MergeStep) -> List[MapMergeVizEvent]:
        messages = [
            "GV per-keyframe details unavailable in saved result; showing saved stage artifact in read-only mode.",
            "Sequence matching path unavailable in saved result; showing saved D-matrix artifact in read-only mode.",
            "Only read-only replay is enabled; no recomputation was performed.",
        ]
        return [
            MapMergeVizEvent(
                event_type="status_note",
                merge_step=step.merge_step,
                stage=STAGE_STATUS,
                submap_id=step.submap_ids[-1],
                keyframe_id=self._synthetic_keyframe_id(step.merge_step, 90 + idx),
                payload={"message": message},
            )
            for idx, message in enumerate(messages)
        ]
