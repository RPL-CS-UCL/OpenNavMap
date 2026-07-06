from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_runtime_events(event_dir: Path) -> List[Dict[str, Any]]:
    events_path = Path(event_dir) / "demo_events.jsonl"
    with events_path.open("r", encoding="utf-8") as event_file:
        return [json.loads(line) for line in event_file if line.strip()]


class MapMergeRuntimeRerunRenderer:
    """Render runtime map-merge demo events to a Rerun recording."""

    SUPPORTED_EVENTS = {
        "stage_annotation",
        "vio_node_observed",
        "odom_edge_observed",
        "covis_edge_observed",
        "trav_edge_observed",
        "dmatrix_computed",
        "metric_edge_added",
        "map_committed",
    }

    EDGE_COLORS = {
        "odom": [0, 255, 0],
        "covis": [0, 100, 255],
        "trav": [255, 255, 255],
    }

    AXIS_LEN = 0.45
    IMAGE_PLANE_DISTANCE = 1.0
    WORLD_AXIS_LEN = 10.0

    def __init__(self, event_dir: Path, render_trace_path: Optional[Path] = None) -> None:
        self.event_dir = Path(event_dir)
        self.render_trace_path = Path(render_trace_path) if render_trace_path else None
        self.render_trace: List[Dict[str, Any]] = []
        self._node_demo_step: Dict[Tuple[int, int], int] = {}
        self._node_positions: Dict[Tuple[int, int], np.ndarray] = {}

    def build_time_map(self, events: List[Dict[str, Any]]) -> None:
        self._node_demo_step = {}
        self._node_positions = {}
        for event in events:
            if event.get("event_type") == "vio_node_observed":
                sid = event.get("submap_id")
                kf_id = event.get("keyframe_id")
                step = event.get("demo_step")
                if sid is not None and kf_id is not None and step is not None:
                    self._node_demo_step[(int(sid), int(kf_id))] = int(step)
                payload = event.get("payload", {})
                node_id = payload.get("node_id")
                position = payload.get("position")
                if sid is not None and node_id is not None and position is not None:
                    self._node_positions[(int(sid), int(node_id))] = np.asarray(position, dtype=np.float32)

    def write(self, output_path: Path) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rr.init("opennavmap_runtime_map_merge", spawn=False)
        self._send_blueprint(rr, rrb)
        self._log_world_axes(rr)
        events = load_runtime_events(self.event_dir)
        self.build_time_map(events)
        for event in events:
            self.log_event(rr, event)
        rr.save(str(output_path))
        self.write_trace()

    def write_trace(self) -> None:
        if self.render_trace_path is None:
            return
        self.render_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.render_trace_path.open("w", encoding="utf-8") as trace_file:
            for record in self.render_trace:
                trace_file.write(json.dumps(record, sort_keys=True) + "\n")

    def _log(self, rr, event: Dict[str, Any], entity_path: str, archetype_name: str, archetype) -> None:
        rr.log(entity_path, archetype)
        self.render_trace.append(
            {
                "entity_path": entity_path,
                "archetype": archetype_name,
                "event_type": event.get("event_type"),
                "demo_step": event.get("demo_step"),
                "merge_step": event.get("merge_step"),
                "submap_id": event.get("submap_id"),
                "keyframe_id": event.get("keyframe_id"),
            }
        )

    @staticmethod
    def _send_blueprint(rr, rrb) -> None:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Map Merge Process",
                    origin="/",
                    background=rrb.BackgroundKind.GradientDark,
                    overrides={
                        "edges/ref/covis/**": [rrb.components.Visible(False)],
                        "edges/query/covis/**": [rrb.components.Visible(False)],
                        "edges/ref/trav/**": [rrb.components.Visible(False)],
                        "edges/query/trav/**": [rrb.components.Visible(False)],
                    },
                ),
                rrb.Vertical(
                    rrb.TextDocumentView(name="Stage Summary", origin="/status/stage_summary"),
                    rrb.Spatial2DView(
                        name="Current Keyframe Image",
                        origin="evidence/current_keyframe_image",
                        background=rrb.BackgroundKind.GradientDark,
                    ),
                    rrb.Spatial2DView(
                        name="Difference Matrix",
                        origin="evidence/dmatrix",
                        background=rrb.BackgroundKind.GradientDark,
                    ),
                ),
                column_shares=[3, 1],
            ),
        )
        rr.send_blueprint(blueprint)

    def _log_world_axes(self, rr) -> None:
        L = self.WORLD_AXIS_LEN
        rr.log(
            "world/axes",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0]] * 3,
                vectors=[[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
                radii=0.08,
                colors=np.asarray([[220, 50, 50], [50, 180, 50], [50, 50, 220]], dtype=np.uint8),
            ),
        )

    def log_event(self, rr, event: Dict[str, Any]) -> None:
        event_type = event.get("event_type")
        if event_type not in self.SUPPORTED_EVENTS:
            return
        self._set_time(rr, event)
        if event_type == "stage_annotation":
            self._log_stage(rr, event)
        elif event_type == "vio_node_observed":
            self._log_keyframe_camera(rr, event)
        elif event_type in {"odom_edge_observed", "covis_edge_observed", "trav_edge_observed"}:
            self._log_graph_edge(rr, event)
        elif event_type == "dmatrix_computed":
            self._log_dmatrix(rr, event)
        elif event_type == "metric_edge_added":
            self._log_metric_edge(rr, event)
        elif event_type == "map_committed":
            self._log_final_map(rr, event)

    def _set_time(self, rr, event: Dict[str, Any]) -> None:
        event_type = event.get("event_type")
        demo_step = int(event.get("demo_step", 0))
        if event_type in {"odom_edge_observed", "covis_edge_observed", "trav_edge_observed"}:
            sid = event.get("submap_id")
            kf_id = event.get("keyframe_id")
            if sid is not None and kf_id is not None:
                demo_step = self._node_demo_step.get((int(sid), int(kf_id)), demo_step)
        rr.set_time_sequence("demo_step", demo_step)
        merge_step = event.get("merge_step")
        if merge_step is not None:
            rr.set_time_sequence("merge_step", int(merge_step))
        keyframe_id = event.get("keyframe_id")
        if keyframe_id is not None:
            rr.set_time_sequence("keyframe_id", int(keyframe_id))

    def _log_stage(self, rr, event: Dict[str, Any]) -> None:
        title = event.get("payload", {}).get("title", "")
        markdown = f"# {title}" if title else ""
        self._log(
            rr,
            event,
            "/status/stage_summary",
            "TextDocument",
            rr.TextDocument(markdown, media_type="text/markdown"),
        )

    @staticmethod
    def _camera_group(submap_id: int) -> str:
        return "ref" if submap_id == 0 else "query"

    def _log_camera(
        self,
        rr,
        event: Dict[str, Any],
        base_path: str,
        position,
        quat_xyzw,
        intrinsics,
        image_size,
        image_path: Optional[str],
    ) -> None:
        transform_kwargs: Dict[str, Any] = {
            "translation": np.asarray(position, dtype=np.float32),
            "axis_length": self.AXIS_LEN,
        }
        if quat_xyzw is not None:
            transform_kwargs["rotation"] = rr.Quaternion(xyzw=np.asarray(quat_xyzw, dtype=np.float32))
        self._log(rr, event, base_path, "Transform3D", rr.Transform3D(**transform_kwargs))

        if intrinsics is not None and image_size is not None:
            K = np.asarray(intrinsics, dtype=np.float32)
            width, height = int(image_size[0]), int(image_size[1])
            self._log(
                rr,
                event,
                f"{base_path}/image",
                "Pinhole",
                rr.Pinhole(
                    focal_length=[float(K[0, 0]), float(K[1, 1])],
                    principal_point=[float(K[0, 2]), float(K[1, 2])],
                    width=width,
                    height=height,
                    image_plane_distance=self.IMAGE_PLANE_DISTANCE,
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )

        if image_path is not None:
            resolved = self._resolve_image_path(image_path)
            if resolved is not None and resolved.exists():
                self._log(
                    rr,
                    event,
                    f"{base_path}/image",
                    "ImageEncoded",
                    rr.ImageEncoded(path=resolved),
                )

    def _log_keyframe_camera(self, rr, event: Dict[str, Any]) -> None:
        payload = event["payload"]
        submap_id = int(event["submap_id"])
        node_id = int(payload["node_id"])
        group = self._camera_group(submap_id)
        camera_path = f"cameras/{group}/{node_id}"

        self._node_positions[(submap_id, node_id)] = np.asarray(payload["position"], dtype=np.float32)

        self._log_camera(
            rr, event, camera_path,
            position=payload["position"],
            quat_xyzw=payload.get("quat_xyzw"),
            intrinsics=payload.get("raw_K", payload.get("K")),
            image_size=payload.get("raw_img_size", payload.get("img_size")),
            image_path=payload.get("rgb_img_path"),
        )

        image_path = payload.get("rgb_img_path")
        if image_path is not None:
            resolved = self._resolve_image_path(image_path)
            if resolved is not None and resolved.exists():
                self._log(
                    rr,
                    event,
                    "evidence/current_keyframe_image",
                    "ImageEncoded",
                    rr.ImageEncoded(path=resolved),
                )

    def _log_graph_edge(self, rr, event: Dict[str, Any]) -> None:
        payload = event["payload"]
        submap_id = int(event["submap_id"])
        group = self._camera_group(submap_id)
        edge_type = payload["edge_type"]
        node_a = int(payload["nodeAid"])
        node_b = int(payload["nodeBid"])
        position_a = np.asarray(payload["position_a"], dtype=np.float32)
        position_b = np.asarray(payload["position_b"], dtype=np.float32)
        color = self.EDGE_COLORS.get(edge_type, [120, 120, 120])
        self._log(
            rr,
            event,
            f"edges/{group}/{edge_type}/{node_a}_{node_b}",
            "LineStrips3D",
            rr.LineStrips3D(
                strips=[np.asarray([position_a, position_b], dtype=np.float32)],
                colors=np.asarray([color], dtype=np.uint8),
                radii=0.035,
            ),
        )

    def _log_dmatrix(self, rr, event: Dict[str, Any]) -> None:
        artifact_path = self._resolve_artifact_path(event, "dmatrix_png")
        if artifact_path is not None and artifact_path.exists():
            self._log(
                rr,
                event,
                "evidence/dmatrix",
                "ImageEncoded",
                rr.ImageEncoded(path=artifact_path),
            )

    def _log_metric_edge(self, rr, event: Dict[str, Any]) -> None:
        payload = event["payload"]
        db_node_id = int(payload["db_node_id"])
        query_node_id = int(payload["query_node_id"])
        query_submap_id = int(event["submap_id"])
        pos_a = self._node_positions.get((0, db_node_id))
        pos_b = self._node_positions.get((query_submap_id, query_node_id))
        if pos_a is not None and pos_b is not None:
            self._log(
                rr,
                event,
                f"edges/metric/{db_node_id}_{query_node_id}",
                "LineStrips3D",
                rr.LineStrips3D(
                    strips=[np.asarray([pos_a, pos_b], dtype=np.float32)],
                    colors=np.asarray([[0, 255, 0]], dtype=np.uint8),
                    radii=0.05,
                ),
            )

    def _log_final_map(self, rr, event: Dict[str, Any]) -> None:
        payload = event["payload"]
        nodes = payload.get("nodes", [])
        if not nodes:
            return

        merge_step = event.get("merge_step", 0)
        if int(merge_step) >= 1:
            self._log(rr, event, "cameras/", "Clear", rr.Clear(recursive=True))
            self._log(rr, event, "edges/", "Clear", rr.Clear(recursive=True))

        for n in nodes:
            nid = int(n["node_id"])
            self._node_positions[(0, nid)] = np.asarray(n["position"], dtype=np.float32)

        positions = np.asarray(
            [n["position"] for n in nodes],
            dtype=np.float32,
        )
        self._log(
            rr,
            event,
            "final_map/nodes",
            "Points3D",
            rr.Points3D(
                positions=positions,
                radii=0.08,
                colors=np.asarray([[255, 255, 0]] * len(nodes), dtype=np.uint8),
            ),
        )

        if int(merge_step) >= 1:
            for n in nodes:
                node_id = int(n["node_id"])
                camera_path = f"final_map/cameras/{node_id}"
                self._log_camera(
                    rr, event, camera_path,
                    position=n["position"],
                    quat_xyzw=n.get("quat_xyzw"),
                    intrinsics=n.get("raw_K"),
                    image_size=n.get("raw_img_size"),
                    image_path=n.get("rgb_img_path"),
                )

        final_positions: Dict[int, np.ndarray] = {}
        for n in nodes:
            final_positions[int(n["node_id"])] = np.asarray(n["position"], dtype=np.float32)
        edges = payload.get("edges", {})
        for edge_type, edge_list in edges.items():
            color = self.EDGE_COLORS.get(edge_type, [120, 120, 120])
            strips = []
            for edge in edge_list:
                na, nb = int(edge[0]), int(edge[1])
                if na in final_positions and nb in final_positions:
                    strips.append(
                        np.asarray([final_positions[na], final_positions[nb]], dtype=np.float32)
                    )
            if strips:
                self._log(
                    rr,
                    event,
                    f"final_map/edges/{edge_type}",
                    "LineStrips3D",
                    rr.LineStrips3D(
                        strips=strips,
                        colors=np.asarray([color] * len(strips), dtype=np.uint8),
                        radii=0.03,
                    ),
                )

    def _resolve_image_path(self, path: str) -> Optional[Path]:
        image_path = Path(path)
        if image_path.is_absolute():
            return image_path
        return self.event_dir / image_path

    def _resolve_artifact_path(self, event: Dict[str, Any], key: str) -> Optional[Path]:
        artifact = event.get("artifacts", {}).get(key)
        if artifact is None:
            return None
        path = Path(artifact)
        if path.is_absolute():
            return path
        return self.event_dir / path
