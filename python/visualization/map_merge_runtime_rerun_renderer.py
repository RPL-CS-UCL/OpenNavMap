from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def load_runtime_events(event_dir: Path) -> List[Dict[str, Any]]:
    events_path = Path(event_dir) / "demo_events.jsonl"
    with events_path.open("r", encoding="utf-8") as event_file:
        return [json.loads(line) for line in event_file if line.strip()]


class MapMergeRuntimeRerunRenderer:
    """Render runtime map-merge demo events to a Rerun recording."""

    def __init__(self, event_dir: Path, render_trace_path: Optional[Path] = None) -> None:
        self.event_dir = Path(event_dir)
        self.render_trace_path = Path(render_trace_path) if render_trace_path else None
        self.render_trace: List[Dict[str, Any]] = []

    def write(self, output_path: Path) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rr.init("opennavmap_runtime_map_merge", spawn=False)
        self._send_blueprint(rr, rrb)
        self._log_world_axes(rr)
        for event in load_runtime_events(self.event_dir):
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
                "keyframe_id": event.get("keyframe_id"),
            }
        )

    @staticmethod
    def _send_blueprint(rr, rrb) -> None:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Map Merge Process", origin="/world"),
                rrb.Vertical(
                    rrb.TextDocumentView(name="Stage Summary", origin="/status/stage_summary"),
                    rrb.Spatial2DView(name="Current Keyframe Image", origin="/evidence/current_keyframe_image"),
                    rrb.Spatial2DView(name="Difference Matrix", origin="/evidence/dmatrix"),
                ),
                column_shares=[3, 1],
            ),
            auto_views=False,
        )
        rr.send_blueprint(blueprint)

    @staticmethod
    def _log_world_axes(rr) -> None:
        origin = [0.0, 0.0, 0.0]
        rr.log(
            "/world/axes",
            rr.Arrows3D(
                origins=[origin, origin, origin],
                vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                radii=0.06,
                colors=np.asarray([[220, 50, 50], [50, 180, 50], [50, 50, 220]], dtype=np.uint8),
            ),
        )

    def log_event(self, rr, event: Dict[str, Any]) -> None:
        self._set_time(rr, event)
        event_type = event.get("event_type")
        if event_type == "stage_annotation":
            self._log_stage(rr, event)
        elif event_type == "vio_node_observed":
            self._log_node(rr, event)
            self._log_current_image(rr, event)
        elif event_type in {"odom_edge_observed", "covis_edge_observed", "trav_edge_observed"}:
            self._log_graph_edge(rr, event)
        elif event_type == "dmatrix_computed":
            self._log_dmatrix(rr, event)
        elif event_type == "metric_edge_added":
            self._log_metric_edge(rr, event)

    @staticmethod
    def _set_time(rr, event: Dict[str, Any]) -> None:
        rr.set_time_sequence("demo_step", int(event.get("demo_step", 0)))
        merge_step = event.get("merge_step")
        if merge_step is not None:
            rr.set_time_sequence("merge_step", int(merge_step))
        keyframe_id = event.get("keyframe_id")
        if keyframe_id is not None:
            rr.set_time_sequence("keyframe_id", int(keyframe_id))

    def _log_stage(self, rr, event: Dict[str, Any]) -> None:
        display_text = event.get("payload", {}).get("display_text", "")
        self._log(rr, event, "/status/stage_summary", "TextDocument", rr.TextDocument(display_text))

    def _log_node(self, rr, event: Dict[str, Any]) -> None:
        payload = event["payload"]
        submap_id = int(event["submap_id"])
        node_id = int(payload["node_id"])
        position = np.asarray(payload["position"], dtype=np.float32)
        quat_xyzw = payload.get("quat_xyzw")
        color = np.asarray([[70, 130, 220]], dtype=np.uint8) if submap_id == 0 else np.asarray([[240, 140, 40]], dtype=np.uint8)
        self._log(
            rr,
            event,
            f"/world/submaps/{submap_id}/nodes/{node_id:06d}",
            "Points3D",
            rr.Points3D([position], colors=color, radii=0.18),
        )
        camera_path = f"/world/submaps/{submap_id}/cameras/{node_id:06d}"
        transform_kwargs = {"translation": position, "axis_length": 1.0}
        if quat_xyzw is not None:
            transform_kwargs["rotation"] = rr.Quaternion(xyzw=np.asarray(quat_xyzw, dtype=np.float32))
        self._log(rr, event, camera_path, "Transform3D", rr.Transform3D(**transform_kwargs))
        intrinsics = payload.get("K") or payload.get("raw_K")
        image_size = payload.get("img_size") or payload.get("raw_img_size")
        if intrinsics is not None and image_size is not None:
            intrinsics = np.asarray(intrinsics, dtype=np.float32)
            width, height = int(image_size[0]), int(image_size[1])
            self._log(
                rr,
                event,
                f"{camera_path}/image",
                "Pinhole",
                rr.Pinhole(
                    focal_length=[float(intrinsics[0, 0]), float(intrinsics[1, 1])],
                    principal_point=[float(intrinsics[0, 2]), float(intrinsics[1, 2])],
                    width=width,
                    height=height,
                    image_plane_distance=0.5,
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )
        image_path = payload.get("rgb_img_path")
        if image_path is not None and Path(image_path).exists():
            self._log(
                rr,
                event,
                f"{camera_path}/image",
                "EncodedImage",
                rr.EncodedImage(path=Path(image_path), media_type="image/jpeg"),
            )

    def _log_graph_edge(self, rr, event: Dict[str, Any]) -> None:
        payload = event["payload"]
        submap_id = int(event["submap_id"])
        edge_type = payload["edge_type"]
        node_a = int(payload["nodeAid"])
        node_b = int(payload["nodeBid"])
        position_a = np.asarray(payload["position_a"], dtype=np.float32)
        position_b = np.asarray(payload["position_b"], dtype=np.float32)
        colors = {
            "odom": [230, 150, 50],
            "covis": [70, 130, 220],
            "trav": [90, 170, 90],
        }
        self._log(
            rr,
            event,
            f"/world/submaps/{submap_id}/edges/{edge_type}/{node_a:06d}_{node_b:06d}",
            "LineStrips3D",
            rr.LineStrips3D(
                strips=[np.asarray([position_a, position_b], dtype=np.float32)],
                colors=np.asarray([colors.get(edge_type, [120, 120, 120])], dtype=np.uint8),
                radii=0.035,
            ),
        )

    def _log_dmatrix(self, rr, event: Dict[str, Any]) -> None:
        artifact_path = self._resolve_artifact(event, "dmatrix_png")
        if artifact_path is not None and artifact_path.exists():
            self._log(
                rr,
                event,
                "/evidence/dmatrix",
                "EncodedImage",
                rr.EncodedImage(path=artifact_path, media_type="image/png"),
            )

    def _log_current_image(self, rr, event: Dict[str, Any]) -> None:
        image_path = event.get("payload", {}).get("rgb_img_path")
        if image_path is None:
            return
        image_path = Path(image_path)
        if image_path.exists():
            self._log(
                rr,
                event,
                "/evidence/current_keyframe_image",
                "EncodedImage",
                rr.EncodedImage(path=image_path, media_type="image/jpeg"),
            )

    def _log_metric_edge(self, rr, event: Dict[str, Any]) -> None:
        payload = event.get("payload", {})
        text = (
            "Metric edge added\n"
            f"DB node: {payload.get('db_node_id')}\n"
            f"Query node: {payload.get('query_node_id')}\n"
            f"Confidence: {payload.get('conf')}"
        )
        self._log(rr, event, "/status/metric_edge", "TextDocument", rr.TextDocument(text))

    def _resolve_artifact(self, event: Dict[str, Any], key: str) -> Optional[Path]:
        artifact = event.get("artifacts", {}).get(key)
        if artifact is None:
            return None
        artifact_path = Path(artifact)
        if artifact_path.is_absolute():
            return artifact_path
        return self.event_dir / artifact_path
