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

    SUPPORTED_EVENTS = {
        "stage_annotation",
        "vio_node_observed",
        "odom_edge_observed",
        "covis_edge_observed",
        "trav_edge_observed",
    }

    EDGE_COLORS = {
        "odom": [230, 150, 50],
        "covis": [70, 130, 220],
        "trav": [90, 170, 90],
    }

    AXIS_LEN = 0.3

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
                "submap_id": event.get("submap_id"),
                "keyframe_id": event.get("keyframe_id"),
            }
        )

    @staticmethod
    def _send_blueprint(rr, rrb) -> None:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Map Merge Process", origin="/"),
                rrb.Vertical(
                    rrb.TextDocumentView(name="Stage Summary", origin="/status/stage_summary"),
                    rrb.Spatial2DView(name="Current Keyframe Image", origin="evidence/current_keyframe_image"),
                ),
                column_shares=[3, 1],
            ),
        )
        rr.send_blueprint(blueprint)

    @staticmethod
    def _log_world_axes(rr) -> None:
        rr.log(
            "world/axes",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0]] * 3,
                vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                radii=0.06,
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

    @staticmethod
    def _camera_group(submap_id: int) -> str:
        return "ref" if submap_id == 0 else "query"

    def _log_keyframe_camera(self, rr, event: Dict[str, Any]) -> None:
        payload = event["payload"]
        submap_id = int(event["submap_id"])
        node_id = int(payload["node_id"])
        group = self._camera_group(submap_id)
        translation = np.asarray(payload["position"], dtype=np.float32)
        quat_xyzw = payload.get("quat_xyzw")
        axis_len = self.AXIS_LEN

        transform_kwargs: Dict[str, Any] = {"translation": translation, "axis_length": axis_len}
        if quat_xyzw is not None:
            transform_kwargs["rotation"] = rr.Quaternion(xyzw=np.asarray(quat_xyzw, dtype=np.float32))

        camera_path = f"sfm/cameras/{group}/{node_id}"
        self._log(rr, event, camera_path, "Transform3D", rr.Transform3D(**transform_kwargs))

        intrinsics = payload.get("K")
        image_size = payload.get("img_size")
        if intrinsics is not None and image_size is not None:
            K = np.asarray(intrinsics, dtype=np.float32)
            width, height = int(image_size[0]), int(image_size[1])
            self._log(
                rr,
                event,
                f"{camera_path}/image",
                "Pinhole",
                rr.Pinhole(
                    focal_length=[float(K[0, 0]), float(K[1, 1])],
                    principal_point=[float(K[0, 2]), float(K[1, 2])],
                    width=width,
                    height=height,
                    image_plane_distance=float(axis_len * 0.5),
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )

        image_path = payload.get("rgb_img_path")
        if image_path is not None:
            resolved = self._resolve_image_path(image_path)
            if resolved is not None and resolved.exists():
                self._log(
                    rr,
                    event,
                    f"{camera_path}/image",
                    "ImageEncoded",
                    rr.ImageEncoded(path=resolved),
                )
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
            f"sfm/edges/{group}/{edge_type}/{node_a}_{node_b}",
            "LineStrips3D",
            rr.LineStrips3D(
                strips=[np.asarray([position_a, position_b], dtype=np.float32)],
                colors=np.asarray([color], dtype=np.uint8),
                radii=0.035,
            ),
        )

    def _resolve_image_path(self, path: str) -> Optional[Path]:
        image_path = Path(path)
        if image_path.is_absolute():
            return image_path
        return self.event_dir / image_path
