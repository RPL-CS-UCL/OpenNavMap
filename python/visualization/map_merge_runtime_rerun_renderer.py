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

    def __init__(self, event_dir: Path) -> None:
        self.event_dir = Path(event_dir)

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

    @staticmethod
    def _log_stage(rr, event: Dict[str, Any]) -> None:
        display_text = event.get("payload", {}).get("display_text", "")
        rr.log("/status/stage_summary", rr.TextDocument(display_text))

    @staticmethod
    def _log_node(rr, event: Dict[str, Any]) -> None:
        payload = event["payload"]
        submap_id = int(event["submap_id"])
        node_id = int(payload["node_id"])
        position = np.asarray(payload["position"], dtype=np.float32)
        color = np.asarray([[70, 130, 220]], dtype=np.uint8) if submap_id == 0 else np.asarray([[240, 140, 40]], dtype=np.uint8)
        rr.log(
            f"/world/submaps/{submap_id}/nodes/{node_id:06d}",
            rr.Points3D([position], colors=color, radii=0.18),
        )

    @staticmethod
    def _log_graph_edge(rr, event: Dict[str, Any]) -> None:
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
        rr.log(
            f"/world/submaps/{submap_id}/edges/{edge_type}/{node_a:06d}_{node_b:06d}",
            rr.LineStrips3D(
                strips=[np.asarray([position_a, position_b], dtype=np.float32)],
                colors=np.asarray([colors.get(edge_type, [120, 120, 120])], dtype=np.uint8),
                radii=0.035,
            ),
        )

    def _log_dmatrix(self, rr, event: Dict[str, Any]) -> None:
        artifact_path = self._resolve_artifact(event, "dmatrix_png")
        if artifact_path is not None and artifact_path.exists():
            rr.log("/evidence/dmatrix", rr.EncodedImage(path=artifact_path, media_type="image/png"))

    def _log_current_image(self, rr, event: Dict[str, Any]) -> None:
        image_path = event.get("payload", {}).get("rgb_img_path")
        if image_path is None:
            return
        image_path = Path(image_path)
        if image_path.exists():
            rr.log("/evidence/current_keyframe_image", rr.EncodedImage(path=image_path, media_type="image/jpeg"))

    @staticmethod
    def _log_metric_edge(rr, event: Dict[str, Any]) -> None:
        payload = event.get("payload", {})
        text = (
            "Metric edge added\n"
            f"DB node: {payload.get('db_node_id')}\n"
            f"Query node: {payload.get('query_node_id')}\n"
            f"Confidence: {payload.get('conf')}"
        )
        rr.log("/status/metric_edge", rr.TextDocument(text))

    def _resolve_artifact(self, event: Dict[str, Any], key: str) -> Optional[Path]:
        artifact = event.get("artifacts", {}).get(key)
        if artifact is None:
            return None
        artifact_path = Path(artifact)
        if artifact_path.is_absolute():
            return artifact_path
        return self.event_dir / artifact_path
