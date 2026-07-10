from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
from io import BytesIO

import numpy as np

from visualization.map_merge_viz_events import MapMergeVizEvent


class MapMergeRerunWriter:
    """Write map-merge visualization events to a Rerun recording."""

    def __init__(
        self,
        image_format: str = "jpg",
        jpeg_quality: int = 85,
        dmatrix_format: str = "png",
        axis_scale: str = "auto",
    ) -> None:
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.dmatrix_format = dmatrix_format
        self.axis_scale = axis_scale

    def write(self, events: Iterable[MapMergeVizEvent], output_path: Path) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        event_list = list(events)

        rr.init("opennavmap_map_merge", spawn=False)
        self._send_blueprint(rr, rrb)
        for event in event_list:
            self._set_time(rr, event)
            self._log_status(rr, event)
            self._log_world_axes(rr, event)
            self._log_map(rr, event)
            self._log_artifacts(rr, event)
        rr.save(str(output_path))

    @staticmethod
    def _send_blueprint(rr, rrb) -> None:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Map Merge Process", origin="/world"),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        name="Current Keyframe Image",
                        origin="/evidence/current_keyframe_image",
                    ),
                    rrb.Spatial2DView(name="Difference Matrix", origin="/evidence/dmatrix"),
                    rrb.Spatial2DView(
                        name="Culling / Matching Evidence",
                        origin="/evidence/keyframe_culling",
                    ),
                    rrb.TextDocumentView(name="Stage Summary", origin="/status/stage_summary"),
                ),
                column_shares=[3, 1],
            ),
            auto_views=False,
        )
        rr.send_blueprint(blueprint)

    @staticmethod
    def _set_time(rr, event: MapMergeVizEvent) -> None:
        rr.set_time_sequence("merge_step", event.merge_step)
        rr.set_time_sequence("keyframe_id", event.keyframe_id)

    @staticmethod
    def _log_status(rr, event: MapMergeVizEvent) -> None:
        message = event.payload.get("message")
        if message is None:
            message = (
                f"Stage: {event.stage}\n"
                f"Event: {event.event_type}\n"
                f"Merge step: {event.merge_step}\n"
                f"Submap: {event.submap_id}"
            )
        rr.log("/status/stage_summary", rr.TextDocument(str(message)))

    @staticmethod
    def _log_world_axes(rr, event: MapMergeVizEvent) -> None:
        axis_length = float(event.payload.get("axis_length", 2.0))
        axis_radius = float(event.payload.get("axis_radius", 0.03))
        origin = [0.0, 0.0, 0.0]
        rr.log(
            "/world/axes",
            rr.Arrows3D(
                origins=[origin, origin, origin],
                vectors=[
                    [axis_length, 0.0, 0.0],
                    [0.0, axis_length, 0.0],
                    [0.0, 0.0, axis_length],
                ],
                radii=axis_radius,
                colors=np.asarray(
                    [[220, 50, 50], [50, 220, 50], [50, 50, 220]], dtype=np.uint8
                ),
            ),
        )

    @staticmethod
    def _log_map(rr, event: MapMergeVizEvent) -> None:
        if event.event_type == "node_observed":
            position = np.asarray(event.payload["position"], dtype=np.float32)
            node_id = int(event.payload["node_id"])
            rr.log(
                f"/world/current_submap/nodes/{node_id:06d}",
                rr.Points3D(
                    [position],
                    colors=np.asarray([[255, 140, 40]], dtype=np.uint8),
                    radii=0.12,
                ),
            )
            return

        if event.event_type in {
            "intrasubmap_edge_observed",
            "intersubmap_edge_observed",
        }:
            position_a = np.asarray(event.payload["position_a"], dtype=np.float32)
            position_b = np.asarray(event.payload["position_b"], dtype=np.float32)
            node_a = int(event.payload["node_a"])
            node_b = int(event.payload["node_b"])
            is_inter = event.event_type == "intersubmap_edge_observed"
            entity_root = "/world/matches/metric_edges" if is_inter else "/world/current_submap/edges/odom"
            color = [40, 220, 90] if is_inter else [255, 170, 70]
            radius = 0.055 if is_inter else 0.035
            rr.log(
                f"{entity_root}/{node_a:06d}_{node_b:06d}",
                rr.LineStrips3D(
                    strips=[np.asarray([position_a, position_b], dtype=np.float32)],
                    radii=radius,
                    colors=np.asarray([color], dtype=np.uint8),
                ),
            )
            return

        positions = event.payload.get("positions")
        if positions is None:
            return
        positions = np.asarray(positions, dtype=np.float32)
        if positions.size == 0:
            return
        rr.log(
            "/world/final_map/nodes",
            rr.Points3D(
                positions,
                colors=np.asarray([[80, 120, 160]], dtype=np.uint8),
                radii=0.08,
            ),
        )
        edge_groups = event.payload.get("edges", {})
        for edge_name, edges in edge_groups.items():
            strips = []
            for node_a, node_b, _weight in edges:
                if node_a < len(positions) and node_b < len(positions):
                    strips.append(np.asarray([positions[node_a], positions[node_b]], dtype=np.float32))
            if strips:
                rr.log(
                    f"/world/final_map/edges/{edge_name}",
                    rr.LineStrips3D(
                        strips=strips,
                        radii=0.02,
                        colors=np.asarray([[100, 149, 237]], dtype=np.uint8),
                    ),
                )

    def _log_artifacts(self, rr, event: MapMergeVizEvent) -> None:
        self._log_encoded_image_artifact(
            rr, "/evidence/current_keyframe_image", event.artifact_refs.current_image
        )
        self._log_dmatrix_artifact(rr, "/evidence/dmatrix", event.artifact_refs.dmatrix)
        self._log_image_artifact(rr, "/evidence/query_image", event.artifact_refs.query_image)
        self._log_image_artifact(rr, "/evidence/reference_image", event.artifact_refs.reference_image)
        self._log_image_artifact(rr, "/evidence/matching_image", event.artifact_refs.matching_image)
        self._log_image_artifact(
            rr, "/evidence/keyframe_culling", event.artifact_refs.keyframe_culling
        )

    @staticmethod
    def _log_image_artifact(rr, entity_path: str, image_path: Optional[Path]) -> None:
        if image_path is None or not Path(image_path).exists():
            return
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        rr.log(entity_path, rr.Image(np.asarray(image)))

    @staticmethod
    def _log_dmatrix_artifact(rr, entity_path: str, image_path: Optional[Path]) -> None:
        if image_path is None or not Path(image_path).exists():
            return
        image_path = Path(image_path)
        if image_path.suffix.lower() == ".png":
            rr.log(entity_path, rr.EncodedImage(path=image_path, media_type="image/png"))
            return

        from PIL import Image

        buffer = BytesIO()
        Image.open(image_path).convert("RGB").save(buffer, format="PNG")
        rr.log(
            entity_path,
            rr.EncodedImage(contents=buffer.getvalue(), media_type="image/png"),
        )

    @staticmethod
    def _log_encoded_image_artifact(rr, entity_path: str, image_path: Optional[Path]) -> None:
        if image_path is None or not Path(image_path).exists():
            return
        image_path = Path(image_path)
        suffix = image_path.suffix.lower()
        media_type = "image/png" if suffix == ".png" else "image/jpeg"
        rr.log(entity_path, rr.EncodedImage(path=image_path, media_type=media_type))
