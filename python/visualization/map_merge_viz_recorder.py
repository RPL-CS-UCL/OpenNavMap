from __future__ import annotations

from pathlib import Path
from typing import List

from visualization.map_merge_result_replay import MapMergeResultReplay
from visualization.map_merge_rerun_writer import MapMergeRerunWriter
from visualization.map_merge_viz_events import MapMergeVizEvent


class MapMergeVizRecorder:
    """Runtime facade for producing offline map-merge Rerun recordings."""

    def __init__(
        self,
        output_path: Path,
        image_format: str = "jpg",
        jpeg_quality: int = 85,
        dmatrix_format: str = "png",
        axis_scale: str = "auto",
    ) -> None:
        self.output_path = Path(output_path)
        self.writer = MapMergeRerunWriter(
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            dmatrix_format=dmatrix_format,
            axis_scale=axis_scale,
        )
        self.events: List[MapMergeVizEvent] = []

    def record_event(self, event: MapMergeVizEvent) -> None:
        self.events.append(event)

    def record_result_map(self, map_root: Path) -> None:
        map_root = Path(map_root)
        replay_root = map_root.parent if map_root.name.startswith("merge_") else map_root
        self.events = MapMergeResultReplay(replay_root).build_events()

    def save(self) -> None:
        self.writer.write(self.events, self.output_path)
