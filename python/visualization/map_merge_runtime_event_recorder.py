from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class MapMergeRuntimeEventRecorder:
    """Append-only runtime event recorder for map-merge demo visualization."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.artifacts_dir = self.output_dir / "artifacts"
        self.events_path = self.output_dir / "demo_events.jsonl"
        self.metadata_path = self.output_dir / "metadata.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.demo_step = 0

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        self.metadata_path.write_text(
            json.dumps(self._to_jsonable(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def artifact_path(self, merge_step: int, name: str) -> Path:
        step_dir = self.artifacts_dir / f"step_{merge_step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir / name

    def record_event(
        self,
        merge_step: int,
        stage: str,
        event_type: str,
        submap_id: Optional[int],
        keyframe_id: Optional[int],
        payload: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Path]] = None,
    ) -> None:
        event = {
            "demo_step": self.demo_step,
            "merge_step": merge_step,
            "stage": stage,
            "event_type": event_type,
            "submap_id": submap_id,
            "keyframe_id": keyframe_id,
            "payload": self._to_jsonable(payload or {}),
            "artifacts": self._relative_artifacts(artifacts or {}),
        }
        with self.events_path.open("a", encoding="utf-8") as event_file:
            event_file.write(json.dumps(event, sort_keys=True) + "\n")
        self.demo_step += 1

    def _relative_artifacts(self, artifacts: Dict[str, Path]) -> Dict[str, str]:
        relative = {}
        for key, path in artifacts.items():
            resolved = Path(path).resolve()
            try:
                relative[key] = str(resolved.relative_to(self.output_dir.resolve()))
            except ValueError:
                relative[key] = str(resolved)
        return relative

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        return value
