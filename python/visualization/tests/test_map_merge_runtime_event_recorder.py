import json
from pathlib import Path

import numpy as np

from visualization.map_merge_runtime_event_recorder import MapMergeRuntimeEventRecorder


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_recorder_creates_metadata_and_jsonl(tmp_path: Path) -> None:
    recorder = MapMergeRuntimeEventRecorder(tmp_path / "rerun_viz")
    recorder.write_metadata({"dataset": "unit", "scene": "s00000"})
    recorder.record_event(
        merge_step=0,
        stage="recording_started",
        event_type="recording_started",
        submap_id=None,
        keyframe_id=None,
        payload={"message": "start"},
    )

    metadata = json.loads((tmp_path / "rerun_viz" / "metadata.json").read_text())
    events = _read_jsonl(tmp_path / "rerun_viz" / "demo_events.jsonl")

    assert metadata["dataset"] == "unit"
    assert events[0]["demo_step"] == 0
    assert events[0]["stage"] == "recording_started"
    assert events[0]["payload"] == {"message": "start"}


def test_recorder_serializes_numpy_payload(tmp_path: Path) -> None:
    recorder = MapMergeRuntimeEventRecorder(tmp_path / "rerun_viz")
    recorder.record_event(
        merge_step=1,
        stage="vio_node_observed",
        event_type="vio_node_observed",
        submap_id=3,
        keyframe_id=7,
        payload={
            "node_id": np.int64(7),
            "position": np.array([1.0, 2.0, 3.0]),
            "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        },
    )

    events = _read_jsonl(tmp_path / "rerun_viz" / "demo_events.jsonl")

    assert events[0]["payload"]["node_id"] == 7
    assert events[0]["payload"]["position"] == [1.0, 2.0, 3.0]
    assert events[0]["payload"]["quat"] == [0.0, 0.0, 0.0, 1.0]


def test_recorder_writes_artifact_relative_paths(tmp_path: Path) -> None:
    recorder = MapMergeRuntimeEventRecorder(tmp_path / "rerun_viz")
    artifact_path = recorder.artifact_path(merge_step=1, name="dmatrix.png")
    artifact_path.write_text("fake", encoding="utf-8")
    recorder.record_event(
        merge_step=1,
        stage="dmatrix_computed",
        event_type="dmatrix_computed",
        submap_id=1,
        keyframe_id=None,
        payload={"shape": [2, 3]},
        artifacts={"dmatrix_png": artifact_path},
    )

    events = _read_jsonl(tmp_path / "rerun_viz" / "demo_events.jsonl")

    assert events[0]["artifacts"] == {"dmatrix_png": "artifacts/step_001/dmatrix.png"}
