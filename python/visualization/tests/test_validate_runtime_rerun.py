import json
from pathlib import Path

from visualization.validate_runtime_rerun import validate_runtime_rerun


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record) + "\n")


def test_validate_runtime_rerun_reports_complete_coverage(tmp_path: Path) -> None:
    event_dir = tmp_path / "rerun_viz"
    event_dir.mkdir()
    events_path = event_dir / "demo_events.jsonl"
    trace_path = tmp_path / "render_trace.jsonl"
    rrd_path = tmp_path / "out.rrd"
    rrd_path.write_bytes(b"rrd")
    _append_jsonl(
        events_path,
        {"event_type": "stage_annotation", "payload": {}, "demo_step": 0},
    )
    _append_jsonl(
        events_path,
        {
            "event_type": "vio_node_observed",
            "submap_id": 0,
            "payload": {"node_id": 3},
            "demo_step": 1,
        },
    )
    _append_jsonl(
        events_path,
        {
            "event_type": "odom_edge_observed",
            "submap_id": 0,
            "payload": {"edge_type": "odom", "nodeAid": 3, "nodeBid": 4},
            "demo_step": 2,
        },
    )
    _append_jsonl(
        events_path,
        {"event_type": "dmatrix_computed", "payload": {}, "demo_step": 3},
    )
    for entity_path, archetype in [
        ("/status/stage_summary", "TextDocument"),
        ("/world/submaps/0/nodes/000003", "Points3D"),
        ("/world/submaps/0/cameras/000003", "Transform3D"),
        ("/world/submaps/0/cameras/000003/image", "Pinhole"),
        ("/world/submaps/0/cameras/000003/image", "EncodedImage"),
        ("/evidence/current_keyframe_image", "EncodedImage"),
        ("/world/submaps/0/edges/odom/000003_000004", "LineStrips3D"),
        ("/evidence/dmatrix", "EncodedImage"),
    ]:
        _append_jsonl(trace_path, {"entity_path": entity_path, "archetype": archetype})

    summary = validate_runtime_rerun(event_dir, trace_path, rrd_path)

    assert summary["rrd_non_empty"] is True
    assert summary["node_camera_transform_coverage"] == "1 / 1"
    assert summary["node_pinhole_coverage"] == "1 / 1"
    assert summary["node_camera_image_coverage"] == "1 / 1"
    assert summary["odom_edges_rendered"] == "1 / 1"
    assert summary["stage_annotations_rendered"] == "1 / 1"
    assert summary["dmatrix_rendered"] is True
    assert summary["passed"] is True


def test_validate_runtime_rerun_fails_when_camera_is_missing(tmp_path: Path) -> None:
    event_dir = tmp_path / "rerun_viz"
    event_dir.mkdir()
    events_path = event_dir / "demo_events.jsonl"
    trace_path = tmp_path / "render_trace.jsonl"
    rrd_path = tmp_path / "out.rrd"
    rrd_path.write_bytes(b"rrd")
    _append_jsonl(
        events_path,
        {
            "event_type": "vio_node_observed",
            "submap_id": 0,
            "payload": {"node_id": 3},
            "demo_step": 1,
        },
    )
    _append_jsonl(trace_path, {"entity_path": "/world/submaps/0/nodes/000003", "archetype": "Points3D"})

    summary = validate_runtime_rerun(event_dir, trace_path, rrd_path)

    assert summary["passed"] is False
    assert summary["node_camera_transform_coverage"] == "0 / 1"
