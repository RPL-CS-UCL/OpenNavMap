import json
from pathlib import Path

from visualization.validate_runtime_rerun import validate_runtime_rerun


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def test_validate_reports_complete_coverage(tmp_path: Path) -> None:
    event_dir = tmp_path / "rerun_viz"
    event_dir.mkdir()
    events_path = event_dir / "demo_events.jsonl"
    trace_path = tmp_path / "render_trace.jsonl"
    rrd_path = tmp_path / "out.rrd"
    rrd_path.write_bytes(b"rrd")

    _append_jsonl(events_path, {"event_type": "stage_annotation", "payload": {}, "demo_step": 0})
    _append_jsonl(events_path, {
        "event_type": "vio_node_observed",
        "submap_id": 0,
        "payload": {"node_id": 3},
        "demo_step": 1,
    })
    _append_jsonl(events_path, {
        "event_type": "vio_node_observed",
        "submap_id": 1,
        "payload": {"node_id": 5},
        "demo_step": 2,
    })
    _append_jsonl(events_path, {
        "event_type": "odom_edge_observed",
        "submap_id": 0,
        "payload": {"edge_type": "odom", "nodeAid": 3, "nodeBid": 4},
        "demo_step": 3,
    })
    _append_jsonl(events_path, {
        "event_type": "dmatrix_computed",
        "submap_id": 1,
        "payload": {"shape": [66, 45]},
        "artifacts": {"dmatrix_png": "artifacts/dmatrix.png"},
        "demo_step": 4,
    })
    _append_jsonl(events_path, {
        "event_type": "metric_edge_added",
        "submap_id": 1,
        "payload": {"db_node_id": 3, "query_node_id": 5, "conf": 31.0},
        "demo_step": 5,
    })
    _append_jsonl(events_path, {
        "event_type": "map_committed",
        "submap_id": 1,
        "payload": {
            "num_final_covis_nodes": 2,
            "nodes": [{"node_id": 3, "position": [1, 2, 3]}, {"node_id": 5, "position": [4, 5, 6]}],
            "edges": {"odom": [[3, 5]]},
        },
        "demo_step": 6,
    })

    for entity_path, archetype in [
        ("/status/stage_summary", "TextDocument"),
        ("cameras/ref/3", "Transform3D"),
        ("cameras/ref/3/image", "Pinhole"),
        ("cameras/ref/3/image", "ImageEncoded"),
        ("evidence/current_keyframe_image", "ImageEncoded"),
        ("cameras/query/5", "Transform3D"),
        ("cameras/query/5/image", "Pinhole"),
        ("cameras/query/5/image", "ImageEncoded"),
        ("edges/ref/odom/3_4", "LineStrips3D"),
        ("evidence/dmatrix", "ImageEncoded"),
        ("edges/metric/3_5", "LineStrips3D"),
        ("final_map/nodes", "Points3D"),
        ("final_map/edges/odom", "LineStrips3D"),
    ]:
        _append_jsonl(trace_path, {"entity_path": entity_path, "archetype": archetype})

    summary = validate_runtime_rerun(event_dir, trace_path, rrd_path)

    assert summary["rrd_non_empty"] is True
    assert summary["stage_summary_rendered"] is True
    assert summary["keyframe_transform_coverage"] == "2 / 2"
    assert summary["keyframe_pinhole_coverage"] == "2 / 2"
    assert summary["keyframe_image_coverage"] == "2 / 2"
    assert summary["current_keyframe_image_rendered"] is True
    assert summary["odom_edges_rendered"] == "1 / 1"
    assert summary["dmatrix_rendered"] is True
    assert summary["metric_edges_rendered"] == "1 / 1"
    assert summary["final_map_nodes_rendered"] is True
    assert summary["passed"] is True


def test_validate_fails_when_camera_missing(tmp_path: Path) -> None:
    event_dir = tmp_path / "rerun_viz"
    event_dir.mkdir()
    events_path = event_dir / "demo_events.jsonl"
    trace_path = tmp_path / "render_trace.jsonl"
    rrd_path = tmp_path / "out.rrd"
    rrd_path.write_bytes(b"rrd")

    _append_jsonl(events_path, {
        "event_type": "vio_node_observed",
        "submap_id": 0,
        "payload": {"node_id": 3},
        "demo_step": 1,
    })
    _append_jsonl(trace_path, {"entity_path": "cameras/ref/3", "archetype": "Transform3D"})

    summary = validate_runtime_rerun(event_dir, trace_path, rrd_path)

    assert summary["passed"] is False
    assert summary["keyframe_transform_coverage"] == "1 / 1"
    assert summary["keyframe_pinhole_coverage"] == "0 / 1"
