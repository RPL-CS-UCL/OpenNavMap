from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple


def _load_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def _node_key(event: dict) -> Tuple[int, int]:
    return int(event["submap_id"]), int(event["payload"]["node_id"])


def _edge_key(event: dict) -> Tuple[int, str, int, int]:
    payload = event["payload"]
    return (
        int(event["submap_id"]),
        payload["edge_type"],
        int(payload["nodeAid"]),
        int(payload["nodeBid"]),
    )


def _coverage(rendered: Set, expected: Set) -> str:
    return f"{len(rendered & expected)} / {len(expected)}"


def _entity_set(trace: Iterable[dict], archetype: Optional[str] = None) -> Set[str]:
    return {
        record["entity_path"]
        for record in trace
        if archetype is None or record.get("archetype") == archetype
    }


def validate_runtime_rerun(event_dir: Path, render_trace: Path, rrd: Path) -> Dict[str, object]:
    events = _load_jsonl(Path(event_dir) / "demo_events.jsonl")
    trace = _load_jsonl(render_trace)
    nodes = {_node_key(event) for event in events if event.get("event_type") == "vio_node_observed"}
    edges = {
        edge_type: {
            _edge_key(event)
            for event in events
            if event.get("event_type") == f"{edge_type}_edge_observed"
        }
        for edge_type in ("odom", "covis", "trav")
    }
    stage_count = sum(1 for event in events if event.get("event_type") == "stage_annotation")
    has_dmatrix_event = any(event.get("event_type") == "dmatrix_computed" for event in events)

    points = _entity_set(trace, "Points3D")
    transforms = _entity_set(trace, "Transform3D")
    pinholes = _entity_set(trace, "Pinhole")
    encoded = _entity_set(trace, "EncodedImage")
    lines = _entity_set(trace, "LineStrips3D")
    text = _entity_set(trace, "TextDocument")

    node_points = {f"/world/submaps/{sid}/nodes/{nid:06d}" for sid, nid in nodes}
    node_transforms = {f"/world/submaps/{sid}/cameras/{nid:06d}" for sid, nid in nodes}
    node_images = {f"/world/submaps/{sid}/cameras/{nid:06d}/image" for sid, nid in nodes}
    edge_paths = {
        edge_type: {
            f"/world/submaps/{sid}/edges/{edge_type}/{node_a:06d}_{node_b:06d}"
            for sid, _etype, node_a, node_b in edge_set
        }
        for edge_type, edge_set in edges.items()
    }
    summary: Dict[str, object] = {
        "rrd_non_empty": Path(rrd).exists() and Path(rrd).stat().st_size > 0,
        "node_point_coverage": _coverage(points, node_points),
        "node_camera_transform_coverage": _coverage(transforms, node_transforms),
        "node_pinhole_coverage": _coverage(pinholes, node_images),
        "node_camera_image_coverage": _coverage(encoded, node_images),
        "node_evidence_image_rendered": "/evidence/current_keyframe_image" in encoded,
        "odom_edges_rendered": _coverage(lines, edge_paths["odom"]),
        "covis_edges_rendered": _coverage(lines, edge_paths["covis"]),
        "trav_edges_rendered": _coverage(lines, edge_paths["trav"]),
        "stage_annotations_rendered": f"{1 if '/status/stage_summary' in text else 0} / {1 if stage_count else 0}",
        "stage_annotation_events": stage_count,
        "dmatrix_rendered": (not has_dmatrix_event) or "/evidence/dmatrix" in encoded,
    }
    summary["passed"] = all(
        [
            summary["rrd_non_empty"],
            summary["node_point_coverage"] == f"{len(nodes)} / {len(nodes)}",
            summary["node_camera_transform_coverage"] == f"{len(nodes)} / {len(nodes)}",
            summary["node_pinhole_coverage"] == f"{len(nodes)} / {len(nodes)}",
            summary["node_camera_image_coverage"] == f"{len(nodes)} / {len(nodes)}",
            summary["node_evidence_image_rendered"],
            summary["odom_edges_rendered"] == f"{len(edges['odom'])} / {len(edges['odom'])}",
            summary["covis_edges_rendered"] == f"{len(edges['covis'])} / {len(edges['covis'])}",
            summary["trav_edges_rendered"] == f"{len(edges['trav'])} / {len(edges['trav'])}",
            summary["stage_annotations_rendered"] == f"{1 if stage_count else 0} / {1 if stage_count else 0}",
            summary["dmatrix_rendered"],
        ]
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate runtime map-merge Rerun render coverage.")
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--render-trace", type=Path, required=True)
    parser.add_argument("--rrd", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = validate_runtime_rerun(args.event_dir, args.render_trace, args.rrd)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
