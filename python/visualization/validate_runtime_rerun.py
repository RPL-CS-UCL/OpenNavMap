from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple


def _load_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def _camera_group(submap_id: int) -> str:
    return "ref" if submap_id == 0 else "query"


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

    nodes = {_node_key(e) for e in events if e.get("event_type") == "vio_node_observed"}
    edges = {
        etype: {
            _edge_key(e)
            for e in events
            if e.get("event_type") == f"{etype}_edge_observed"
        }
        for etype in ("odom", "covis", "trav")
    }
    stage_count = sum(1 for e in events if e.get("event_type") == "stage_annotation")

    expected_transforms = {f"sfm/cameras/{_camera_group(sid)}/{nid}" for sid, nid in nodes}
    expected_pinholes = {f"sfm/cameras/{_camera_group(sid)}/{nid}/image" for sid, nid in nodes}
    expected_images = set(expected_pinholes)

    expected_edge_paths = {
        etype: {
            f"sfm/edges/{_camera_group(sid)}/{etype}/{na}_{nb}"
            for sid, _et, na, nb in edge_set
        }
        for etype, edge_set in edges.items()
    }

    transforms = _entity_set(trace, "Transform3D")
    pinholes = _entity_set(trace, "Pinhole")
    encoded = _entity_set(trace, "ImageEncoded")
    lines = _entity_set(trace, "LineStrips3D")
    text = _entity_set(trace, "TextDocument")

    summary: Dict[str, object] = {
        "rrd_non_empty": Path(rrd).exists() and Path(rrd).stat().st_size > 0,
        "stage_summary_rendered": "/status/stage_summary" in text if stage_count else True,
        "keyframe_transform_coverage": _coverage(transforms, expected_transforms),
        "keyframe_pinhole_coverage": _coverage(pinholes, expected_pinholes),
        "keyframe_image_coverage": _coverage(encoded, expected_images),
        "current_keyframe_image_rendered": "evidence/current_keyframe_image" in encoded,
        "odom_edges_rendered": _coverage(lines, expected_edge_paths["odom"]),
        "covis_edges_rendered": _coverage(lines, expected_edge_paths["covis"]),
        "trav_edges_rendered": _coverage(lines, expected_edge_paths["trav"]),
    }
    summary["passed"] = all(
        [
            summary["rrd_non_empty"],
            summary["stage_summary_rendered"],
            summary["keyframe_transform_coverage"] == f"{len(nodes)} / {len(nodes)}",
            summary["keyframe_pinhole_coverage"] == f"{len(nodes)} / {len(nodes)}",
            summary["keyframe_image_coverage"] == f"{len(nodes)} / {len(nodes)}",
            summary["current_keyframe_image_rendered"],
            summary["odom_edges_rendered"] == f"{len(edges['odom'])} / {len(edges['odom'])}",
            summary["covis_edges_rendered"] == f"{len(edges['covis'])} / {len(edges['covis'])}",
            summary["trav_edges_rendered"] == f"{len(edges['trav'])} / {len(edges['trav'])}",
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
