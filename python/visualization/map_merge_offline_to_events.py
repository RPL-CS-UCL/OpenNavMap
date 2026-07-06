from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
import numpy as np

import argparse
import json
from typing import Optional, Sequence


@dataclass
class PoseEntry:
    img_name: str
    quat_xyzw: list[float]  # [qx, qy, qz, qw]
    position: list[float]   # [tx, ty, tz]


@dataclass
class EdgeEntry:
    src: int
    dst: int
    weight: float


@dataclass
class IntrinsicsEntry:
    K: list[list[float]]  # 3x3
    img_size: list[int]   # [w, h]


def detect_merge_dirs(results_dir: Path) -> list[Path]:
    """Detect merge_* subdirectories in results_dir, sorted by merge order.

    Sorting key: number of underscore-separated parts (merge_0=1, merge_0_1=2, ...).
    Files (like merge_finalmap) are excluded.
    """
    candidates = [
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("merge_")
    ]
    return sorted(candidates, key=lambda d: d.name.count("_"))


def _w2c_to_c2w(quat_wxyz: list[float], translation: list[float]) -> tuple[list[float], list[float]]:
    """Convert W2C [qw,qx,qy,qz,tx,ty,tz] to C2W [qx,qy,qz,qw,cx,cy,cz].

    Matches online convert_pose_inv in utils_geom.py:
    - C2W quaternion = conjugate of W2C quaternion (inverse for unit quaternion)
    - C2W translation = -R_w2c^T @ t_w2c
    """
    qw, qx, qy, qz = quat_wxyz
    # C2W quaternion in xyzw = conjugate: [-qx, -qy, -qz, qw]
    quat_c2w_xyzw = [-qx, -qy, -qz, qw]
    # Build W2C rotation matrix
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    t = np.array(translation, dtype=np.float64)
    # C2W translation = -R^T @ t
    t_c2w = (-R.T @ t).tolist()
    return quat_c2w_xyzw, t_c2w


def load_poses(poses_file: Path) -> list[PoseEntry]:
    """Parse poses.txt: 'img_name qw qx qy qz tx ty tz' per line (W2C, wxyz).

    Converts W2C poses to C2W (camera-to-world) for the renderer, matching
    online convert_pose_inv in utils_geom.py.
    """
    result: list[PoseEntry] = []
    for line in poses_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        img_name = parts[0]
        qw, qx, qy, qz = (float(x) for x in parts[1:5])
        tx, ty, tz = (float(x) for x in parts[5:8])
        quat_xyzw, position = _w2c_to_c2w([qw, qx, qy, qz], [tx, ty, tz])
        result.append(PoseEntry(img_name, quat_xyzw, position))
    return result


def load_edges(edges_file: Path) -> list[EdgeEntry]:
    """Parse edges_*.txt: 'src dst weight' per line."""
    result: list[EdgeEntry] = []
    for line in edges_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        result.append(EdgeEntry(int(parts[0]), int(parts[1]), float(parts[2])))
    return result


def load_intrinsics(intrinsics_file: Path) -> dict[str, IntrinsicsEntry]:
    """Parse intrinsics.txt: 'img_name fx fy cx cy w h' per line."""
    result: dict[str, IntrinsicsEntry] = {}
    for line in intrinsics_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        img_name = parts[0]
        fx, fy, cx, cy = (float(x) for x in parts[1:5])
        w, h = int(parts[5]), int(parts[6])
        K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        result[img_name] = IntrinsicsEntry(K, [w, h])
    return result


def identify_new_nodes(
    prev_img_names: list[str], curr_img_names: list[str]
) -> list[int]:
    """Return indices of curr_img_names whose image name is not in prev_img_names."""
    prev_set = set(prev_img_names)
    return [i for i, name in enumerate(curr_img_names) if name not in prev_set]


def _load_merge_data(merge_dir: Path) -> dict:
    """Load all data files from a merge directory."""
    return {
        "poses": load_poses(merge_dir / "poses.txt"),
        "edges": {
            "odom": load_edges(merge_dir / "edges_odom.txt"),
            "covis": load_edges(merge_dir / "edges_covis.txt"),
            "trav": load_edges(merge_dir / "edges_trav.txt"),
        },
        "intrinsics": load_intrinsics(merge_dir / "intrinsics.txt"),
        "seq_dir": merge_dir / "seq",
    }


def _make_k_and_size(
    img_name: str, intrinsics: dict[str, IntrinsicsEntry]
) -> tuple[list[list[float]] | None, list[int] | None]:
    """Look up intrinsics for an image; return (K, img_size) or (None, None)."""
    entry = intrinsics.get(img_name)
    if entry is None:
        return None, None
    return entry.K, entry.img_size


def _build_map_committed_nodes(
    poses: list[PoseEntry],
    intrinsics: dict[str, IntrinsicsEntry],
    seq_dir: Path,
) -> list[dict]:
    """Build nodes list for map_committed event from all poses."""
    nodes = []
    for i, pose in enumerate(poses):
        K, img_size = _make_k_and_size(pose.img_name, intrinsics)
        img_path = seq_dir / Path(pose.img_name).name
        node = {
            "node_id": i,
            "position": pose.position,
            "quat_xyzw": pose.quat_xyzw,
        }
        if K is not None:
            node["raw_K"] = K
            node["raw_img_size"] = img_size
        if img_path.exists():
            node["rgb_img_path"] = str(img_path)
        nodes.append(node)
    return nodes


def _build_map_committed_edges(edges: dict[str, list[EdgeEntry]]) -> dict[str, list[list[int]]]:
    """Build edges dict for map_committed event."""
    result: dict[str, list[list[int]]] = {}
    for edge_type, edge_list in edges.items():
        result[edge_type] = [[e.src, e.dst] for e in edge_list]
    return result


def _emit_stage(
    events: list[dict], demo_step: int, merge_step: int, submap_id: int, title: str,
    subtitle: str = "", stage_index: int = 0, stage_total: int = 1,
) -> int:
    events.append({
        "demo_step": demo_step,
        "merge_step": merge_step,
        "submap_id": submap_id,
        "keyframe_id": None,
        "event_type": "stage_annotation",
        "payload": {
            "title": title,
            "subtitle": subtitle,
            "display_text": f"Stage {stage_index}/{stage_total}\n{title}\n{subtitle}",
            "stage_index": stage_index,
            "stage_total": stage_total,
        },
        "artifacts": {},
    })
    return demo_step + 1


def _emit_vio_node(
    events: list[dict], demo_step: int, merge_step: int, submap_id: int,
    keyframe_id: int, pose: PoseEntry,
    intrinsics: dict[str, IntrinsicsEntry], seq_dir: Path,
) -> int:
    K, img_size = _make_k_and_size(pose.img_name, intrinsics)
    img_path = seq_dir / Path(pose.img_name).name
    payload = {
        "node_id": keyframe_id,
        "position": pose.position,
        "quat_xyzw": pose.quat_xyzw,
    }
    if K is not None:
        payload["raw_K"] = K
        payload["raw_img_size"] = img_size
    if img_path.exists():
        payload["rgb_img_path"] = str(img_path)
    events.append({
        "demo_step": demo_step,
        "merge_step": merge_step,
        "submap_id": submap_id,
        "keyframe_id": keyframe_id,
        "event_type": "vio_node_observed",
        "payload": payload,
        "artifacts": {},
    })
    return demo_step + 1


def _emit_edge(
    events: list[dict], demo_step: int, merge_step: int, submap_id: int,
    edge_type: str, edge: EdgeEntry, poses: list[PoseEntry],
) -> int:
    pos_a = poses[edge.src].position if edge.src < len(poses) else [0, 0, 0]
    pos_b = poses[edge.dst].position if edge.dst < len(poses) else [0, 0, 0]
    events.append({
        "demo_step": demo_step,
        "merge_step": merge_step,
        "submap_id": submap_id,
        "keyframe_id": max(edge.src, edge.dst),
        "event_type": f"{edge_type}_edge_observed",
        "payload": {
            "edge_type": edge_type,
            "nodeAid": edge.src,
            "nodeBid": edge.dst,
            "position_a": pos_a,
            "position_b": pos_b,
            "weight": edge.weight,
        },
        "artifacts": {},
    })
    return demo_step + 1


def _emit_map_committed(
    events: list[dict], demo_step: int, merge_step: int, submap_id: int,
    nodes: list[dict], edges: dict[str, list[list[int]]],
) -> int:
    events.append({
        "demo_step": demo_step,
        "merge_step": merge_step,
        "submap_id": submap_id,
        "keyframe_id": None,
        "event_type": "map_committed",
        "payload": {
            "nodes": nodes,
            "edges": edges,
            "num_final_covis_nodes": len(nodes),
        },
        "artifacts": {},
    })
    return demo_step + 1


def _emit_metric_edge(
    events: list[dict], demo_step: int, merge_step: int, submap_id: int,
    db_node_id: int, query_node_id: int,
) -> int:
    """Emit a green cross-submap edge (metric_edge_added).

    submap_id = raw submap's ID so renderer looks up pos_b from
    _node_positions[(submap_id, query_node_id)] populated by vio_node_observed.
    pos_a uses _node_positions[(0, db_node_id)] from previous map_committed.
    """
    events.append({
        "demo_step": demo_step,
        "merge_step": merge_step,
        "submap_id": submap_id,
        "keyframe_id": None,
        "event_type": "metric_edge_added",
        "payload": {
            "db_node_id": db_node_id,
            "query_node_id": query_node_id,
            "conf": 1.0,
        },
        "artifacts": {},
    })
    return demo_step + 1


def find_cross_submap_edges(
    prev_edges: dict[str, list[EdgeEntry]],
    curr_edges: dict[str, list[EdgeEntry]],
) -> list[tuple[int, int, str]]:
    """Find new non-consecutive edges (cross-submap) between merge steps.

    Returns list of (ref_node_id, query_node_id, edge_type).
    Non-consecutive = |src - dst| > 1 (cross-submap loop closures).
    """
    prev_set: set[tuple[int, int]] = set()
    for et in ("odom", "covis", "trav"):
        for e in prev_edges[et]:
            prev_set.add((e.src, e.dst))

    result: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for et in ("odom", "covis", "trav"):
        for e in curr_edges[et]:
            if (e.src, e.dst) not in prev_set and (e.src, e.dst) not in seen:
                if abs(e.src - e.dst) > 1:
                    ref_node = min(e.src, e.dst)
                    query_node = max(e.src, e.dst)
                    result.append((ref_node, query_node, et))
                    seen.add((e.src, e.dst))
    return result


def generate_events(
    results_dir: Path, output_dir: Path,
    raw_data_dir: Path | None = None,
) -> list[dict]:
    """Generate demo_events.jsonl-compatible events from offline merge data.

    Reads merge_* directories in order, produces events for the existing
    MapMergeRuntimeRerunRenderer. If raw_data_dir is provided, also plots
    raw submap keyframes (in local coordinate frame) before the merged result.
    """
    merge_dirs = detect_merge_dirs(results_dir)
    if not merge_dirs:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict] = []
    demo_step = 0
    prev_data: dict | None = None

    for merge_step, merge_dir in enumerate(merge_dirs):
        data = _load_merge_data(merge_dir)
        poses = data["poses"]
        img_names = [p.img_name for p in poses]
        intrinsics = data["intrinsics"]
        seq_dir = data["seq_dir"]

        if merge_step == 0:
            demo_step = _emit_stage(
                events, demo_step, 0, 0, "Load Reference Map",
                subtitle="Replay keyframes from reference submap.",
                stage_index=1, stage_total=8,
            )
            for i, pose in enumerate(poses):
                demo_step = _emit_vio_node(
                    events, demo_step, 0, 0, i, pose, intrinsics, seq_dir
                )
            for edge_type in ("odom", "covis", "trav"):
                for edge in data["edges"][edge_type]:
                    demo_step = _emit_edge(
                        events, demo_step, 0, 0, edge_type, edge, poses
                    )
            nodes = _build_map_committed_nodes(poses, intrinsics, seq_dir)
            edges = _build_map_committed_edges(data["edges"])
            demo_step = _emit_map_committed(events, demo_step, 0, 0, nodes, edges)
        else:
            new_submap_name = merge_dir.name.split("_")[-1]
            submap_id = int(new_submap_name)

            # Stage 2: Load Submap (match online subtitle)
            demo_step = _emit_stage(
                events, demo_step, merge_step, submap_id,
                f"Load Submap {new_submap_name}",
                subtitle="Replay keyframes and odom/covis/trav graph edges for the query submap.",
                stage_index=2, stage_total=8,
            )

            # Plot RAW submap keyframes + ALL raw edges (local frame, RAW poses)
            raw_data = None
            if raw_data_dir is not None:
                raw_dir = raw_data_dir / new_submap_name
                if raw_dir.exists():
                    raw_data = _load_merge_data(raw_dir)
                    for i, raw_pose in enumerate(raw_data["poses"]):
                        demo_step = _emit_vio_node(
                            events, demo_step, merge_step, submap_id,
                            i, raw_pose, raw_data["intrinsics"], raw_data["seq_dir"]
                        )
                    # ALL raw submap edges (odom/covis/trav), RAW poses
                    for edge_type in ("odom", "covis", "trav"):
                        for edge in raw_data["edges"][edge_type]:
                            demo_step = _emit_edge(
                                events, demo_step, merge_step, submap_id,
                                edge_type, edge, raw_data["poses"]
                            )

            # Green cross-submap edges (BEFORE PGO, connecting final map to raw submap)
            if prev_data and raw_data is not None:
                cross_edges = find_cross_submap_edges(prev_data["edges"], data["edges"])
                prev_pose_count = len(prev_data["poses"])
                for ref_node, query_node, _et in cross_edges:
                    query_local = query_node - prev_pose_count
                    if 0 <= query_local < len(raw_data["poses"]):
                        demo_step = _emit_metric_edge(
                            events, demo_step, merge_step, submap_id,
                            ref_node, query_local
                        )

            # Stage 7: Pose Graph Optimization (match online title)
            demo_step = _emit_stage(
                events, demo_step, merge_step, submap_id,
                f"Pose Graph Optimization: Reference Map-Submap {new_submap_name}",
                subtitle="Optimize the merged pose graph.",
                stage_index=7, stage_total=8,
            )

            # Stage 8: Finish Map Merging (per merge step, match online)
            demo_step = _emit_stage(
                events, demo_step, merge_step, submap_id,
                "Finish Map Merging",
                subtitle="Merge the optimized query submap into the reference map and update graph edges.",
                stage_index=8, stage_total=8,
            )

            # map_committed (Clears cameras/edges, shows merged cameras with PGO poses)
            nodes = _build_map_committed_nodes(poses, intrinsics, seq_dir)
            edges = _build_map_committed_edges(data["edges"])
            demo_step = _emit_map_committed(events, demo_step, merge_step, submap_id, nodes, edges)

        prev_data = data

    return events


def write_events(events: list[dict], event_dir: Path) -> None:
    """Write events as demo_events.jsonl in event_dir."""
    event_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = event_dir / "demo_events.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert offline map-merge results to Rerun .rrd visualization."
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Path to s00000_results_in_* directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for rerun_viz/")
    parser.add_argument("--raw-data-dir", type=Path, default=None,
                        help="Path to s00000_aria_data_390/ directory with raw submap data")
    parser.add_argument("--render", action="store_true",
                        help="Render .rrd after generating events")
    parser.add_argument("--rerun-output", type=Path, default=None,
                        help="Path for .rrd file (requires --render)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    event_dir = args.output_dir / "rerun_viz"
    events = generate_events(args.results_dir, event_dir, args.raw_data_dir)
    write_events(events, event_dir)
    print(f"Wrote {len(events)} events to {event_dir / 'demo_events.jsonl'}")

    if args.render:
        if args.rerun_output is None:
            args.rerun_output = args.output_dir / "output.rrd"
        from visualization.map_merge_runtime_rerun_renderer import (
            MapMergeRuntimeRerunRenderer,
        )
        MapMergeRuntimeRerunRenderer(event_dir).write(args.rerun_output)
        print(f"Rendered .rrd to {args.rerun_output}")


if __name__ == "__main__":
    main()
