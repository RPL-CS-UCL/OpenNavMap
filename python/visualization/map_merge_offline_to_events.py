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
    quat_xyzw: list[float]
    position: list[float]


@dataclass
class EdgeEntry:
    src: int
    dst: int
    weight: float


@dataclass
class IntrinsicsEntry:
    K: list[list[float]]
    img_size: list[int]


def detect_merge_dirs(results_dir: Path) -> list[Path]:
    candidates = []
    for d in results_dir.iterdir():
        if not (d.is_dir() and d.name.startswith("merge_")):
            continue
        parts = d.name.split("_")[1:]
        if all(p.isdigit() for p in parts):
            candidates.append(d)
    return sorted(candidates, key=lambda d: d.name.count("_"))


def _w2c_to_c2w(quat_wxyz: list[float], translation: list[float]) -> tuple[list[float], list[float]]:
    qw, qx, qy, qz = quat_wxyz
    quat_c2w_xyzw = [-qx, -qy, -qz, qw]
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    t = np.array(translation, dtype=np.float64)
    t_c2w = (-R.T @ t).tolist()
    return quat_c2w_xyzw, t_c2w


def load_poses(poses_file: Path) -> list[PoseEntry]:
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
    result: list[EdgeEntry] = []
    for line in edges_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        result.append(EdgeEntry(int(parts[0]), int(parts[1]), float(parts[2])))
    return result


def load_intrinsics(intrinsics_file: Path) -> dict[str, IntrinsicsEntry]:
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
    prev_set = set(prev_img_names)
    return [i for i, name in enumerate(curr_img_names) if name not in prev_set]


def compute_dmatrix(
    ref_descs: dict[str, np.ndarray], query_descs: dict[str, np.ndarray]
) -> np.ndarray:
    ref_keys = list(ref_descs.keys())
    query_keys = list(query_descs.keys())
    if not ref_keys or not query_keys:
        return np.zeros((len(ref_keys), len(query_keys)), dtype=np.float32)
    ref_mat = np.stack([ref_descs[k] for k in ref_keys])
    query_mat = np.stack([query_descs[k] for k in query_keys])
    ref_norm = ref_mat / (np.linalg.norm(ref_mat, axis=1, keepdims=True) + 1e-8)
    query_norm = query_mat / (np.linalg.norm(query_mat, axis=1, keepdims=True) + 1e-8)
    return ref_norm @ query_norm.T


def get_new_descriptors(
    prev_descs: dict[str, np.ndarray], curr_descs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    prev_keys = set(prev_descs.keys())
    return {k: v for k, v in curr_descs.items() if k not in prev_keys}


def plot_dmatrix(
    dmatrix: np.ndarray,
    output_path: Path,
    threshold: float = 0.5,
) -> Path:
    setting_font(fontsize=13, titlesize=13, legend_fontsize=13, font_family="Palatino")
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.serif"] = ["DejaVu Serif"]
    palette = acquire_color_palette()
    markers = acquire_marker()
    green = palette[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dmatrix, cmap="Greys", aspect="auto")
    im.set_clim(0.0, 1.0)
    if dmatrix.size > 0:
        ref_idx, query_idx = np.where(dmatrix >= threshold)
        if len(query_idx) > 0:
            ax.scatter(query_idx, ref_idx, c=[green], s=25, alpha=1.0, marker=markers[0])
    ax.set_xlabel("Query Index", fontsize=13)
    ax.set_ylabel("Reference Index", fontsize=13)
    ax.set_title("Difference Matrix", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_merge_data(merge_dir: Path) -> dict:
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
    entry = intrinsics.get(img_name)
    if entry is None:
        return None, None
    return entry.K, entry.img_size


def _build_map_committed_nodes(
    poses: list[PoseEntry],
    intrinsics: dict[str, IntrinsicsEntry],
    seq_dir: Path,
) -> list[dict]:
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
    result: dict[str, list[list[int]]] = {}
    for edge_type, edge_list in edges.items():
        result[edge_type] = [[e.src, e.dst] for e in edge_list]
    return result


# --- Event emitters with time support ---

T_FAST = 0.33
T_PROCESS = 3.0
T_ZERO = 0.0
T_EDGE = 0.0  # edges don't advance time (appear at keyframe time via build_time_map)


def _emit_stage(
    events: list[dict], demo_step: int, time: float, merge_step: int, submap_id: int,
    title: str, subtitle: str = "", stage_index: int = 0, stage_total: int = 1,
    step_inc: float = T_FAST,
) -> tuple[int, float]:
    events.append({
        "demo_step": demo_step,
        "time": time,
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
    return demo_step + 1, time + step_inc


def _emit_vio_node(
    events: list[dict], demo_step: int, time: float, merge_step: int, submap_id: int,
    keyframe_id: int, pose: PoseEntry,
    intrinsics: dict[str, IntrinsicsEntry], seq_dir: Path,
    step_inc: float = T_FAST,
) -> tuple[int, float]:
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
        "time": time,
        "merge_step": merge_step,
        "submap_id": submap_id,
        "keyframe_id": keyframe_id,
        "event_type": "vio_node_observed",
        "payload": payload,
        "artifacts": {},
    })
    return demo_step + 1, time + step_inc


def _emit_edge(
    events: list[dict], demo_step: int, time: float, merge_step: int, submap_id: int,
    edge_type: str, edge: EdgeEntry, poses: list[PoseEntry],
    step_inc: float = T_EDGE,
) -> tuple[int, float]:
    pos_a = poses[edge.src].position if edge.src < len(poses) else [0, 0, 0]
    pos_b = poses[edge.dst].position if edge.dst < len(poses) else [0, 0, 0]
    events.append({
        "demo_step": demo_step,
        "time": time,
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
    return demo_step + 1, time + step_inc


def _emit_map_committed(
    events: list[dict], demo_step: int, time: float, merge_step: int, submap_id: int,
    nodes: list[dict], edges: dict[str, list[list[int]]],
    step_inc: float = T_ZERO,
) -> tuple[int, float]:
    events.append({
        "demo_step": demo_step,
        "time": time,
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
    return demo_step + 1, time + step_inc


def _emit_metric_edge(
    events: list[dict], demo_step: int, time: float, merge_step: int, submap_id: int,
    db_node_id: int, query_node_id: int,
    step_inc: float = T_FAST,
) -> tuple[int, float]:
    events.append({
        "demo_step": demo_step,
        "time": time,
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
    return demo_step + 1, time + step_inc


def find_cross_submap_edges(
    prev_edges: dict[str, list[EdgeEntry]],
    curr_edges: dict[str, list[EdgeEntry]],
) -> list[tuple[int, int, str]]:
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
    merge_dirs = detect_merge_dirs(results_dir)
    if not merge_dirs:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict] = []
    demo_step = 0
    time = 0.0
    prev_data: dict | None = None

    for merge_step, merge_dir in enumerate(merge_dirs):
        data = _load_merge_data(merge_dir)
        poses = data["poses"]
        img_names = [p.img_name for p in poses]
        intrinsics = data["intrinsics"]
        seq_dir = data["seq_dir"]

        if merge_step == 0:
            demo_step, time = _emit_stage(
                events, demo_step, time, 0, 0, "Load Reference Map",
                subtitle="Replay keyframes from reference submap.",
                stage_index=1, stage_total=8,
            )
            for i, pose in enumerate(poses):
                demo_step, time = _emit_vio_node(
                    events, demo_step, time, 0, 0, i, pose, intrinsics, seq_dir
                )
            for edge_type in ("odom", "covis", "trav"):
                for edge in data["edges"][edge_type]:
                    demo_step, time = _emit_edge(
                        events, demo_step, time, 0, 0, edge_type, edge, poses
                    )
            nodes = _build_map_committed_nodes(poses, intrinsics, seq_dir)
            edges = _build_map_committed_edges(data["edges"])
            demo_step, time = _emit_map_committed(events, demo_step, time, 0, 0, nodes, edges)
        else:
            new_submap_name = merge_dir.name.split("_")[-1]
            submap_id = int(new_submap_name)

            demo_step, time = _emit_stage(
                events, demo_step, time, merge_step, submap_id,
                f"Load Submap {new_submap_name}",
                subtitle="Replay keyframes and odom/covis/trav graph edges for the query submap.",
                stage_index=2, stage_total=8,
            )

            raw_data = None
            if raw_data_dir is not None:
                raw_dir = raw_data_dir / new_submap_name
                if raw_dir.exists():
                    raw_data = _load_merge_data(raw_dir)
                    for i, raw_pose in enumerate(raw_data["poses"]):
                        demo_step, time = _emit_vio_node(
                            events, demo_step, time, merge_step, submap_id,
                            i, raw_pose, raw_data["intrinsics"], raw_data["seq_dir"]
                        )
                    for edge_type in ("odom", "covis", "trav"):
                        for edge in raw_data["edges"][edge_type]:
                            demo_step, time = _emit_edge(
                                events, demo_step, time, merge_step, submap_id,
                                edge_type, edge, raw_data["poses"]
                            )

            # Stage 3: Conduct Visual Localization and Create Loop Factors
            demo_step, time = _emit_stage(
                events, demo_step, time, merge_step, submap_id,
                "Conduct Visual Localization and Create Loop Factors",
                subtitle=f"Match query submap {new_submap_name} keyframes to reference map.",
                stage_index=3, stage_total=8,
                step_inc=T_PROCESS,
            )

            # Green cross-submap edges (loop factors)
            if prev_data and raw_data is not None:
                cross_edges = find_cross_submap_edges(prev_data["edges"], data["edges"])
                prev_pose_count = len(prev_data["poses"])
                for ref_node, query_node, _et in cross_edges:
                    query_local = query_node - prev_pose_count
                    if 0 <= query_local < len(raw_data["poses"]):
                        demo_step, time = _emit_metric_edge(
                            events, demo_step, time, merge_step, submap_id,
                            ref_node, query_local
                        )

            demo_step, time = _emit_stage(
                events, demo_step, time, merge_step, submap_id,
                f"Pose Graph Optimization: Reference Map-Submap {new_submap_name}",
                subtitle="Optimize the merged pose graph.",
                stage_index=7, stage_total=8,
                step_inc=T_PROCESS,
            )

            demo_step, time = _emit_stage(
                events, demo_step, time, merge_step, submap_id,
                "Finish Map Merging",
                subtitle="Merge the optimized query submap into the reference map and update graph edges.",
                stage_index=8, stage_total=8,
                step_inc=T_PROCESS,
            )

            nodes = _build_map_committed_nodes(poses, intrinsics, seq_dir)
            edges = _build_map_committed_edges(data["edges"])
            demo_step, time = _emit_map_committed(events, demo_step, time, merge_step, submap_id, nodes, edges)

        prev_data = data

    return events


def write_events(events: list[dict], event_dir: Path) -> None:
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
    parser.add_argument("--image-scale", type=float, default=1.0,
                        help="Image resize factor (e.g., 0.33 for 1/3 resolution)")
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
        MapMergeRuntimeRerunRenderer(
            event_dir, image_scale=args.image_scale,
        ).write(args.rerun_output)
        print(f"Rendered .rrd to {args.rerun_output}")


if __name__ == "__main__":
    main()
