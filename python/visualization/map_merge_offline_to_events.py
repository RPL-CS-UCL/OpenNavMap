from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from utils.utils_setting_color_font import (
        setting_font,
        acquire_color_palette,
        acquire_marker,
    )
except ImportError:
    def setting_font(fontsize=13, titlesize=13, legend_fontsize=13, font_family="Palatino"):
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["DejaVu Serif"]
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["axes.titlesize"] = titlesize
        plt.rcParams["legend.fontsize"] = legend_fontsize
        plt.rcParams["text.usetex"] = False

    def acquire_color_palette():
        palette = np.zeros((60, 3), dtype=np.float32)
        palette[0] = [0, 152 / 255, 83 / 255]
        palette[1] = [228 / 255, 53 / 255, 39 / 255]
        return palette

    def acquire_marker():
        return ['o', 's', '^', 'D', 'X', '*', '+']

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


def load_poses(poses_file: Path) -> list[PoseEntry]:
    """Parse poses.txt: 'img_name qx qy qz qw tx ty tz' per line."""
    result: list[PoseEntry] = []
    for line in poses_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        img_name = parts[0]
        qx, qy, qz, qw = (float(x) for x in parts[1:5])
        tx, ty, tz = (float(x) for x in parts[5:8])
        result.append(PoseEntry(img_name, [qx, qy, qz, qw], [tx, ty, tz]))
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


def load_descriptors(desc_file: Path) -> dict[str, np.ndarray]:
    """Parse database_descriptors.txt: 'img_name d1 d2 ... d256' per line."""
    result: dict[str, np.ndarray] = {}
    for line in desc_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_name = parts[0]
        desc = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        result[img_name] = desc
    return result


def identify_new_nodes(
    prev_img_names: list[str], curr_img_names: list[str]
) -> list[int]:
    """Return indices of curr_img_names whose image name is not in prev_img_names."""
    prev_set = set(prev_img_names)
    return [i for i, name in enumerate(curr_img_names) if name not in prev_set]


def compute_dmatrix(
    ref_descs: dict[str, np.ndarray], query_descs: dict[str, np.ndarray]
) -> np.ndarray:
    """Compute cosine similarity matrix between reference and query descriptors.

    Returns matrix of shape (len(ref), len(query)).
    """
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
    """Return descriptors in curr_descs whose key is not in prev_descs."""
    prev_keys = set(prev_descs.keys())
    return {k: v for k, v in curr_descs.items() if k not in prev_keys}


def plot_dmatrix(
    dmatrix: np.ndarray,
    output_path: Path,
    threshold: float = 0.5,
) -> Path:
    """Plot D-matrix in online paper style: Greys imshow + green scatter overlay.

    Matches _plot_runtime_dmatrix_panels styling in map_merge_pipeline.py:
    - setting_font(fontsize=13, titlesize=13, font_family="Palatino")
    - cmap="Greys" background, clim=[0,1]
    - green scatter (s=40, marker='o') for pairs above threshold
    - dpi=300, bbox_inches="tight", figsize=(6,5)
    """
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
        query_idx, ref_idx = np.where(dmatrix >= threshold)
        if len(query_idx) > 0:
            ax.scatter(
                query_idx,
                ref_idx,
                c=[green],
                s=40,
                alpha=1.0,
                marker=markers[0],
            )

    ax.set_xlabel("Query Index", fontsize=13)
    ax.set_ylabel("Reference Index", fontsize=13)
    ax.set_title("Difference Matrix", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
        "descriptors": (
            load_descriptors(merge_dir / "database_descriptors.txt")
            if (merge_dir / "database_descriptors.txt").exists()
            else {}
        ),
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
    events: list[dict], demo_step: int, merge_step: int, title: str,
    subtitle: str = "", stage_index: int = 0, stage_total: int = 1,
) -> int:
    events.append({
        "demo_step": demo_step,
        "merge_step": merge_step,
        "submap_id": merge_step,
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


def _emit_dmatrix(
    events: list[dict], demo_step: int, merge_step: int,
    dmatrix: np.ndarray, png_path: Path,
) -> int:
    events.append({
        "demo_step": demo_step,
        "merge_step": merge_step,
        "submap_id": merge_step,
        "keyframe_id": None,
        "event_type": "dmatrix_computed",
        "payload": {"shape": list(dmatrix.shape)},
        "artifacts": {"dmatrix_png": str(png_path)},
    })
    return demo_step + 1


def _emit_map_committed(
    events: list[dict], demo_step: int, merge_step: int,
    nodes: list[dict], edges: dict[str, list[list[int]]],
) -> int:
    events.append({
        "demo_step": demo_step,
        "merge_step": merge_step,
        "submap_id": merge_step,
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


def generate_events(results_dir: Path, output_dir: Path) -> list[dict]:
    """Generate demo_events.jsonl-compatible events from offline merge data.

    Reads merge_* directories in order, produces events for the existing
    MapMergeRuntimeRerunRenderer.
    """
    merge_dirs = detect_merge_dirs(results_dir)
    if not merge_dirs:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

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
                events, demo_step, 0, "Load Reference Map",
                subtitle="Replay keyframes from reference submap.",
                stage_index=1, stage_total=3,
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
            demo_step = _emit_map_committed(events, demo_step, 0, nodes, edges)
        else:
            submap_id = merge_step
            new_submap_name = merge_dir.name.split("_")[-1]
            demo_step = _emit_stage(
                events, demo_step, merge_step,
                f"Load Submap {new_submap_name}",
                subtitle=f"Add submap {new_submap_name} to merged map.",
                stage_index=1, stage_total=4,
            )

            prev_img_names = [p.img_name for p in prev_data["poses"]] if prev_data else []
            new_indices = identify_new_nodes(prev_img_names, img_names)
            for idx in new_indices:
                demo_step = _emit_vio_node(
                    events, demo_step, merge_step, submap_id,
                    idx, poses[idx], intrinsics, seq_dir
                )

            prev_edge_set: set[tuple[int, int]] = set()
            if prev_data:
                for et in ("odom", "covis", "trav"):
                    for e in prev_data["edges"][et]:
                        prev_edge_set.add((e.src, e.dst))
            for edge_type in ("odom", "covis", "trav"):
                for edge in data["edges"][edge_type]:
                    if (edge.src, edge.dst) not in prev_edge_set:
                        demo_step = _emit_edge(
                            events, demo_step, merge_step, submap_id,
                            edge_type, edge, poses
                        )

            if prev_data and prev_data["descriptors"] and data["descriptors"]:
                query_descs = get_new_descriptors(
                    prev_data["descriptors"], data["descriptors"]
                )
                if query_descs:
                    dmatrix = compute_dmatrix(prev_data["descriptors"], query_descs)
                    png_path = artifacts_dir / f"dmatrix_merge_{merge_step}.png"
                    plot_dmatrix(dmatrix, png_path, threshold=0.5)
                    demo_step = _emit_stage(
                        events, demo_step, merge_step,
                        f"Compute Difference Matrix - Reference Map-Submap {new_submap_name}",
                        subtitle=f"Cosine similarity {dmatrix.shape[0]}x{dmatrix.shape[1]}.",
                        stage_index=2, stage_total=4,
                    )
                    demo_step = _emit_dmatrix(
                        events, demo_step, merge_step, dmatrix, png_path
                    )

            demo_step = _emit_stage(
                events, demo_step, merge_step,
                f"Pose Graph Optimization: Reference Map-Submap {new_submap_name}",
                subtitle="PGO refines all poses in the merged map.",
                stage_index=3, stage_total=4,
            )

            nodes = _build_map_committed_nodes(poses, intrinsics, seq_dir)
            edges = _build_map_committed_edges(data["edges"])
            demo_step = _emit_map_committed(events, demo_step, merge_step, nodes, edges)

        prev_data = data

    demo_step = _emit_stage(
        events, demo_step, len(merge_dirs) - 1,
        "Finish Map Merging",
        subtitle="All submaps merged into final map.",
        stage_index=4, stage_total=4,
    )

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
    parser.add_argument("--render", action="store_true",
                        help="Render .rrd after generating events")
    parser.add_argument("--rerun-output", type=Path, default=None,
                        help="Path for .rrd file (requires --render)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    event_dir = args.output_dir / "rerun_viz"
    events = generate_events(args.results_dir, event_dir)
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
