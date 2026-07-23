#! /usr/bin/env python
"""Rerun (.rrd) visualization for an OpenNavMap map session.

Keyframes are logged as ``rr.Transform3D`` (pose axes) + ``rr.Pinhole`` (camera
frustum) with the stored rgb (``rr.Image``) and depth (``rr.DepthImage``) shown
in the frustum, on a ``time`` timeline (each node's timestamp). Graph edges
appear at their later endpoint's timestamp; the L4 object graph is drawn as
oriented bounding boxes. Saved as .rrd (open with `rerun x.rrd`).

World frame is z-up (built by the sim_builder); camera frame is CV (x-right,
y-down, z-forward) = rerun's default Pinhole convention.

Library:  visualize_map(map_manager, "out.rrd")
CLI:      python -m visualization.map_rerun_viz --map <map_dir> --out <file>.rrd
"""
import argparse
import ctypes
import os
import sys
from pathlib import Path

# Preload this env's libstdc++ (GLIBCXX) BEFORE importing rerun/PIL.
_LIBSTDCXX = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
if os.path.exists(_LIBSTDCXX):
    try:
        ctypes.CDLL(_LIBSTDCXX, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rerun as rr  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # opennavmap/python

_EDGE_COLORS = {"odom": [31, 119, 180], "covis": [44, 160, 44], "trav": [255, 127, 14]}
_DEPTH_METER = 1000.0  # stored depth png is uint16 millimetres


def _node_time(node, fallback) -> float:
    return float(getattr(node, "time", fallback))


def _load_rgb(path: Path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def _load_depth(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _log_keyframes(manager, axis_length: float) -> None:
    covis = manager.covis
    if covis is not None and covis.get_num_node() > 0:
        root = Path(covis.map_root)
        for nid in sorted(covis.nodes, key=lambda i: _node_time(covis.get_node(i), i)):
            node = covis.get_node(nid)
            rr.set_time_seconds("time", _node_time(node, nid))
            ent = f"world/keyframes/{nid}"
            rr.log(ent, rr.Transform3D(
                translation=np.asarray(node.trans, float).reshape(3),
                rotation=rr.Quaternion(xyzw=np.asarray(node.quat, float).reshape(4)),
                axis_length=axis_length))
            try:
                w, h = int(node.img_size[0]), int(node.img_size[1])
                rr.log(f"{ent}/cam", rr.Pinhole(
                    image_from_camera=np.asarray(node.K, float).reshape(3, 3), resolution=[w, h]))
            except Exception:
                continue
            rgb = _load_rgb(root / node.rgb_img_name)
            if rgb is not None:
                rr.log(f"{ent}/cam/image", rr.Image(rgb))
            depth = _load_depth(root / node.depth_img_name)
            if depth is not None:
                rr.log(f"{ent}/cam/depth", rr.DepthImage(depth, meter=_DEPTH_METER))
        return
    # fallback: pose axes only (no covis / no images)
    graph = manager.odom
    if graph is not None:
        for nid in sorted(graph.nodes, key=lambda i: _node_time(graph.get_node(i), i)):
            node = graph.get_node(nid)
            rr.set_time_seconds("time", _node_time(node, nid))
            rr.log(f"world/keyframes/{nid}", rr.Transform3D(
                translation=np.asarray(node.trans, float).reshape(3),
                rotation=rr.Quaternion(xyzw=np.asarray(node.quat, float).reshape(4)),
                axis_length=axis_length))


def _log_edges(manager) -> None:
    for name in ("odom", "covis", "trav"):
        graph = manager.graphs.get(name)
        if graph is None or graph.get_num_node() == 0:
            continue
        pos = {i: np.asarray(nd.trans, float).reshape(3) for i, nd in graph.nodes.items()}
        tim = {i: _node_time(nd, i) for i, nd in graph.nodes.items()}
        seen, edges = set(), []
        for nid, node in graph.nodes.items():
            for neighbor, _w in node.edges.values():
                key = (min(str(nid), str(neighbor.id)), max(str(nid), str(neighbor.id)))
                if key in seen:
                    continue
                seen.add(key)
                edges.append((max(tim[nid], tim[neighbor.id]), nid, neighbor.id))
        for t, a, b in sorted(edges):
            rr.set_time_seconds("time", t)
            rr.log(f"world/edges/{name}/{a}-{b}",
                   rr.LineStrips3D([[pos[a].tolist(), pos[b].tolist()]], colors=_EDGE_COLORS[name]))


def _log_objects(manager) -> None:
    object_graph = manager.graphs.get("object")
    if object_graph is None or object_graph.get_num_node() == 0:
        return
    from scipy.spatial.transform import Rotation
    centers, half_sizes, rotations, labels = [], [], [], []
    for node in object_graph.nodes.values():
        centers.append(np.asarray(node.obb.center, float).reshape(3))
        half_sizes.append(np.asarray(node.obb.size, float).reshape(3) / 2.0)
        quat = Rotation.from_matrix(np.asarray(node.obb.R, float).reshape(3, 3)).as_quat()
        rotations.append(rr.Quaternion(xyzw=quat))
        labels.append(f"{node.label} {node.confidence:.2f}")
    rr.log("world/objects", rr.Boxes3D(centers=centers, half_sizes=half_sizes,
                                       rotations=rotations, labels=labels, colors=[214, 39, 40]))


def visualize_map(manager, out_rrd: str, app_id: str = "opennavmap", axis_length: float = 0.5) -> str:
    """Log the map (keyframe pose+frustum+images, edges, object OBBs) to a Rerun .rrd."""
    rr.init(app_id)
    # world frame axes (z-up), longer than keyframe axes (2x)
    rr.log("world", rr.Transform3D(axis_length=2.0 * axis_length), static=True)
    _log_keyframes(manager, axis_length)
    _log_edges(manager)
    _log_objects(manager)
    out_path = Path(out_rrd)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_path))
    return str(out_path)


def _load_map(map_dir: Path):
    """Best-effort load of a stored map (covis with rgb/depth for frustum viz)."""
    from map_manager import MapManager
    manager = MapManager(map_dir)
    configs = {"odom": {}, "trav": {}}
    if (map_dir / "intrinsics.txt").exists():
        configs["covis"] = {"resize": None, "depth_scale": 1.0,
                            "load_rgb": False, "load_depth": False, "normalized": False}
    if (map_dir / "objects.json").exists():
        configs["object"] = {}
    manager.load_graphs(configs)
    return manager


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True, help="stored map directory")
    parser.add_argument("--out", type=Path, required=True, help="output .rrd path")
    args = parser.parse_args()
    out = visualize_map(_load_map(args.map), str(args.out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
