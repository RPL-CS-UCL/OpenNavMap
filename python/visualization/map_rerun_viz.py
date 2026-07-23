#! /usr/bin/env python
"""Rerun (.rrd) visualization for an OpenNavMap map session.

Visualizes the keyframe graph (covis/odom/trav trajectory + edges) and the L4
object graph (oriented bounding boxes with label/confidence) into a single Rerun
recording saved as .rrd (open with `rerun <file>.rrd`).

Usage as a library:
    from map_rerun_viz import visualize_map
    visualize_map(map_manager, "outputs/viz/map.rrd")

Usage as a standalone program (load a stored map, then visualize):
    python -m visualization.map_rerun_viz --map <map_dir> --out <file>.rrd
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

import numpy as np
import rerun as rr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # opennavmap/python

_EDGE_COLORS = {"odom": [31, 119, 180], "covis": [44, 160, 44], "trav": [255, 127, 14]}


def _positions(graph):
    return {nid: np.asarray(node.trans, float).reshape(3) for nid, node in graph.nodes.items()}


def _edge_strips(graph, pos):
    strips, seen = [], set()
    for nid, node in graph.nodes.items():
        for neighbor, _w in node.edges.values():
            key = (min(str(nid), str(neighbor.id)), max(str(nid), str(neighbor.id)))
            if key in seen or nid not in pos or neighbor.id not in pos:
                continue
            seen.add(key)
            strips.append([pos[nid].tolist(), pos[neighbor.id].tolist()])
    return strips


def visualize_map(manager, out_rrd: str, app_id: str = "opennavmap") -> str:
    """Log the map (keyframe graph + object graph) to a Rerun .rrd file."""
    rr.init(app_id)

    pos_graph = manager.odom or manager.covis
    if pos_graph is not None and pos_graph.get_num_node() > 0:
        pos = _positions(pos_graph)
        ordered = np.array([pos[i] for i in sorted(pos, key=str)])
        rr.log("world/keyframes", rr.Points3D(ordered, radii=0.04, colors=[80, 80, 80]))
        rr.log("world/trajectory", rr.LineStrips3D([ordered.tolist()], colors=[150, 150, 150]))

    for name in ("odom", "covis", "trav"):
        graph = manager.graphs.get(name)
        if graph is None or graph.get_num_node() == 0:
            continue
        strips = _edge_strips(graph, _positions(graph))
        if strips:
            rr.log(f"world/edges/{name}", rr.LineStrips3D(strips, colors=_EDGE_COLORS[name]))

    object_graph = manager.graphs.get("object")
    if object_graph is not None and object_graph.get_num_node() > 0:
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

    out_path = Path(out_rrd)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_path))
    return str(out_path)


def _load_map(map_dir: Path):
    """Best-effort load of a stored map (odom/trav/object always; covis poses+edges)."""
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
