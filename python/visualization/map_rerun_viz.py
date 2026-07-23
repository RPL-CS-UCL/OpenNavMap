#! /usr/bin/env python
"""Rerun (.rrd) visualization for an OpenNavMap map session.

Keyframes are logged as ``rr.Transform3D`` (pose axes) on a ``time`` timeline
using each node's timestamp; graph edges appear at their later endpoint's
timestamp (so a time-scrubbed replay shows the map being built). The L4 object
graph is drawn as oriented bounding boxes. Saved as .rrd (open with `rerun x.rrd`).

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

import numpy as np
import rerun as rr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # opennavmap/python

_EDGE_COLORS = {"odom": [31, 119, 180], "covis": [44, 160, 44], "trav": [255, 127, 14]}


def _node_time(node, fallback) -> float:
    return float(getattr(node, "time", fallback))


def visualize_map(manager, out_rrd: str, app_id: str = "opennavmap", axis_length: float = 0.5) -> str:
    """Log the map (keyframe poses + graph edges + object OBBs) to a Rerun .rrd."""
    rr.init(app_id)

    pos_graph = manager.odom or manager.covis
    times = {}
    if pos_graph is not None:
        # keyframes as pose transforms on the timeline (sorted by timestamp)
        for nid, node in pos_graph.nodes.items():
            times[nid] = _node_time(node, nid)
        for nid in sorted(pos_graph.nodes, key=lambda i: times[i]):
            node = pos_graph.get_node(nid)
            rr.set_time_seconds("time", times[nid])
            rr.log(
                f"world/keyframes/{nid}",
                rr.Transform3D(
                    translation=np.asarray(node.trans, float).reshape(3),
                    rotation=rr.Quaternion(xyzw=np.asarray(node.quat, float).reshape(4)),
                    axis_length=axis_length,
                ),
            )

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

    object_graph = manager.graphs.get("object")
    if object_graph is not None and object_graph.get_num_node() > 0:
        from scipy.spatial.transform import Rotation
        rr.set_time_seconds("time", max(times.values()) if times else 0.0)
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
