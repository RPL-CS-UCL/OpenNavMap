#! /usr/bin/env python
"""Rerun (.rrd) visualization for an OpenNavMap map session.

Follows the litevloc ``utils_rerun`` conventions (reusing its functions where
possible):
- ``world/axes``              : XYZ world-frame axes (log_world_frame_axes, red/green/blue)
- ``map/nodes/{id}``          : per-keyframe Transform3D (timeless)
- ``map/nodes/{id}/camera``   : Pinhole camera frustum + RGB image (color from seq/)
- ``map/nodes/{id}/body``     : small green cube marking the keyframe
- ``map/edges/{covis,odom,trav}`` : graph edges as blue LineStrips3D (log_map_edges)
- ``map/objects``             : L4 object graph oriented bounding boxes

Map entities are timeless (whole map shown at once), matching utils_rerun.
Saved as .rrd (open with `rerun x.rrd`).

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
import rerun.blueprint as rrb  # noqa: E402

_ONM_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # opennavmap/python
_LITEVLOC = os.path.join(os.path.dirname(_ONM_PY), "third_party", "litevloc_code", "python")
for _p in (_ONM_PY, _LITEVLOC):
    if _p not in sys.path:
        sys.path.append(_p)

from utils.utils_rerun import log_map_edges, log_world_frame_axes  # litevloc rerun utils

_BODY_HALF = np.array([0.03, 0.03, 0.03], dtype=np.float32)
_NODE_COLOR = np.array([[0, 180, 100]], dtype=np.uint8)
_OBJ_COLOR = np.array([[214, 39, 40]], dtype=np.uint8)
_FRUSTUM_DIST = 0.75  # enlarged camera-frustum image-plane distance


def _load_rgb(path: Path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def log_map_nodes(covis) -> None:
    """Per-keyframe camera frustum + RGB image + body cube (timeless).

    Same entity layout as utils_rerun.log_map_nodes, but the RGB is loaded from
    the stored ``seq/`` (our ImageNode keeps rgb_image=None to save memory).
    """
    from scipy.spatial.transform import Rotation as R

    root = Path(covis.map_root)
    for node in covis.nodes.values():
        entity = f"map/nodes/{node.id}"
        width, height = int(node.img_size[0]), int(node.img_size[1])
        rot = R.from_quat(np.asarray(node.quat, float).reshape(4)).as_matrix()
        rr.log(entity, rr.Transform3D(
            translation=np.asarray(node.trans, float).reshape(3).tolist(),
            mat3x3=rot.tolist()), static=True)
        rr.log(entity + "/camera", rr.Pinhole(
            image_from_camera=np.asarray(node.K, float).reshape(3, 3),
            width=width, height=height, image_plane_distance=_FRUSTUM_DIST), static=True)
        rr.log(entity + "/body", rr.Boxes3D(half_sizes=[_BODY_HALF], colors=_NODE_COLOR), static=True)
        rgb = _load_rgb(root / node.rgb_img_name)
        if rgb is not None:
            rr.log(entity + "/camera", rr.Image(rgb), static=True)


def log_map_objects(manager) -> None:
    """L4 object graph as oriented bounding boxes at ``map/objects`` (timeless)."""
    object_graph = manager.graphs.get("object")
    if object_graph is None or object_graph.get_num_node() == 0:
        return
    from scipy.spatial.transform import Rotation as R

    centers, half_sizes, rotations, labels = [], [], [], []
    for node in object_graph.nodes.values():
        centers.append(np.asarray(node.obb.center, float).reshape(3))
        half_sizes.append(np.asarray(node.obb.size, float).reshape(3) / 2.0)
        rotations.append(rr.Quaternion(
            xyzw=R.from_matrix(np.asarray(node.obb.R, float).reshape(3, 3)).as_quat()))
        labels.append(f"{node.label} {node.confidence:.2f}")
    rr.log("map/objects", rr.Boxes3D(centers=centers, half_sizes=half_sizes,
                                     rotations=rotations, labels=labels, colors=_OBJ_COLOR), static=True)


def visualize_map(manager, out_rrd: str, app_id: str = "opennavmap") -> str:
    """Log the map (world axes, keyframe frustums+rgb+body, edges, objects) to a Rerun .rrd."""
    rr.init(app_id, spawn=False)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial3DView(name="OpenNavMap", origin="/"), auto_space_views=False))

    log_world_frame_axes(length=0.5)
    covis = manager.covis
    if covis is not None and covis.get_num_node() > 0:
        log_map_nodes(covis)
    for edge_type in ("covis", "odom", "trav"):
        graph = manager.graphs.get(edge_type)
        if graph is not None and graph.get_num_node() > 0:
            log_map_edges(graph, edge_type)
    log_map_objects(manager)

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
