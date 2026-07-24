#! /usr/bin/env python
"""Rerun (.rrd) visualization for an OpenNavMap map session.

Follows the litevloc ``utils_rerun`` entity conventions (reusing its world-axes
helper), with two project-specific tweaks:
- keyframe nodes + edges appear on a ``node_time`` timeline (scrub to see the map
  built keyframe-by-keyframe), instead of being fully timeless;
- covis/odom/trav edges are color-coded (green/blue/orange).

Entities:
- ``world/axes``            : XYZ world-frame axes (log_world_frame_axes, timeless)
- ``map/nodes/{id}``        : per-keyframe Transform3D (on node_time)
- ``map/nodes/{id}/camera`` : Pinhole camera frustum (no rgb texture)
- ``map/nodes/{id}/body``   : small green cube marking the keyframe
- ``map/edges/{type}/{a}-{b}`` : color-coded edges appearing at the later endpoint
- ``map/objects/boxes``     : L4 object graph OBBs (timeless)
- ``map/objects/centers``   : object OBB geometric centers (timeless)
- ``map/objects/points/{id}``: per-object detected point cloud (timeless)
- ``map/objects/vis_edges/*``: object-center -> keyframe visibility edges (timeless, red)
- ``camera/color`` / ``camera/depth`` : current keyframe rgb/depth (2D horizontal window);
                             rgb has visible objects' projected 3D OBB wireframes + type
                             labels drawn on it via OpenCV (clipped to the original WxH)

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

from utils.utils_rerun import log_world_frame_axes  # litevloc rerun utils

_BODY_HALF = np.array([0.03, 0.03, 0.03], dtype=np.float32)
_NODE_COLOR = np.array([[0, 180, 100]], dtype=np.uint8)
_OBJ_COLOR = np.array([[214, 39, 40]], dtype=np.uint8)
_OBJ_PCD_COLOR = np.array([[255, 152, 150]], dtype=np.uint8)
_OBJ_VIS_COLOR = np.array([[255, 0, 0]], dtype=np.uint8)  # object->keyframe edges (red)
_EDGE_COLORS = {"covis": [44, 160, 44], "odom": [31, 119, 180], "trav": [255, 127, 14]}
_FRUSTUM_DIST = 0.75   # enlarged camera-frustum image-plane distance
_DEPTH_METER = 1000.0  # stored depth png is uint16 millimetres
_TIMELINE = "node_time"

# OBB local-corner signs (bottom face 0-3 at -z, top face 4-7 at +z), and a
# single-stroke vertex order that traces all 12 edges (some repeated) so each box
# projects to one labelled LineStrips2D strip.
_OBB_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)
_OBB_STROKE = [0, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4]
# Distinct overlay colors cycled per visible object (bright on rgb).
_OBB_PALETTE = np.array([
    [255, 214, 0], [0, 229, 255], [124, 252, 0], [255, 105, 180],
    [255, 128, 0], [0, 255, 128], [180, 120, 255], [255, 80, 80],
], dtype=np.uint8)


def _load_rgb(path: Path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def _load_depth(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _node_time(node, fallback) -> float:
    return float(getattr(node, "time", fallback))


def log_map_nodes(covis, visible: "dict | None" = None) -> None:
    """Per-keyframe frustum + body cube + rgb/depth panels, on the node_time timeline.

    If ``visible`` (kf_id -> [object node, ...]) is given, each visible object's 3D
    OBB is projected and drawn onto the rgb with OpenCV (clipped to the image, so
    the logged rgb keeps its original size) before logging."""
    from scipy.spatial.transform import Rotation as R

    visible = visible or {}
    root = Path(covis.map_root)
    for nid in sorted(covis.nodes, key=lambda i: _node_time(covis.get_node(i), i)):
        node = covis.get_node(nid)
        rr.set_time_seconds(_TIMELINE, _node_time(node, nid))
        entity = f"map/nodes/{nid}"
        width, height = int(node.img_size[0]), int(node.img_size[1])
        rot = R.from_quat(np.asarray(node.quat, float).reshape(4)).as_matrix()
        rr.log(entity, rr.Transform3D(
            translation=np.asarray(node.trans, float).reshape(3).tolist(), mat3x3=rot.tolist()))
        rr.log(entity + "/camera", rr.Pinhole(
            image_from_camera=np.asarray(node.K, float).reshape(3, 3),
            width=width, height=height, image_plane_distance=_FRUSTUM_DIST))
        rr.log(entity + "/body", rr.Boxes3D(half_sizes=[_BODY_HALF], colors=_NODE_COLOR))
        rgb = _load_rgb(root / node.rgb_img_name)
        if rgb is not None:
            rgb = _draw_obb_on_image(rgb, visible.get(nid, []), node)
            rr.log("camera/color", rr.Image(rgb))
        depth = _load_depth(root / node.depth_img_name)
        if depth is not None:
            rr.log("camera/depth", rr.DepthImage(depth, meter=_DEPTH_METER))


def log_map_edges(graph, edge_type: str) -> None:
    """Color-coded edges (covis=green/odom=blue/trav=orange), each appearing at its
    later endpoint's node_time."""
    pos = {i: np.asarray(nd.trans, float).reshape(3) for i, nd in graph.nodes.items()}
    tim = {i: _node_time(nd, i) for i, nd in graph.nodes.items()}
    color = np.array([_EDGE_COLORS[edge_type]], dtype=np.uint8)
    seen, edges = set(), []
    for nid, node in graph.nodes.items():
        for neighbor, _w in node.edges.values():
            key = (min(nid, neighbor.id), max(nid, neighbor.id))
            if key in seen:
                continue
            seen.add(key)
            edges.append((max(tim[nid], tim[neighbor.id]), nid, neighbor.id))
    for t, a, b in sorted(edges):
        rr.set_time_seconds(_TIMELINE, t)
        rr.log(f"map/edges/{edge_type}/{a}-{b}",
               rr.LineStrips3D(strips=[np.array([pos[a], pos[b]], dtype=np.float32)],
                               radii=0.0025, colors=color))


def log_map_objects(manager) -> None:
    """L4 object graph: OBBs at ``map/objects/boxes`` + per-object detected point
    clouds at ``map/objects/points/{id}`` (all timeless)."""
    object_graph = manager.graphs.get("object")
    if object_graph is None or object_graph.get_num_node() == 0:
        return
    from scipy.spatial.transform import Rotation as R

    root = Path(object_graph.map_root)
    centers, half_sizes, rotations, labels = [], [], [], []
    for node in object_graph.nodes.values():
        centers.append(np.asarray(node.obb.center, float).reshape(3))
        half_sizes.append(np.asarray(node.obb.size, float).reshape(3) / 2.0)
        rotations.append(rr.Quaternion(
            xyzw=R.from_matrix(np.asarray(node.obb.R, float).reshape(3, 3)).as_quat()))
        labels.append(f"{node.label} {node.confidence:.2f}")
        if node.pointcloud_ref:
            pcd_path = root / node.pointcloud_ref
            if pcd_path.exists():
                pts = np.load(pcd_path).astype(np.float32).reshape(-1, 3)
                rr.log(f"map/objects/points/{node.id}",
                       rr.Points3D(pts, colors=_OBJ_PCD_COLOR, radii=0.01), static=True)
    rr.log("map/objects/boxes", rr.Boxes3D(centers=centers, half_sizes=half_sizes,
                                           rotations=rotations, labels=labels,
                                           colors=_OBJ_COLOR), static=True)
    # OBB geometric centers -- the endpoint that object->keyframe edges connect to.
    rr.log("map/objects/centers",
           rr.Points3D(centers, colors=_OBJ_COLOR, radii=0.05), static=True)


def _obb_world_corners(obb) -> np.ndarray:
    """8 world-frame corners of an OBB (center/size/R), ordered per _OBB_SIGNS."""
    center = np.asarray(obb.center, float).reshape(3)
    half = np.asarray(obb.size, float).reshape(3) / 2.0
    rot = np.asarray(obb.R, float).reshape(3, 3)
    return center + (_OBB_SIGNS * half) @ rot.T


def _project_corners(world_corners: np.ndarray, node) -> "np.ndarray | None":
    """Project world corners into a keyframe's CV image plane. Returns 8x2 pixel
    coords, or None if any corner is at/behind the camera (skip partial boxes)."""
    from scipy.spatial.transform import Rotation as R

    rot = R.from_quat(np.asarray(node.quat, float).reshape(4)).as_matrix()  # cam->world
    trans = np.asarray(node.trans, float).reshape(3)
    cam = (world_corners - trans) @ rot  # world->cam == R^T @ (p - t)
    z = cam[:, 2]
    if np.any(z <= 1e-3):
        return None
    K = np.asarray(node.K, float).reshape(3, 3)
    uv = (K @ cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def _visible_objects(manager) -> dict:
    """Map kf_id -> [object node, ...] from each object's ``observed_keyframes``."""
    object_graph = manager.graphs.get("object")
    visible: dict = {}
    if object_graph is None:
        return visible
    for node in object_graph.nodes.values():
        for kf_id, _score in node.observed_keyframes:
            visible.setdefault(kf_id, []).append(node)
    return visible


def _draw_obb_on_image(rgb: np.ndarray, objs: list, node) -> np.ndarray:
    """Draw each object's projected 3D OBB wireframe + type label onto a copy of
    the rgb (OpenCV clips to the image, keeping the original WxH). ``rgb`` is RGB
    uint8; colors are passed as RGB so channels stay correct in ``rr.Image``."""
    if not objs:
        return rgb
    img = np.ascontiguousarray(rgb).copy()
    for i, obj in enumerate(objs):
        uv = _project_corners(_obb_world_corners(obj.obb), node)
        if uv is None:
            continue
        color = tuple(int(c) for c in _OBB_PALETTE[i % len(_OBB_PALETTE)])
        pts = uv[_OBB_STROKE].round().astype(np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        top = uv[int(np.argmin(uv[:, 1]))]
        org = (int(round(top[0])), max(10, int(round(top[1])) - 4))
        cv2.putText(img, obj.label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def log_object_visibility_edges(manager) -> None:
    """object->keyframe visibility edges (each node's ``observed_keyframes``) as red
    lines from the object OBB center to the observing keyframe body (timeless)."""
    object_graph = manager.graphs.get("object")
    covis = manager.covis
    if object_graph is None or covis is None or object_graph.get_num_node() == 0:
        return
    for node in object_graph.nodes.values():
        oc = np.asarray(node.obb.center, float).reshape(3)
        for kf_id, _score in node.observed_keyframes:
            if kf_id not in covis.nodes:
                continue
            kc = np.asarray(covis.get_node(kf_id).trans, float).reshape(3)
            rr.log(f"map/objects/vis_edges/{node.id}-{kf_id}",
                   rr.LineStrips3D(strips=[np.array([oc, kc], dtype=np.float32)],
                                   radii=0.0015, colors=_OBJ_VIS_COLOR), static=True)


def visualize_map(manager, out_rrd: str, app_id: str = "opennavmap") -> str:
    """Log the map to a Rerun .rrd: 3D map on top, rgb/depth horizontal window below."""
    rr.init(app_id, spawn=False)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(name="OpenNavMap", origin="/"),
            rrb.Horizontal(
                rrb.Spatial2DView(name="rgb", origin="/camera/color"),
                rrb.Spatial2DView(name="depth", origin="/camera/depth"),
            ),
            row_shares=[3, 1],
        ),
        auto_space_views=False,
    ))

    log_world_frame_axes(length=1.5)
    covis = manager.covis
    if covis is not None and covis.get_num_node() > 0:
        log_map_nodes(covis, _visible_objects(manager))
    for edge_type in ("covis", "odom", "trav"):
        graph = manager.graphs.get(edge_type)
        if graph is not None and graph.get_num_node() > 0:
            log_map_edges(graph, edge_type)
    log_map_objects(manager)
    log_object_visibility_edges(manager)

    out_path = Path(out_rrd)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_path))
    return str(out_path)


def main() -> int:
    from map_manager import load_map  # T1.4: shared reload helper

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True, help="stored map directory")
    parser.add_argument("--out", type=Path, required=True, help="output .rrd path")
    args = parser.parse_args()
    out = visualize_map(load_map(args.map), str(args.out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
