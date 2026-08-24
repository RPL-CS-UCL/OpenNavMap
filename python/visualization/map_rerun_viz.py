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
- ``map/objects/labels/{id}``: per-object category+confidence text above each box
                             (one single-instance Points3D each so rerun 0.17 renders
                             it as always-on 3D text, not a hover-only marker; timeless)
- ``map/objects/points/{id}``: per-object detected point cloud (timeless)
- ``map/objects/vis_edges/*``: object-center -> keyframe visibility edges (timeless, red)
- ``map/rooms/labels/{id}``  : room-layer node (v2.2): one single-instance Points3D
                             per room, lifted 2 m above its member-keyframe centroid
                             with an always-on "label (id)" text (timeless)
- ``map/rooms/hier_edges/*`` : room node -> member keyframe hierarchy edges (timeless)
- ``map/rooms/adj_edges/*``  : room <-> room adjacency edges (timeless)
- ``camera/color`` / ``camera/depth`` : current keyframe rgb/depth (2D horizontal window);
                             rgb has visible objects' projected 3D OBB wireframes + type
                             labels drawn on it via OpenCV (clipped to the original WxH)
- ``map/explore/frontiers``    : P3 frontier candidates (on node_time; opengoalnav
                             ``core.explore.FrontierCandidate``-shaped objects, duck-typed
                             so this fork stays independent of the main project)
- ``map/explore/decision_path``: P3 agent path so far, segmented by the arbitration
                             action that produced each step (go_to_frontier=blue,
                             go_to_target=green), on node_time

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
_ROOM_RAISE = 2.0  # room node height above its member-kf centroid (visual "upper layer")
_ROOM_PALETTE = np.array([
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
    [188, 189, 34], [23, 190, 207], [214, 39, 40], [31, 119, 180],
], dtype=np.uint8)
_ROOM_ADJ_COLOR = np.array([[148, 103, 189]], dtype=np.uint8)  # room<->room (purple)
_FRONTIER_COLOR = np.array([[255, 200, 0]], dtype=np.uint8)  # P3 T3.5: frontier candidates (yellow)
_PATH_EXPLORE_COLOR = [31, 119, 180]  # P3 T3.5: go_to_frontier path segments (blue)
_PATH_CONVERGE_COLOR = [44, 160, 44]  # P3 T3.5: go_to_target path segments (green)
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
# The 12 box edges as corner-index pairs. Used by the near-plane clipper, which
# has to treat each edge on its own instead of drawing one single stroke.
_OBB_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
# Near plane (metres). Anything closer is behind/at the camera: u = fx*x/z blows
# up to tens of thousands of pixels there, which draws a line straight across the
# frame -- worse than drawing nothing. Edges get cut at this depth instead.
_NEAR_PLANE = 0.05
# Distinct overlay colors, picked by object id so one object keeps its color
# across every keyframe it appears in (bright on rgb).
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


def log_map_nodes(covis, visible: "dict | None" = None, images: bool = True) -> None:
    """Per-keyframe frustum + body cube (+ rgb/depth panels if ``images``), on the node_time timeline.

    If ``visible`` (kf_id -> [object node, ...]) is given and ``images`` is True, each visible
    object's 3D OBB is projected and drawn onto the rgb with OpenCV (clipped to the image, so
    the logged rgb keeps its original size) before logging. Callers that log rgb/depth on their
    own timeline (e.g. per-raw-frame instead of per-keyframe) should pass ``images=False`` here
    to avoid two writers racing for the same ``camera/color`` / ``camera/depth`` entities."""
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
        if not images:
            continue
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
                               radii=0.01, colors=color))


def log_map_objects(manager) -> None:
    """L4 object graph: OBBs at ``map/objects/boxes`` + per-object detected point
    clouds at ``map/objects/points/{id}`` (all timeless)."""
    object_graph = manager.graphs.get("object")
    if object_graph is None or object_graph.get_num_node() == 0:
        return
    from scipy.spatial.transform import Rotation as R

    root = Path(object_graph.map_root)
    centers, half_sizes, rotations = [], [], []
    for node in object_graph.nodes.values():
        center = np.asarray(node.obb.center, float).reshape(3)
        half = np.asarray(node.obb.size, float).reshape(3) / 2.0
        centers.append(center)
        half_sizes.append(half)
        rotations.append(rr.Quaternion(
            xyzw=R.from_matrix(np.asarray(node.obb.R, float).reshape(3, 3)).as_quat()))
        # Always-on category/confidence text in the 3D view: one single-instance
        # Points3D per object, anchored just above the box top. In rerun 0.17 a
        # single label on a single-instance entity is drawn as persistent 3D text,
        # whereas a batched Points3D with many labels shows only hover-only markers.
        anchor = center + np.array([0.0, 0.0, float(np.max(half)) + 0.1])
        rr.log(f"map/objects/labels/{node.id}",
               rr.Points3D([anchor], colors=_OBJ_COLOR, radii=0.01,
                           labels=[f"{node.label} {node.confidence:.2f}"]), static=True)
        if node.pointcloud_ref:
            pcd_path = root / node.pointcloud_ref
            if pcd_path.exists():
                pts = np.load(pcd_path).astype(np.float32).reshape(-1, 3)
                rr.log(f"map/objects/points/{node.id}",
                       rr.Points3D(pts, colors=_OBJ_PCD_COLOR, radii=0.01), static=True)
    rr.log("map/objects/boxes", rr.Boxes3D(centers=centers, half_sizes=half_sizes,
                                           rotations=rotations, colors=_OBJ_COLOR), static=True)
    # OBB geometric centers -- the endpoint that object->keyframe edges connect to.
    rr.log("map/objects/centers",
           rr.Points3D(centers, colors=_OBJ_COLOR, radii=0.05), static=True)


def _obb_world_corners(obb) -> np.ndarray:
    """8 world-frame corners of an OBB (center/size/R), ordered per _OBB_SIGNS."""
    center = np.asarray(obb.center, float).reshape(3)
    half = np.asarray(obb.size, float).reshape(3) / 2.0
    rot = np.asarray(obb.R, float).reshape(3, 3)
    return center + (_OBB_SIGNS * half) @ rot.T


def _corners_in_camera(world_corners: np.ndarray, node) -> np.ndarray:
    """World corners -> this keyframe's camera frame (CV: x right, y down, z fwd)."""
    from scipy.spatial.transform import Rotation as R

    rot = R.from_quat(np.asarray(node.quat, float).reshape(4)).as_matrix()  # cam->world
    trans = np.asarray(node.trans, float).reshape(3)
    return (world_corners - trans) @ rot  # world->cam == R^T @ (p - t)


def _project_points(cam_points: np.ndarray, node) -> np.ndarray:
    """Camera-frame points (all with z > 0) -> Nx2 pixel coords."""
    K = np.asarray(node.K, float).reshape(3, 3)
    uv = (K @ np.asarray(cam_points, float).T).T
    return uv[:, :2] / uv[:, 2:3]


def _clip_segment_to_image(p0: np.ndarray, p1: np.ndarray,
                           width: int, height: int) -> "np.ndarray | None":
    """Liang-Barsky: clip a pixel-space segment to [0,width] x [0,height].

    Returns the (possibly shortened) 2x2 endpoint array, or None if the whole
    segment falls outside the image. Unlike a plain clamp, this keeps the
    surviving portion on the original line (no direction distortion)."""
    dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0]), (dx, width - p0[0]), (-dy, p0[1]), (dy, height - p0[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    delta = np.array([dx, dy])
    return np.stack([p0 + t0 * delta, p0 + t1 * delta])


def _clip_obb_edges(world_corners: np.ndarray, node,
                    near: float = _NEAR_PLANE) -> "list[np.ndarray]":
    """Project an OBB's 12 edges, clipping each one at the near plane and at the
    image border.

    Returns a list of 2x2 arrays (one per surviving edge, endpoints in pixels).
    Each edge is handled independently, so a box that is only partly in front of
    the camera still draws its visible portion instead of vanishing:

      both endpoints in front  -> drawn as is
      one in front, one behind -> the behind end is moved to where the edge
                                  crosses z = near, and the edge is drawn
      both behind              -> dropped

    The projected segment is then clipped to [0,width] x [0,height] (the node's
    own image size). cv2.polylines used to do this step implicitly for the
    OpenCV overlay caller, but callers that hand these pixel coords to something
    else (e.g. rerun's LineStrips2D) get raw, unclipped coordinates back — an
    edge that projects far outside the frame (nearby box, oblique angle) can
    otherwise blow up a viewer's auto-fit range and shrink the image to a speck.
    """
    cam = _corners_in_camera(world_corners, node)
    z = cam[:, 2]
    width, height = int(node.img_size[0]), int(node.img_size[1])
    segments = []
    for i, j in _OBB_EDGES:
        a, b = cam[i], cam[j]
        in_a, in_b = z[i] > near, z[j] > near
        if not in_a and not in_b:
            continue
        if not in_a or not in_b:
            # Walk from the visible end towards the hidden one and stop at z = near
            src, dst = (a, b) if in_a else (b, a)
            t = (near - src[2]) / (dst[2] - src[2])
            clipped = src + t * (dst - src)
            a, b = (src, clipped)
        pix = _project_points(np.stack([a, b]), node)
        seg = _clip_segment_to_image(pix[0], pix[1], width, height)
        if seg is not None:
            segments.append(seg)
    return segments


def _project_corners(world_corners: np.ndarray, node) -> "np.ndarray | None":
    """All 8 corners as 8x2 pixel coords, or None if any is at/behind the camera.

    Kept for callers that need the full corner set (e.g. a 2D bounding box). To
    draw a wireframe use _clip_obb_edges, which survives partial visibility.
    """
    cam = _corners_in_camera(world_corners, node)
    if np.any(cam[:, 2] <= 1e-3):
        return None
    return _project_points(cam, node)


def _object_color(obj, fallback: int = 0) -> tuple:
    """Overlay color for an object, keyed on its id so it stays the same in every
    keyframe. Ids look like ``obj_12``; anything unparseable falls back to the
    caller's index."""
    digits = "".join(ch for ch in str(getattr(obj, "id", "")) if ch.isdigit())
    key = int(digits) if digits else int(fallback)
    return tuple(int(c) for c in _OBB_PALETTE[key % len(_OBB_PALETTE)])


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


def _draw_obb_on_image(rgb: np.ndarray, objs: list, node, label_fn=None,
                       font_scale: float = 0.4, outline: bool = False) -> np.ndarray:
    """Draw each object's projected 3D OBB wireframe + label onto a copy of the rgb.

    Edges are near-plane clipped per edge, so a box the camera is standing inside
    of still shows the part that is in front of it. OpenCV clips the rest to the
    image, keeping the original WxH. ``rgb`` is RGB uint8; colors are passed as
    RGB so channels stay correct in ``rr.Image``.

    label_fn  optional obj -> str; defaults to the bare class name
    outline    draw the text twice (thick black underneath) so it stays readable
               on light backgrounds like white walls
    """
    if not objs:
        return rgb
    img = np.ascontiguousarray(rgb).copy()
    for i, obj in enumerate(objs):
        segments = _clip_obb_edges(_obb_world_corners(obj.obb), node)
        if not segments:
            continue
        color = _object_color(obj, i)
        for seg in segments:
            pts = seg.round().astype(np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2,
                          lineType=cv2.LINE_AA)
        uv = np.concatenate(segments, axis=0)
        top = uv[int(np.argmin(uv[:, 1]))]
        org = (int(round(top[0])), max(10, int(round(top[1])) - 4))
        text = obj.label if label_fn is None else label_fn(obj)
        if outline:
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    1, cv2.LINE_AA)
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


def log_map_rooms(manager) -> None:
    """Room layer (v2.2) as a simple hierarchy: one 3D node per room with an
    always-on room-type label, edges down to its member keyframes, plus
    room<->room adjacency edges. All timeless.

    The room node is anchored ``_ROOM_RAISE`` m above its member-keyframe
    position centroid (fallback: the stored room centroid), so the rooms read
    as an upper layer floating over the keyframe graph. Single-instance
    Points3D per room = persistent 3D text in rerun 0.17 (same trick as
    ``map/objects/labels/{id}``).
    """
    room_graph = manager.graphs.get("room")
    covis = manager.covis
    if room_graph is None or room_graph.get_num_node() == 0:
        return
    anchors = {}
    for idx, node in enumerate(room_graph.nodes.values()):
        member_pos = []
        if covis is not None:
            member_pos = [np.asarray(covis.get_node(k).trans, float).reshape(3)
                          for k in node.member_keyframes if covis.get_node(k) is not None]
        base = np.mean(member_pos, axis=0) if member_pos else np.asarray(node.trans, float).reshape(3)
        anchor = base + np.array([0.0, 0.0, _ROOM_RAISE])
        anchors[node.id] = anchor
        color = _ROOM_PALETTE[idx % len(_ROOM_PALETTE)][None, :]
        rr.log(f"map/rooms/labels/{node.id}",
               rr.Points3D([anchor], colors=color, radii=0.08,
                           labels=[f"{node.label} ({node.id})"]), static=True)
        for kf_id in node.member_keyframes:
            if covis is None or covis.get_node(kf_id) is None:
                continue
            kf_pos = np.asarray(covis.get_node(kf_id).trans, float).reshape(3)
            rr.log(f"map/rooms/hier_edges/{node.id}-{kf_id}",
                   rr.LineStrips3D(strips=[np.array([anchor, kf_pos], dtype=np.float32)],
                                   radii=0.004, colors=color), static=True)
    seen = set()
    for node in room_graph.nodes.values():
        for neighbor, _w in node.edges.values():
            key = tuple(sorted((str(node.id), str(neighbor.id))))
            if key in seen or neighbor.id not in anchors:
                continue
            seen.add(key)
            rr.log(f"map/rooms/adj_edges/{key[0]}-{key[1]}",
                   rr.LineStrips3D(
                       strips=[np.array([anchors[node.id], anchors[neighbor.id]], dtype=np.float32)],
                       radii=0.02, colors=_ROOM_ADJ_COLOR), static=True)


def log_frontier_candidates(frontiers: list, timestamp) -> None:
    """P3 T3.5: log frontier candidates as points at ``map/explore/frontiers``.

    Follows ``log_map_edges``'s pattern (single ``_TIMELINE`` timestamp per
    call, not ``static=True``) so a scrub through node_time also replays how
    the frontier set evolved. ``frontiers`` is duck-typed (``.frontier_id``,
    ``.position``, ``.region_size``) rather than importing
    ``core.explore.FrontierCandidate``, keeping this fork free of a reverse
    dependency on the main opengoalnav project.

    Args:
        frontiers: frontier candidates for the current step; no-op if empty.
        timestamp: ``_TIMELINE`` seconds to log this snapshot at.
    """
    if not frontiers:
        return
    rr.set_time_seconds(_TIMELINE, timestamp)
    positions = [np.asarray(f.position, dtype=np.float32).reshape(3) for f in frontiers]
    labels = [f"id={f.frontier_id} size={f.region_size}" for f in frontiers]
    rr.log("map/explore/frontiers", rr.Points3D(positions, colors=_FRONTIER_COLOR, radii=0.06, labels=labels))


def log_decision_path(positions: list, actions: list, timestamp) -> None:
    """P3 T3.5: log the agent's path at ``map/explore/decision_path``, colored per segment.

    ``actions[i]`` is the arbitration action that produced ``positions[i]``
    (so the segment ``positions[i-1] -> positions[i]`` is colored by
    ``actions[i]``); ``go_to_frontier`` segments are blue, ``go_to_target``
    segments are green.

    Args:
        positions: agent positions so far, oldest first; no-op if fewer than 2.
        actions: one action per position, same length and order as ``positions``.
        timestamp: ``_TIMELINE`` seconds to log this snapshot at.
    """
    if len(positions) < 2:
        return
    rr.set_time_seconds(_TIMELINE, timestamp)
    pts = [np.asarray(p, dtype=np.float32).reshape(3) for p in positions]
    strips, colors = [], []
    for i in range(1, len(pts)):
        strips.append(np.array([pts[i - 1], pts[i]], dtype=np.float32))
        colors.append(_PATH_CONVERGE_COLOR if actions[i] == "go_to_target" else _PATH_EXPLORE_COLOR)
    rr.log("map/explore/decision_path",
           rr.LineStrips3D(strips=strips, radii=0.015, colors=np.array(colors, dtype=np.uint8)))


def visualize_map(
    manager,
    out_rrd: str,
    app_id: str = "opennavmap",
    frontiers: "list | None" = None,
    decision_positions: "list | None" = None,
    decision_actions: "list | None" = None,
    explore_timestamp: float = 0.0,
) -> str:
    """Log the map to a Rerun .rrd: 3D map on top, rgb/depth horizontal window below."""
    rr.init(app_id, spawn=False)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="OpenNavMap", origin="/"),
            rrb.Vertical(
                rrb.Spatial2DView(name="rgb", origin="/camera/color"),
                rrb.Spatial2DView(name="depth", origin="/camera/depth"),
            ),
            column_shares=[3.5, 1],
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
    log_map_rooms(manager)
    if frontiers:
        log_frontier_candidates(frontiers, explore_timestamp)
    if decision_positions:
        log_decision_path(decision_positions, decision_actions or [], explore_timestamp)

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
