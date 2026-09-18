"""Write one submap directory in the OpenNavMap / LiteVLoc map format.

Pose convention on disk (``poses.txt`` and ``poses_abs_gt.txt``): each row
is ``name qw qx qy qz tx ty tz`` of the WORLD-TO-CAMERA transform, i.e. the
inverse of the camera pose. ``poses.txt`` is expressed in a per-session
frame (the first keyframe's body pose when anchored), ``poses_abs_gt.txt``
in the global frame of the recording.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from rosbag_convert.bag_reader import Intrinsics, ns_to_seconds
from utils.utils_geom import DEFAULT_IMAGE_TEMPLATE, convert_matrix_to_vec

MIN_KEYFRAMES_PER_SUBMAP = 3  # a 1-row edge file loads as 1-D and breaks base_graph.read_edge_list
GPS_NAN_ROW = "nan nan nan nan nan"
EDGE_TYPES = ("odom", "covis", "trav")


@dataclass
class Frame:
    """One keyframe: image name, stamp and its camera / body poses in the global frame."""
    name: str
    stamp_ns: int
    T_w_c: np.ndarray
    T_w_b: np.ndarray


def frame_name(index: int) -> str:
    return DEFAULT_IMAGE_TEMPLATE.format(frame_id=index)


def w2c_row(T_w_c: np.ndarray) -> np.ndarray:
    """``[qw qx qy qz tx ty tz]`` of ``inv(T_w_c)``, the on-disk pose format."""
    trans, quat_wxyz = convert_matrix_to_vec(np.linalg.inv(T_w_c), mode="wxyz")
    return np.concatenate([quat_wxyz, trans])


def _write_rows(path: Path, rows: Iterable[str]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(row + "\n")


def _pose_rows(frames: List[Frame], T_ref_w: np.ndarray) -> List[str]:
    rows = []
    for fr in frames:
        vec = w2c_row(T_ref_w @ fr.T_w_c)
        rows.append(fr.name + " " + " ".join(f"{v:.6f}" for v in vec))
    return rows


def _chain_edges(frames: List[Frame]) -> List[str]:
    rows = []
    for i in range(len(frames) - 1):
        dist = float(np.linalg.norm(frames[i + 1].T_w_c[:3, 3] - frames[i].T_w_c[:3, 3]))
        rows.append(f"{i} {i + 1} {dist:.6f}")
    return rows


def write_submap_metadata(out_dir: Path, frames: List[Frame], intr: Intrinsics,
                          anchor_to_first_body: bool) -> None:
    """Write every text file of a submap (images are written separately).

    Args:
        out_dir: Submap directory (created if needed).
        frames: Keyframes in time order; ``frames[i]`` becomes node ``i``.
        intr: Intrinsics shared by all frames.
        anchor_to_first_body: If True, ``poses.txt`` is relative to
            ``frames[0].T_w_b``; otherwise it equals ``poses_abs_gt.txt``.

    Raises:
        ValueError: If fewer than ``MIN_KEYFRAMES_PER_SUBMAP`` frames are given.
    """
    if len(frames) < MIN_KEYFRAMES_PER_SUBMAP:
        raise ValueError(f"A submap needs at least {MIN_KEYFRAMES_PER_SUBMAP} keyframes, got {len(frames)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T_ref_w = np.linalg.inv(frames[0].T_w_b) if anchor_to_first_body else np.eye(4)

    _write_rows(out_dir / "timestamps.txt",
                (f"{fr.name} {ns_to_seconds(fr.stamp_ns):.9f}" for fr in frames))
    intr_str = f"{intr.fx:.6f} {intr.fy:.6f} {intr.cx:.6f} {intr.cy:.6f} {intr.width} {intr.height}"
    _write_rows(out_dir / "intrinsics.txt", (f"{fr.name} {intr_str}" for fr in frames))
    _write_rows(out_dir / "poses.txt", _pose_rows(frames, T_ref_w))
    _write_rows(out_dir / "poses_abs_gt.txt", _pose_rows(frames, np.eye(4)))
    _write_rows(out_dir / "gps_data.txt", (f"{fr.name} {GPS_NAN_ROW}" for fr in frames))
    chain = _chain_edges(frames)
    for edge_type in EDGE_TYPES:
        _write_rows(out_dir / f"edges_{edge_type}.txt", chain)


def write_image(out_dir: Path, name: str, img_bgr: np.ndarray, jpeg_quality: int) -> None:
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
        raise IOError(f"Failed to write image {path}")


def write_orders_file(root: Path, scene: str, submap_ids: List[int]) -> Path:
    """Write ``<root>/<scene>_orders.txt`` whose first (only) line is the in-order sequence."""
    path = Path(root) / f"{scene}_orders.txt"
    _write_rows(path, [" ".join(str(i) for i in submap_ids)])
    return path
