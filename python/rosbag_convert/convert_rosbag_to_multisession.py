"""Convert one ROS1 bag into N simulated-session submaps.

Usage:
    PYTHONPATH=python:third_party/litevloc_code/python \\
    python python/rosbag_convert/convert_rosbag_to_multisession.py \\
        --config python/rosbag_convert/config/cmt_szs_odin1.yaml [--overwrite] [--max_frames N]

Two passes over the bag keep memory flat: pass 1 reads stamps and poses
and decides the keyframes, pass 2 decodes and writes only those images.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from rosbag_convert import bag_reader
from rosbag_convert.bag_reader import Trajectory
from rosbag_convert.config import CAMERA_POSE_FROM_TF, SPLIT_EQUAL_DISTANCE, BagConversionConfig
from rosbag_convert.keyframes import select_keyframes
from rosbag_convert.map_writer import (Frame, frame_name, write_image, write_orders_file,
                                       write_submap_metadata)
from rosbag_convert.sessions import split_by_distance, split_by_time
from rosbag_convert.sync import associate_by_stamp, cumulative_path_length

logger = logging.getLogger("rosbag_convert")
ROTATION_ONLY_EDGE_M = 0.05  # odom edges shorter than this come from in-place turns


def _static_extrinsic(cfg: BagConversionConfig) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(cfg.T_body_camera_quat_xyzw).as_matrix()
    T[:3, 3] = cfg.T_body_camera_translation
    return T


def load_trajectories(cfg: BagConversionConfig) -> Tuple[Trajectory, Trajectory]:
    """Pass 1 (poses): return ``(camera_traj, body_traj)`` in the odometry frame."""
    body = bag_reader.read_odometry(cfg.bag_path, cfg.odometry_topic)
    if cfg.camera_pose_source == CAMERA_POSE_FROM_TF:
        cam = bag_reader.read_tf_pairs(cfg.bag_path, cfg.tf_topic, cfg.odom_frame, cfg.camera_frame)
    else:
        cam = Trajectory(body.stamps.copy(), body.poses @ _static_extrinsic(cfg))
    logger.info("camera poses: %d, body poses: %d", len(cam), len(body))
    return cam, body


def build_sessions(cfg: BagConversionConfig, cam: Trajectory, body: Trajectory,
                   image_stamps_ns: np.ndarray) -> List[List[Frame]]:
    """Associate images with poses, split into sessions and select keyframes.

    Returns:
        One list of ``Frame`` per session, names restarting at 0 in each.
    """
    img_s = image_stamps_ns / bag_reader.NS_PER_S
    img_to_cam = associate_by_stamp(img_s, cam.stamps, cfg.sync_tolerance_s)
    cam_to_body = dict(associate_by_stamp(cam.stamps, body.stamps, cfg.sync_tolerance_s))
    synced = [(i, c, cam_to_body[c]) for i, c in img_to_cam if c in cam_to_body]
    logger.info("synced %d / %d images (tolerance %.3f s)", len(synced), len(image_stamps_ns),
                cfg.sync_tolerance_s)
    if not synced:
        raise RuntimeError("No image could be matched to a pose; check topics and sync_tolerance_s")

    T_w_c = np.stack([cam.poses[c] for _, c, _ in synced])
    T_w_b = np.stack([body.poses[b] for _, _, b in synced])
    stamps_ns = np.array([image_stamps_ns[i] for i, _, _ in synced], dtype=np.int64)

    if cfg.split_mode == SPLIT_EQUAL_DISTANCE:
        ranges = split_by_distance(T_w_c[:, :3, 3], cfg.num_sessions)
    else:
        ranges = split_by_time(stamps_ns / bag_reader.NS_PER_S, cfg.split_time_boundaries_s)

    sessions: List[List[Frame]] = []
    for sid, (start, end) in enumerate(ranges):
        kf = select_keyframes(T_w_c[start:end], cfg.kf_trans_thresh_m, cfg.kf_rot_thresh_deg)
        frames = [Frame(frame_name(k), int(stamps_ns[start + j]), T_w_c[start + j], T_w_b[start + j])
                  for k, j in enumerate(kf)]
        length = cumulative_path_length(T_w_c[start:end, :3, 3])[-1]
        logger.info("session %d: frames %d..%d (%d), path %.1f m, keyframes %d",
                    sid, start, end, end - start, length, len(frames))
        sessions.append(frames)
    return sessions


def _log_edge_stats(sid: int, frames: List[Frame]) -> None:
    dists = np.array([np.linalg.norm(b.T_w_c[:3, 3] - a.T_w_c[:3, 3]) for a, b in zip(frames, frames[1:])])
    logger.info("session %d odom edges: min %.3f m, max %.3f m, rotation-only (<%.2f m): %d",
                sid, dists.min(), dists.max(), ROTATION_ONLY_EDGE_M, int((dists < ROTATION_ONLY_EDGE_M).sum()))


def export_images(cfg: BagConversionConfig, sessions: List[List[Frame]]) -> None:
    """Pass 2: decode and write the selected keyframe images."""
    wanted: Dict[int, Tuple[int, str]] = {}
    for sid, frames in enumerate(sessions):
        for fr in frames:
            wanted[fr.stamp_ns] = (sid, fr.name)
    written = 0
    for ns, img in bag_reader.iter_images(cfg.bag_path, cfg.image_topic, set(wanted)):
        sid, name = wanted[ns]
        write_image(cfg.data_dir / str(sid), name, img, cfg.jpeg_quality)
        written += 1
    missing = [(sid, fr.name) for sid, frames in enumerate(sessions) for fr in frames
               if not (cfg.data_dir / str(sid) / fr.name).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} keyframe images were not found in the bag, e.g. {missing[:3]}")
    logger.info("wrote %d images", written)


def convert(cfg: BagConversionConfig, overwrite: bool, max_frames: Optional[int]) -> Path:
    """Run both passes and write ``cfg.data_dir`` plus the orders file."""
    if cfg.data_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{cfg.data_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(cfg.data_dir)
    intr = bag_reader.read_camera_info(cfg.bag_path, cfg.camera_info_topic)
    image_stamps_ns = bag_reader.read_image_stamps_ns(cfg.bag_path, cfg.image_topic)
    if max_frames is not None:
        image_stamps_ns = image_stamps_ns[:max_frames]
    cam, body = load_trajectories(cfg)
    sessions = build_sessions(cfg, cam, body, image_stamps_ns)
    for sid, frames in enumerate(sessions):
        write_submap_metadata(cfg.data_dir / str(sid), frames, intr, cfg.anchor_to_first_body)
        _log_edge_stats(sid, frames)
    export_images(cfg, sessions)
    orders = write_orders_file(cfg.output_root, cfg.scene, list(range(len(sessions))))
    logger.info("orders file: %s", orders)
    return cfg.data_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="per-bag YAML config")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing data dir")
    parser.add_argument("--max_frames", type=int, default=None, help="only use the first N images (smoke runs)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = convert(BagConversionConfig.from_yaml(args.config), args.overwrite, args.max_frames)
    logger.info("done: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
