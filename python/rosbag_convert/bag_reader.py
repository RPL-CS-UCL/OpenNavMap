"""Read images, poses and intrinsics from a ROS1 bag without ROS.

This is the only module that imports ``rosbags``. Every reader opens the
bag itself so callers can make independent passes: pass 1 reads the small
messages (stamps, poses, intrinsics), pass 2 decodes only the images that
were selected as keyframes, which keeps memory flat for a 27 GB bag.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.spatial.transform import Rotation

NS_PER_S = 1_000_000_000
SUPPORTED_ENCODINGS = ("bgr8", "rgb8", "mono8")


class BagReadError(RuntimeError):
    """Raised when a topic is missing or a message cannot be interpreted."""


@dataclass
class Intrinsics:
    """Pinhole intrinsics of the (already undistorted) camera."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class Trajectory:
    """Time-stamped 4x4 poses sorted by stamp (seconds)."""
    stamps: np.ndarray
    poses: np.ndarray

    def __len__(self) -> int:
        return len(self.stamps)


def stamp_to_ns(header) -> int:
    return int(header.stamp.sec) * NS_PER_S + int(header.stamp.nanosec)


def ns_to_seconds(ns: int) -> float:
    return ns / NS_PER_S


def open_bag(bag_path: Path) -> Reader:
    """Return an unopened ``Reader``; use it as a context manager (``with``)."""
    return Reader(Path(bag_path))


def typestore_for(reader: Reader):
    """Typestore with every message definition recorded in the (opened) bag.

    The stock ROS1 typestore lacks e.g. ``tf2_msgs/msg/TFMessage``; the
    bag records every message definition it used, so we register those.
    """
    typestore = get_typestore(Stores.ROS1_NOETIC)
    for conn in reader.connections:
        if conn.msgtype not in typestore.types:
            typestore.register(get_types_from_msg(conn.msgdef, conn.msgtype))
    return typestore


def _connections(reader: Reader, topic: str) -> list:
    conns = [c for c in reader.connections if c.topic == topic]
    if not conns:
        raise BagReadError(f"Topic '{topic}' not found in bag")
    return conns


def _pose_to_matrix(position, orientation) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, 3] = (position.x, position.y, position.z)
    tf[:3, :3] = Rotation.from_quat(
        (orientation.x, orientation.y, orientation.z, orientation.w)).as_matrix()
    return tf


def _as_trajectory(stamps: List[float], poses: List[np.ndarray]) -> Trajectory:
    order = np.argsort(stamps, kind="stable")
    return Trajectory(np.asarray(stamps, dtype=np.float64)[order],
                      np.asarray(poses, dtype=np.float64)[order])


def read_camera_info(bag_path: Path, topic: str) -> Intrinsics:
    """Return intrinsics from the first CameraInfo message.

    Raises:
        BagReadError: If the topic is absent or the image is not rectified
            (non-zero distortion), since the map format is pinhole-only.
    """
    with open_bag(bag_path) as reader:
        typestore = typestore_for(reader)
        for conn, _, raw in reader.messages(connections=_connections(reader, topic)):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            # ROS1 definitions spell the arrays D/K, ROS2-style ones d/k
            dist = msg.D if hasattr(msg, "D") else msg.d
            k_flat = msg.K if hasattr(msg, "K") else msg.k
            if np.any(np.asarray(dist) != 0):
                raise BagReadError(f"'{topic}' has non-zero distortion; map format needs undistorted images")
            k = np.asarray(k_flat).reshape(3, 3)
            return Intrinsics(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]),
                              int(msg.width), int(msg.height))
    raise BagReadError(f"No message on '{topic}'")


def read_odometry(bag_path: Path, topic: str) -> Trajectory:
    """Read ``nav_msgs/Odometry`` as T_odom_body poses."""
    stamps, poses = [], []
    with open_bag(bag_path) as reader:
        typestore = typestore_for(reader)
        for conn, _, raw in reader.messages(connections=_connections(reader, topic)):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            stamps.append(ns_to_seconds(stamp_to_ns(msg.header)))
            poses.append(_pose_to_matrix(msg.pose.pose.position, msg.pose.pose.orientation))
    return _as_trajectory(stamps, poses)


def read_tf_pairs(bag_path: Path, topic: str, parent: str, child: str) -> Trajectory:
    """Collect every ``parent -> child`` transform on a TF topic as T_parent_child."""
    stamps, poses = [], []
    with open_bag(bag_path) as reader:
        typestore = typestore_for(reader)
        for conn, _, raw in reader.messages(connections=_connections(reader, topic)):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            for tf in msg.transforms:
                if tf.header.frame_id == parent and tf.child_frame_id == child:
                    stamps.append(ns_to_seconds(stamp_to_ns(tf.header)))
                    poses.append(_pose_to_matrix(tf.transform.translation, tf.transform.rotation))
    if not stamps:
        raise BagReadError(f"No '{parent}' -> '{child}' transform on '{topic}'")
    return _as_trajectory(stamps, poses)


def read_image_stamps_ns(bag_path: Path, topic: str) -> np.ndarray:
    """Header stamps (integer ns) of every image, in bag order, without decoding pixels."""
    stamps = []
    with open_bag(bag_path) as reader:
        typestore = typestore_for(reader)
        for conn, _, raw in reader.messages(connections=_connections(reader, topic)):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            stamps.append(stamp_to_ns(msg.header))
    return np.asarray(stamps, dtype=np.int64)


def _decode_image(msg) -> np.ndarray:
    if msg.encoding not in SUPPORTED_ENCODINGS:
        raise BagReadError(f"Unsupported image encoding '{msg.encoding}'")
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    channels = 1 if msg.encoding == "mono8" else 3
    data = np.asarray(msg.data, dtype=np.uint8).reshape(h, step)[:, : w * channels]
    img = data.reshape(h, w, channels)
    if msg.encoding == "rgb8":
        img = img[:, :, ::-1]
    elif msg.encoding == "mono8":
        img = np.repeat(img, 3, axis=2)
    return np.ascontiguousarray(img)


def iter_images(bag_path: Path, topic: str, wanted_ns: Set[int]) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(stamp_ns, bgr_image)`` for the images whose stamp is in ``wanted_ns``.

    Stops early once every wanted stamp has been seen.
    """
    remaining = set(wanted_ns)
    with open_bag(bag_path) as reader:
        typestore = typestore_for(reader)
        for conn, _, raw in reader.messages(connections=_connections(reader, topic)):
            if not remaining:
                return
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            ns = stamp_to_ns(msg.header)
            if ns in remaining:
                remaining.discard(ns)
                yield ns, _decode_image(msg)
