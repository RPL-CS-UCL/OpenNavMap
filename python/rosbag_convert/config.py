"""Per-bag conversion settings loaded from a YAML file.

One YAML per rosbag: topic names, TF frame names, keyframe thresholds and
how to split the recording into simulated sessions. Everything the
converter needs to know about a specific recording lives here so that a
new sensor or a new bag never requires a code change.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

CAMERA_POSE_FROM_TF = "tf"
CAMERA_POSE_FROM_ODOMETRY = "odometry_static_extrinsic"
SPLIT_EQUAL_DISTANCE = "equal_distance"
SPLIT_TIME_BOUNDARIES = "time_boundaries"


class ConfigError(ValueError):
    """Raised when the YAML is missing required keys or has inconsistent values."""


@dataclass
class BagConversionConfig:
    """Settings for converting one rosbag into multi-session map directories.

    Attributes:
        bag_path: Path to the ROS1 ``.bag`` file.
        output_root: Dataset root, e.g. ``.../map_multisession_eval/cmt_szs``.
        scene: Scene prefix used in directory and orders-file names.
        sensor_tag: Sensor name embedded in the data directory name.
        image_topic: ``sensor_msgs/Image`` topic (bgr8 / rgb8 / mono8).
        camera_info_topic: ``sensor_msgs/CameraInfo`` topic; the first
            message provides the intrinsics for the whole bag.
        odometry_topic: ``nav_msgs/Odometry`` topic giving the body pose
            in the odometry frame.
        tf_topic: ``tf2_msgs/TFMessage`` topic.
        odom_frame: TF frame id of the odometry (world) frame.
        body_frame: TF frame id of the odometry child (robot body).
        camera_frame: TF frame id of the camera optical frame.
        camera_pose_source: ``"tf"`` reads ``odom_frame -> camera_frame``
            directly from TF; ``"odometry_static_extrinsic"`` composes the
            odometry with ``T_body_camera_*`` below.
        T_body_camera_translation: Fallback extrinsic translation [m].
        T_body_camera_quat_xyzw: Fallback extrinsic rotation (x, y, z, w).
        sync_tolerance_s: Max |image stamp - pose stamp| to accept a pair.
        kf_trans_thresh_m: Keep a frame once the robot moved this far.
        kf_rot_thresh_deg: ... or turned this much since the last keyframe.
        num_sessions: Number of simulated sessions.
        split_mode: ``"equal_distance"`` or ``"time_boundaries"``.
        split_time_boundaries_s: Cut times relative to the first synced
            frame, only for ``"time_boundaries"`` (``num_sessions - 1`` values).
        anchor_to_first_body: Express each session's ``poses.txt`` in the
            frame of its first keyframe's body pose (gravity-aligned, like
            the released Aria data).
        jpeg_quality: OpenCV JPEG quality for exported keyframes.
    """

    bag_path: str
    output_root: str
    scene: str = "s00000"
    sensor_tag: str = "odin"
    image_topic: str = "/odin1/image/undistorted"
    camera_info_topic: str = "/odin1/camera_info"
    odometry_topic: str = "/odin1/odometry"
    tf_topic: str = "/tf"
    odom_frame: str = "odom"
    body_frame: str = "imu"
    camera_frame: str = "camera_0"
    camera_pose_source: str = CAMERA_POSE_FROM_TF
    T_body_camera_translation: Optional[List[float]] = None
    T_body_camera_quat_xyzw: Optional[List[float]] = None
    sync_tolerance_s: float = 0.02
    kf_trans_thresh_m: float = 3.9
    kf_rot_thresh_deg: float = 60.0
    num_sessions: int = 3
    split_mode: str = SPLIT_EQUAL_DISTANCE
    split_time_boundaries_s: Optional[List[float]] = None
    anchor_to_first_body: bool = True
    jpeg_quality: int = 95

    @property
    def data_dir_name(self) -> str:
        """Directory name following the ``<scene>_<sensor>_data_<dist>`` convention."""
        dist_tag = int(round(self.kf_trans_thresh_m * 100))
        return f"{self.scene}_{self.sensor_tag}_data_{dist_tag:03d}"

    @property
    def data_dir(self) -> Path:
        return Path(self.output_root) / self.data_dir_name

    @classmethod
    def from_yaml(cls, path: Path) -> "BagConversionConfig":
        """Load and validate a config from YAML.

        Raises:
            ConfigError: On unknown keys, missing required keys or an
                inconsistent combination of values.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"Unknown config keys in {path}: {unknown}")
        for key in ("bag_path", "output_root"):
            if key not in raw:
                raise ConfigError(f"Missing required key '{key}' in {path}")
        cfg = cls(**raw)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if self.camera_pose_source not in (CAMERA_POSE_FROM_TF, CAMERA_POSE_FROM_ODOMETRY):
            raise ConfigError(f"camera_pose_source must be '{CAMERA_POSE_FROM_TF}' or "
                              f"'{CAMERA_POSE_FROM_ODOMETRY}', got '{self.camera_pose_source}'")
        if self.camera_pose_source == CAMERA_POSE_FROM_ODOMETRY and (
                self.T_body_camera_translation is None or self.T_body_camera_quat_xyzw is None):
            raise ConfigError("T_body_camera_translation and T_body_camera_quat_xyzw are required "
                              f"when camera_pose_source is '{CAMERA_POSE_FROM_ODOMETRY}'")
        if self.split_mode not in (SPLIT_EQUAL_DISTANCE, SPLIT_TIME_BOUNDARIES):
            raise ConfigError(f"Unknown split_mode '{self.split_mode}'")
        if self.split_mode == SPLIT_TIME_BOUNDARIES:
            n_bounds = len(self.split_time_boundaries_s or [])
            if n_bounds != self.num_sessions - 1:
                raise ConfigError(
                    f"split_time_boundaries_s needs {self.num_sessions - 1} values, got {n_bounds}")
        if self.num_sessions < 1:
            raise ConfigError("num_sessions must be >= 1")
