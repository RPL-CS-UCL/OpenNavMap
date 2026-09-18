from pathlib import Path

import pytest

from rosbag_convert.config import BagConversionConfig, ConfigError


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(body)
    return path


def test_from_yaml_reads_fields_and_defaults(tmp_path):
    cfg_path = _write_yaml(tmp_path, """
bag_path: /data/x.bag
output_root: /out
kf_trans_thresh_m: 3.9
kf_rot_thresh_deg: 60.0
""")
    cfg = BagConversionConfig.from_yaml(cfg_path)
    assert cfg.bag_path == "/data/x.bag"
    assert cfg.image_topic == "/odin1/image/undistorted"
    assert cfg.num_sessions == 3
    assert cfg.data_dir_name == "s00000_odin_data_390"
    assert cfg.data_dir == Path("/out/s00000_odin_data_390")


def test_from_yaml_rejects_unknown_key(tmp_path):
    cfg_path = _write_yaml(tmp_path, "bag_path: a\noutput_root: b\nfoo: 1\n")
    with pytest.raises(ConfigError, match="foo"):
        BagConversionConfig.from_yaml(cfg_path)


def test_from_yaml_requires_bag_path(tmp_path):
    cfg_path = _write_yaml(tmp_path, "output_root: b\n")
    with pytest.raises(ConfigError, match="bag_path"):
        BagConversionConfig.from_yaml(cfg_path)


def test_static_extrinsic_required_when_source_is_odometry(tmp_path):
    cfg_path = _write_yaml(tmp_path, """
bag_path: a
output_root: b
camera_pose_source: odometry_static_extrinsic
""")
    with pytest.raises(ConfigError, match="T_body_camera"):
        BagConversionConfig.from_yaml(cfg_path)
