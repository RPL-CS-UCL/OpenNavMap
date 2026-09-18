"""Smoke test against the real cmt_szs bag; skipped when the file is absent."""
import os

import numpy as np
import pytest

from rosbag_convert import bag_reader

BAG = "/Titan/dataset/data_opennavmap/raw_vps_data/cmt_szs/go2-record-20260917.bag"
pytestmark = pytest.mark.skipif(not os.path.exists(BAG), reason="real bag not available")


def test_camera_info_matches_known_values():
    intr = bag_reader.read_camera_info(BAG, "/odin1/camera_info")
    assert (intr.width, intr.height) == (1600, 1296)
    assert abs(intr.fx - 738.71) < 0.1 and abs(intr.cy - 654.77) < 0.1


def test_tf_pairs_and_odometry_share_stamps():
    cam = bag_reader.read_tf_pairs(BAG, "/tf", "odom", "camera_0")
    odom = bag_reader.read_odometry(BAG, "/odin1/odometry")
    assert len(cam.stamps) == len(odom.stamps) == 5690
    assert np.all(np.diff(cam.stamps) > 0)
    assert np.max(np.abs(cam.stamps - odom.stamps)) < 1e-3


def test_first_image_decodes_to_bgr():
    stamps = bag_reader.read_image_stamps_ns(BAG, "/odin1/image/undistorted")
    assert len(stamps) == 5690
    ns, img = next(bag_reader.iter_images(BAG, "/odin1/image/undistorted", {int(stamps[0])}))
    assert ns == int(stamps[0])
    assert img.shape == (1296, 1600, 3) and img.dtype == np.uint8
