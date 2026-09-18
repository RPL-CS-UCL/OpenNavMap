import shutil
from pathlib import Path

import pytest

from navmap_console.services.validation import frame_index, validate_submap_dir


@pytest.fixture
def map_copy(tmp_path: Path, synthetic_map: Path) -> Path:
    dst = tmp_path / "000"
    shutil.copytree(synthetic_map, dst)
    return dst


def test_frame_index():
    assert frame_index("seq/000123.color.jpg") == 123
    assert frame_index("seq/abc.jpg") is None


def test_synthetic_map_is_valid(map_copy: Path):
    report = validate_submap_dir(map_copy)
    assert report.ok, report.errors
    assert report.num_frames == 12
    assert report.descriptor_dim == 256
    assert report.has_iqa
    names = {f.name: f.status for f in report.files}
    assert names["poses.txt"] == "ok" and names["gps_data.txt"] == "ok"


def test_missing_gps_is_an_error(map_copy: Path):
    (map_copy / "gps_data.txt").unlink()
    report = validate_submap_dir(map_copy)
    assert not report.ok
    assert any("gps_data.txt" in e for e in report.errors)


def test_short_descriptor_file_is_incomplete(map_copy: Path):
    lines = (map_copy / "database_descriptors.txt").read_text().splitlines()
    (map_copy / "database_descriptors.txt").write_text("\n".join(lines[:-1]) + "\n")
    report = validate_submap_dir(map_copy)
    assert not report.ok
    check = next(f for f in report.files if f.name == "database_descriptors.txt")
    assert check.status == "incomplete" and check.lines == 11


def test_missing_gt_is_optional(map_copy: Path):
    (map_copy / "poses_abs_gt.txt").unlink()
    report = validate_submap_dir(map_copy)
    assert report.ok and not report.has_gt


def test_missing_image_is_an_error(map_copy: Path):
    first = sorted((map_copy / "seq").glob("*.jpg"))[0]
    first.unlink()
    report = validate_submap_dir(map_copy)
    assert not report.ok
    assert any(first.name in e for e in report.errors)


def test_missing_poses_is_fatal(tmp_path: Path):
    report = validate_submap_dir(tmp_path)
    assert not report.ok and report.num_frames == 0
