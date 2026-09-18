import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "third_party" / "litevloc_code" / "python"))

utils_map_merging = pytest.importorskip("utils_map_merging")
parse_arguments = utils_map_merging.parse_arguments

LEGACY = ["--dataset_root", "/d", "--scene", "s00000", "--order_index", "0", "--method", "m"]


def test_legacy_args_still_parse():
    args = parse_arguments(LEGACY)
    assert args.scene == "s00000" and args.submap_list is None
    assert args.step_dir_style == "cumulative" and args.start_step == 0


def test_legacy_requires_dataset_fields():
    with pytest.raises(SystemExit):
        parse_arguments(["--scene", "s", "--order_index", "0", "--method", "m"])


def test_submap_list_mode(tmp_path: Path):
    args = parse_arguments(["--submap_list", str(tmp_path / "in.txt"), "--result_dir", str(tmp_path / "out"),
                            "--step_dir_style", "indexed"])
    assert args.dataset_root is None and args.method == "console"
    assert args.step_dir_style == "indexed"


def test_submap_list_requires_result_dir(tmp_path: Path):
    with pytest.raises(SystemExit):
        parse_arguments(["--submap_list", str(tmp_path / "in.txt")])


def test_append_requires_start_step(tmp_path: Path):
    base = ["--submap_list", "x", "--result_dir", "y", "--append_from", str(tmp_path)]
    with pytest.raises(SystemExit):
        parse_arguments(base)
    assert parse_arguments(base + ["--start_step", "3"]).start_step == 3
