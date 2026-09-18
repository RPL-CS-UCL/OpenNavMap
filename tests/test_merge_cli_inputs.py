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


# --- run_incremental_merge helpers (Task 12) -------------------------------

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402

SYNTHETIC = REPO / "python" / "visualization" / "example_data" / "synthetic_map"


@pytest.fixture(scope="module")
def pipeline():
    pytest.importorskip("gtsam")
    pytest.importorskip("torch")
    return pytest.importorskip("map_merge_pipeline")


def test_step_dir_name(pipeline):
    assert pipeline.step_dir_name("cumulative", "merge", 0, "0") == "merge_0"
    assert pipeline.step_dir_name("cumulative", "merge_0", 1, "1") == "merge_0_1"
    assert pipeline.step_dir_name("indexed", "whatever", 7, "ses_x") == "merge_007_ses_x"


def test_resolve_submap_dirs_from_list(pipeline, tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    lst = tmp_path / "in.txt"
    lst.write_text(f"# comment\n{a}\n\n{b}\n")
    args = argparse.Namespace(submap_list=str(lst), max_submaps=None)
    assert pipeline.resolve_submap_dirs(args) == [("a", a), ("b", b)]
    args.max_submaps = 1
    assert pipeline.resolve_submap_dirs(args) == [("a", a)]


def test_resolve_submap_dirs_missing(pipeline, tmp_path: Path):
    lst = tmp_path / "in.txt"
    lst.write_text(str(tmp_path / "nope") + "\n")
    with pytest.raises(FileNotFoundError):
        pipeline.resolve_submap_dirs(argparse.Namespace(submap_list=str(lst), max_submaps=None))


def test_resolve_submap_dirs_legacy(pipeline, tmp_path: Path):
    (tmp_path / "s_orders.txt").write_text("1 0\n0 1\n")
    (tmp_path / "s_aria_data_390" / "0").mkdir(parents=True)
    (tmp_path / "s_aria_data_390" / "1").mkdir()
    args = argparse.Namespace(submap_list=None, dataset_root=str(tmp_path), scene="s", order_index=1,
                              data_dir=None, max_submaps=None)
    assert [sid for sid, _ in pipeline.resolve_submap_dirs(args)] == ["0", "1"]


def test_resolve_result_dir(pipeline, tmp_path: Path):
    args = argparse.Namespace(result_dir=str(tmp_path / "r"))
    assert pipeline.resolve_result_dir(args) == tmp_path / "r"
    args = argparse.Namespace(result_dir=None, dataset_root=str(tmp_path), output_root=None, scene="s",
                              order_index=0, method="m", use_iqa=True, use_ig=False, use_td=True)
    assert pipeline.resolve_result_dir(args) == tmp_path / "s_results_in_m_iqatd"


def test_validate_append_map_dir(pipeline, tmp_path: Path):
    good = tmp_path / "good"
    shutil.copytree(SYNTHETIC, good)
    pipeline.validate_append_map_dir(good)  # synthetic_map is consecutive and complete
    first_img = sorted((good / "seq").glob("*.jpg"))[0]
    first_img.unlink()
    with pytest.raises(ValueError, match=first_img.name):
        pipeline.validate_append_map_dir(good)


def test_update_finalmap_link(pipeline, tmp_path: Path):
    (tmp_path / "merge_000_a").mkdir()
    (tmp_path / "merge_001_b").mkdir()
    pipeline.update_finalmap_link(tmp_path, tmp_path / "merge_000_a")
    pipeline.update_finalmap_link(tmp_path, tmp_path / "merge_001_b")
    assert (tmp_path / "merge_finalmap").resolve() == (tmp_path / "merge_001_b").resolve()


def test_save_dmatrix_raw(pipeline, tmp_path: Path):
    D = np.random.rand(3, 5).astype(np.float32)
    pipeline.save_dmatrix_raw(tmp_path, D, [10, 11, 12], [100, 101, 102, 103, 104])
    back = np.load(tmp_path / "D_matrix.npy")
    assert back.dtype == np.float16 and back.shape == (3, 5)
    assert np.allclose(back.astype(np.float32), D, atol=1e-3)
    axes = json.loads((tmp_path / "D_matrix_axes.json").read_text())
    assert axes == {"rows": "db", "row_node_ids": [10, 11, 12], "cols": "query",
                    "col_node_ids": [100, 101, 102, 103, 104]}
