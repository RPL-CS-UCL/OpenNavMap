import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np

import benchmark_map_merge.run_baseline as run_baseline
from benchmark_map_merge.run_baseline import (
    _classify_merge_failure,
    _build_sfm_summary,
    _has_colmap_model_files,
    _write_sfm_summary,
)


class _FakeModel:
    images = {1: object(), 2: object()}
    points3D = {1: object(), 2: object(), 3: object()}


def test_build_sfm_summary_includes_result_files(tmp_path: Path) -> None:
    rrd_path = tmp_path / "sfm_reconstruction.rrd"
    topdown_path = tmp_path / "topdown_poses.png"
    pairs_path = tmp_path / "pairs-sfm.txt"
    rrd_path.write_bytes(b"rrd")
    topdown_path.write_bytes(b"png-data")
    pairs_path.write_text("a b\nb c\n")

    summary = _build_sfm_summary(
        _FakeModel(),
        sampled_frames=11,
        total_frames=20,
        sfm_pairs_path=pairs_path,
        sfm_rrd_path=rrd_path,
        topdown_path=topdown_path,
    )

    assert summary["num_registered_images"] == 2
    assert summary["num_points3D"] == 3
    assert summary["num_sampled_frames"] == 11
    assert summary["num_total_ref_frames"] == 20
    assert summary["num_sfm_pairs"] == 2
    assert summary["sfm_reconstruction_rrd"]["size_bytes"] == 3
    assert summary["topdown_poses_png"]["size_bytes"] == 8


def test_write_sfm_summary_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "logs" / "sfm_summary.json"
    _write_sfm_summary({"num_registered_images": 2}, output_path)

    assert json.loads(output_path.read_text()) == {"num_registered_images": 2}


def test_has_colmap_model_files_requires_all_binary_files(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    empty_dir = tmp_path / "empty"
    partial_dir = tmp_path / "partial"
    complete_dir = tmp_path / "complete"
    directory_file_dir = tmp_path / "directory_file"

    empty_dir.mkdir()
    partial_dir.mkdir()
    complete_dir.mkdir()
    directory_file_dir.mkdir()
    for file_name in ("cameras.bin", "images.bin"):
        (partial_dir / file_name).write_bytes(b"")
    for file_name in ("cameras.bin", "images.bin", "points3D.bin"):
        (complete_dir / file_name).write_bytes(b"")
    for file_name in ("cameras.bin", "images.bin"):
        (directory_file_dir / file_name).write_bytes(b"")
    (directory_file_dir / "points3D.bin").mkdir()

    assert not _has_colmap_model_files(missing_dir)
    assert not _has_colmap_model_files(empty_dir)
    assert not _has_colmap_model_files(partial_dir)
    assert not _has_colmap_model_files(directory_file_dir)
    assert _has_colmap_model_files(complete_dir)


def test_sfm_result_root_includes_sfm_ba_suffix() -> None:
    result_root = run_baseline._build_result_root(
        Path("/tmp/dataset"),
        "in_2sub",
        "hloc_sfm_netvlad_splg",
        sfm_sample_dist=0.25,
    )

    assert result_root.name == "s00000_results_in_2sub_hloc_sfm_netvlad_splg_025_value2"


def test_sfm_only_result_root_uses_compact_name() -> None:
    result_root = run_baseline._build_result_root(
        Path("/tmp/dataset"),
        "in_2sub",
        "hloc_sfm_netvlad_splg",
        sfm_sample_dist=0.25,
        sfm_only=True,
    )

    assert result_root.name == "s00000_sfm_netvlad_splg_025"


def test_sfm_only_disk_result_root_uses_method_tag() -> None:
    result_root = run_baseline._build_result_root(
        Path("/tmp/dataset"),
        "in_2sub",
        "hloc_sfm_netvlad_disk_dilg",
        sfm_sample_dist=0.25,
        sfm_only=True,
    )

    assert result_root.name == "s00000_sfm_netvlad_disk_dilg_025"


def test_result_root_default_full_data() -> None:
    result_root = run_baseline._build_result_root(
        Path("/tmp/dataset"),
        "in_2sub",
        "hloc_sfm_netvlad_splg",
        sfm_sample_dist=0.25,
    )

    assert "025" in result_root.name


def test_result_root_appends_threshold_value_suffix() -> None:
    result_root = run_baseline._build_result_root(
        Path("/tmp/dataset"),
        "in",
        "hloc_sfm_netvlad_splg",
        sfm_sample_dist=0.25,
        num_retrieval=10,
        geo_verify_min_matches=150,
        pnp_min_inliers=50,
    )

    assert result_root.name.endswith("_value2")


def test_result_root_uses_explicit_threshold_value0_suffix() -> None:
    result_root = run_baseline._build_result_root(
        Path("/tmp/dataset"),
        "in",
        "hloc_sfm_netvlad_disk_dilg",
        sfm_sample_dist=0.25,
        num_retrieval=10,
        geo_verify_min_matches=300,
        pnp_min_inliers=70,
    )

    assert result_root.name == "s00000_results_in_hloc_sfm_netvlad_disk_dilg_025_value0"


def test_result_root_uses_explicit_threshold_value1_suffix() -> None:
    result_root = run_baseline._build_result_root(
        Path("/tmp/dataset"),
        "in",
        "hloc_sfm_netvlad_disk_dilg",
        sfm_sample_dist=0.25,
        num_retrieval=10,
        geo_verify_min_matches=400,
        pnp_min_inliers=110,
    )

    assert result_root.name == "s00000_results_in_hloc_sfm_netvlad_disk_dilg_025_value1"


def test_cli_has_submap_sfm_and_submap_merge_flags() -> None:
    import sys
    result = subprocess.run(
        [sys.executable, "run_baseline.py", "--help"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert "--submap-sfm" in result.stdout
    assert "--submap-merge" in result.stdout
    assert "--sfm-ba-iter" in result.stdout
    assert "--only-build-sfm" not in result.stdout
    assert "--vio-ba-iter" not in result.stdout


def test_run_baseline_script_help_lists_supported_envs() -> None:
    script_path = Path(__file__).parent.parent / "scripts" / "run_baseline.sh"

    result = subprocess.run(
        ["bash", str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--env" in result.stdout
    assert "ucl_campus_aria" in result.stdout
    assert "hkust_campus" in result.stdout
    assert "vineyard" in result.stdout


def test_run_baseline_script_uses_env_dataset_root(tmp_path: Path) -> None:
    script_path = Path(__file__).parent.parent / "scripts" / "run_baseline.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bc = fake_bin / "bc"
    fake_bc.write_text("#!/bin/sh\nprintf '25\\n'\n")
    fake_bc.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--mode",
            "merge",
            "--env",
            "hkust_campus",
            "--prebuilt-sfm-root",
            str(tmp_path / "sfm"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert "--dataset-root /Titan/dataset/data_opennavmap/map_multisession_eval/hkust_campus" in result.stdout
    assert "--dataset-name" not in result.stdout


def test_run_baseline_script_supports_disk_sfm_method(tmp_path: Path) -> None:
    script_path = Path(__file__).parent.parent / "scripts" / "run_baseline.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bc = fake_bin / "bc"
    fake_bc.write_text("#!/bin/sh\nprintf '25\\n'\n")
    fake_bc.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--mode",
            "sfm",
            "--method",
            "hloc_sfm_netvlad_disk_dilg",
            "--env",
            "vineyard",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "s00000_sfm_netvlad_disk_dilg_025" in result.stdout
    assert "--method hloc_sfm_netvlad_disk_dilg" in result.stdout


def test_run_baseline_script_passes_clean_work(tmp_path: Path) -> None:
    script_path = Path(__file__).parent.parent / "scripts" / "run_baseline.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bc = fake_bin / "bc"
    fake_bc.write_text("#!/bin/sh\nprintf '25\\n'\n")
    fake_bc.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--mode",
            "merge",
            "--env",
            "vineyard",
            "--prebuilt-sfm-root",
            str(tmp_path / "sfm"),
            "--clean-work",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "--clean-work" in result.stdout


def test_run_baseline_script_result_dir_uses_threshold_value_suffix(tmp_path: Path) -> None:
    script_path = Path(__file__).parent.parent / "scripts" / "run_baseline.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bc = fake_bin / "bc"
    fake_bc.write_text("#!/bin/sh\nprintf '25\\n'\n")
    fake_bc.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--mode",
            "merge",
            "--env",
            "vineyard",
            "--prebuilt-sfm-root",
            str(tmp_path / "sfm"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "s00000_results_in_hloc_sfm_netvlad_splg_025_value2" in result.stdout


def test_run_baseline_script_result_dir_uses_explicit_threshold_value0_suffix(tmp_path: Path) -> None:
    script_path = Path(__file__).parent.parent / "scripts" / "run_baseline.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bc = fake_bin / "bc"
    fake_bc.write_text("#!/bin/sh\nprintf '25\\n'\n")
    fake_bc.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--mode",
            "merge",
            "--method",
            "hloc_sfm_netvlad_disk_dilg",
            "--env",
            "vineyard",
            "--prebuilt-sfm-root",
            str(tmp_path / "sfm"),
            "--num-retrieval",
            "10",
            "--geo-verify-min-matches",
            "300",
            "--pnp-min-inliers",
            "70",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "s00000_results_in_hloc_sfm_netvlad_disk_dilg_025_value0" in result.stdout
    assert "--num-retrieval 10" in result.stdout
    assert "--geo-verify-min-matches 300" in result.stdout
    assert "--pnp-min-inliers 70" in result.stdout


def test_run_evaluation_script_supports_output_dir(tmp_path: Path) -> None:
    script_path = Path(__file__).parent.parent / "scripts" / "run_evaluation.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$TEST_ARGS_FILE\"\n"
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TEST_ARGS_FILE"] = str(tmp_path / "args.txt")
    env["PYTHON"] = str(fake_python)

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--config",
            "OpenNavMap_map_merge.yaml",
            "--output-dir",
            str(tmp_path / "report_value1"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert f"Report dir : {tmp_path / 'report_value1'}" in result.stdout
    assert f"--output_dir={tmp_path / 'report_value1'}" in (tmp_path / "args.txt").read_text()


def test_run_order_reuses_cached_incremental_sfm(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    data_root = dataset_root / "s00000_aria_data_000"
    ref_dir = data_root / "s0"
    inc_dir = data_root / "s1"
    for submap_dir in (ref_dir, inc_dir):
        (submap_dir / "seq").mkdir(parents=True)

    (dataset_root / "s00000_orders.txt").write_text("s0 s1\n")

    result_root = run_baseline._build_result_root(
        dataset_root,
        "in_2sub",
        "hloc_sfm_netvlad_splg",
        sfm_sample_dist=0.25,
    )
    cached_merge_dir = result_root / "merge_s0_s1" / "submap_disc_0"
    cached_sfm_dir = cached_merge_dir / "sfm"
    cached_sfm_dir.mkdir(parents=True)
    for file_name in ("cameras.bin", "images.bin", "points3D.bin"):
        (cached_sfm_dir / file_name).write_bytes(b"cache")
    (cached_merge_dir / "poses.txt").write_text(
        "seq/000000.color.jpg 1 0 0 0 0 0 0\n"
        "seq/000001.color.jpg 1 0 0 0 1 0 0\n"
        "seq/000002.color.jpg 1 0 0 0 2 0 0\n"
        "seq/000003.color.jpg 1 0 0 0 3 0 0\n"
        "seq/000004.color.jpg 1 0 0 0 4 0 0\n"
        "seq/000005.color.jpg 1 0 0 0 5 0 0\n"
    )
    merge_stats_path = result_root / "logs" / "submap1_merge_stats.json"
    merge_stats_path.parent.mkdir(parents=True)
    merge_stats_path.write_text(json.dumps({
        "num_pnp_success": 5,
        "sampled_inc_images": [f"seq/{i:06d}.color.jpg" for i in range(5)],
        "model_name_to_orig": {},
    }))

    def fake_get_image_list(submap_dir: Path) -> List[str]:
        if submap_dir == ref_dir:
            return ["seq/000000.color.jpg"]
        return [f"seq/{i:06d}.color.jpg" for i in range(5)]

    fake_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(run_baseline, "_get_image_list", fake_get_image_list)
    monkeypatch.setattr(
        run_baseline,
        "read_poses",
        lambda path: {
            f"seq/{i:06d}.color.jpg": fake_pose + np.array([0, 0, 0, 0, i, 0, 0])
            for i in range(6)
        },
    )
    monkeypatch.setattr(
        run_baseline,
        "read_timestamps",
        lambda path: {f"seq/{i:06d}.color.jpg": float(i) for i in range(5)},
    )
    monkeypatch.setattr(
        run_baseline,
        "read_intrinsics",
        lambda path: {f"seq/{i:06d}.color.jpg": np.ones(6) for i in range(5)},
    )
    monkeypatch.setattr(
        run_baseline,
        "read_gps",
        lambda path: {f"seq/{i:06d}.color.jpg": np.ones(5) for i in range(5)},
    )
    monkeypatch.setattr(run_baseline, "read_edges_odom", lambda path: [])
    monkeypatch.setattr(run_baseline, "save_sfm_vis", lambda *args, **kwargs: args[1].write_bytes(b"rrd"))
    monkeypatch.setattr(run_baseline, "save_topdown_pose_viz", lambda *args, **kwargs: args[3].write_bytes(b"png"))
    monkeypatch.setattr(run_baseline, "export_to_eval_structure", lambda *args, **kwargs: None)

    class FakeImage:
        registered = True
        image_id = 1
        name = "seq/000000.color.jpg"

    class FakeReconstruction:
        read_paths: List[str] = []
        write_paths: List[str] = []

        def __init__(self) -> None:
            self.images = {1: FakeImage()}
            self.points3D = {}

        def read_binary(self, path: str) -> None:
            self.read_paths.append(path)

        def write_binary(self, path: str) -> None:
            self.write_paths.append(path)

    class FakeSfmMerger:
        last_sfm_sampled_frames = 1
        local_feature_name = "feats-sp.h5"
        feature_conf = {"output": "feats-sp"}
        retrieval_conf = {"output": "global-feats-netvlad"}
        matcher_conf = {"output": "matches-sp-lightglue"}

        def __init__(self, work_dir: Path, *args, **kwargs) -> None:
            self.work_dir = work_dir

        def build_submap_sfm(self, *args, **kwargs) -> FakeReconstruction:
            return FakeReconstruction()

        def merge_model_with_se3(self, *args, **kwargs):
            raise AssertionError("merge_model_with_se3 should not run on cache hit")

        @staticmethod
        def extract_w2c_vec_from_image(image: FakeImage) -> np.ndarray:
            return fake_pose

    monkeypatch.setattr(run_baseline, "HlocSfmMapMerger", FakeSfmMerger)
    monkeypatch.setattr(
        run_baseline,
        "pycolmap",
        SimpleNamespace(Reconstruction=FakeReconstruction),
        raising=False,
    )

    run_baseline.run_order(
        dataset_root=dataset_root,
        method="hloc_sfm_netvlad_splg",
        order_index=0,
        max_submaps=2,
        skip_eval_export=True,
        overwrite=False,
        submap_merge=True,
    )

    assert FakeReconstruction.read_paths == [str(cached_sfm_dir)]
    assert "[Cache]" in (
        result_root / "logs" / "pipeline.log"
    ).read_text()
    summary = json.loads(merge_stats_path.read_text())
    assert summary["num_pnp_success"] == 5


def test_sfm_summary_can_store_merge_params(tmp_path: Path) -> None:
    output_path = tmp_path / "logs" / "sfm_summary.json"
    summary = {
        "num_registered_images": 2,
        "merge_params": {
            "num_retrieval": 10,
            "geo_verify_min_matches": 400,
            "pnp_min_inliers": 110,
        },
    }

    _write_sfm_summary(summary, output_path)

    loaded = json.loads(output_path.read_text())
    assert loaded["merge_params"]["num_retrieval"] == 10
    assert loaded["merge_params"]["geo_verify_min_matches"] == 400
    assert loaded["merge_params"]["pnp_min_inliers"] == 110


def test_classify_merge_failure_no_pnp_success() -> None:
    message, summary_error = _classify_merge_failure({
        "num_pnp_success": 0,
        "num_se3_inliers": 0,
        "error": "SE(3) estimation failed",
    })

    assert message == "no PnP success"
    assert summary_error == "no PnP success"


def test_classify_merge_failure_insufficient_pnp_anchors() -> None:
    message, summary_error = _classify_merge_failure({
        "num_pnp_success": 1,
        "num_se3_inliers": 0,
        "error": "SE(3) estimation failed",
    })

    assert message == "insufficient PnP anchors for SE(3)"
    assert summary_error == "insufficient PnP anchors for SE(3)"


def test_classify_merge_failure_se3_after_pnp() -> None:
    message, summary_error = _classify_merge_failure({
        "num_pnp_success": 4,
        "num_se3_inliers": 0,
        "error": "SE(3) estimation failed",
    })

    assert message == "SE(3) estimation failed after PnP"
    assert summary_error == "SE(3) estimation failed after PnP"


def test_run_order_exports_eval_method_name_with_value_suffix(
    monkeypatch, tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    data_root = dataset_root / "s00000_aria_data_000"
    ref_dir = data_root / "s0"
    ref_dir.mkdir(parents=True)
    (dataset_root / "s00000_orders.txt").write_text("s0\n")
    (ref_dir / "intrinsics.txt").write_text(
        "seq/000000.color.jpg 444.0 445.0 511.5 287.5 1024 576\n"
    )

    fake_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(run_baseline, "_get_image_list", lambda path: ["seq/000000.color.jpg"])
    monkeypatch.setattr(run_baseline, "read_poses", lambda path: {"seq/000000.color.jpg": fake_pose})
    monkeypatch.setattr(run_baseline, "read_timestamps", lambda path: {"seq/000000.color.jpg": 0.0})
    monkeypatch.setattr(run_baseline, "read_intrinsics", lambda path: {"seq/000000.color.jpg": np.ones(6)})
    monkeypatch.setattr(run_baseline, "read_gps", lambda path: {"seq/000000.color.jpg": np.ones(5)})
    monkeypatch.setattr(run_baseline, "read_edges_odom", lambda path: [])
    monkeypatch.setattr(run_baseline, "save_sfm_vis", lambda *args, **kwargs: args[1].write_bytes(b"rrd"))
    monkeypatch.setattr(run_baseline, "save_topdown_pose_viz", lambda *args, **kwargs: args[3].write_bytes(b"png"))
    monkeypatch.setattr(run_baseline, "create_finalmap_symlink", lambda *args, **kwargs: None)

    captured = {}

    def fake_export_to_eval_structure(merge_dir, traj_eval_data_root, dataset_order_name, method_name):
        captured["dataset_order_name"] = dataset_order_name
        captured["method_name"] = method_name
        return traj_eval_data_root / "gt.txt", traj_eval_data_root / "est.txt"

    monkeypatch.setattr(run_baseline, "export_to_eval_structure", fake_export_to_eval_structure)

    class FakeImage:
        registered = True
        image_id = 1
        name = "seq/000000.color.jpg"

    class FakeReconstruction:
        def __init__(self) -> None:
            self.images = {1: FakeImage()}
            self.points3D = {1: object()}

        def write_binary(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeSfmMerger:
        last_sfm_sampled_frames = 1
        local_feature_name = "feats-disk.h5"
        feature_conf = {"output": "feats-disk"}
        retrieval_conf = {"output": "global-feats-netvlad"}
        matcher_conf = {"output": "matches-disk-lightglue"}

        def __init__(self, *args, **kwargs) -> None:
            pass

        def build_submap_sfm(self, *args, **kwargs):
            return FakeReconstruction()

        @staticmethod
        def extract_w2c_vec_from_image(image: FakeImage) -> np.ndarray:
            return fake_pose

    monkeypatch.setattr(run_baseline, "HlocSfmMapMerger", FakeSfmMerger)

    run_baseline.run_order(
        dataset_root=dataset_root,
        method="hloc_sfm_netvlad_disk_dilg",
        order_index=0,
        traj_eval_data_root=tmp_path / "traj_eval",
        skip_eval_export=False,
        overwrite=True,
        submap_merge=True,
        num_retrieval=10,
        geo_verify_min_matches=300,
        pnp_min_inliers=70,
    )

    assert captured["dataset_order_name"] == "dataset_s00000_in"
    assert captured["method_name"] == "hloc_sfm_netvlad_disk_dilg_025_value0"
