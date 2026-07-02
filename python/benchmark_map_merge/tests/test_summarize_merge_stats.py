import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_merge_stats.py"
)


def test_summarize_merge_stats_outputs_single_line_summary(tmp_path: Path) -> None:
    stats_path = tmp_path / "submap3_merge_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "submap_idx": 3,
                "num_pnp_sampled": 1336,
                "num_pnp_success": 589,
                "num_se3_inliers": 512,
                "image_matching": {
                    "min_inliers": 120,
                    "geo_verify_tp_fp": {
                        "threshold_f_inliers": 120,
                        "num_pairs_above_thresh": 5236,
                        "num_tp": 4840,
                        "num_fp": 396,
                        "tp_ratio": 0.9244,
                        "fp_ratio": 0.0756,
                    },
                },
                "pnp_error_stats": {
                    "num_error_lt2m": 524,
                    "num_error_ge2m": 65,
                    "ratio_lt2m": 0.8896,
                },
            }
        )
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(stats_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == (
        "submap3 | geo(thr=120, pairs=5236, tp=4840, fp=396, tp_ratio=0.9244) | "
        "pnp(sampled=1336, success=589, se3_inliers=512, <2m=524, >=2m=65, ratio_lt2m=0.8896)"
    )


def test_summarize_merge_stats_uses_na_for_missing_fields(tmp_path: Path) -> None:
    stats_path = tmp_path / "custom_merge_stats.json"
    stats_path.write_text(json.dumps({"num_pnp_success": 2}))

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(stats_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == (
        "custom_merge_stats | geo(thr=n/a, pairs=n/a, tp=n/a, fp=n/a, tp_ratio=n/a) | "
        "pnp(sampled=n/a, success=2, se3_inliers=n/a, <2m=n/a, >=2m=n/a, ratio_lt2m=n/a)"
    )


def test_summarize_merge_stats_outputs_one_line_per_file(tmp_path: Path) -> None:
    stats_a = tmp_path / "submap1_merge_stats.json"
    stats_b = tmp_path / "submap2_merge_stats.json"
    stats_a.write_text(json.dumps({"submap_idx": 1, "num_pnp_success": 5}))
    stats_b.write_text(json.dumps({"submap_idx": 2, "num_pnp_success": 7}))

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(stats_a), str(stats_b)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "submap1 | geo(thr=n/a, pairs=n/a, tp=n/a, fp=n/a, tp_ratio=n/a) | "
        "pnp(sampled=n/a, success=5, se3_inliers=n/a, <2m=n/a, >=2m=n/a, ratio_lt2m=n/a)",
        "submap2 | geo(thr=n/a, pairs=n/a, tp=n/a, fp=n/a, tp_ratio=n/a) | "
        "pnp(sampled=n/a, success=7, se3_inliers=n/a, <2m=n/a, >=2m=n/a, ratio_lt2m=n/a)",
    ]


def test_summarize_merge_stats_supports_top_level_geo_verify_fields(tmp_path: Path) -> None:
    stats_path = tmp_path / "submap3_merge_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "submap_idx": 3,
                "num_pnp_sampled": 1336,
                "num_pnp_success": 589,
                "num_se3_inliers": 512,
                "min_inliers": 120,
                "geo_verify_tp_fp": {
                    "threshold_f_inliers": 120,
                    "num_pairs_above_thresh": 5236,
                    "num_tp": 4840,
                    "num_fp": 396,
                    "tp_ratio": 0.9244,
                    "fp_ratio": 0.0756,
                },
                "pnp_error_stats": {
                    "num_error_lt2m": 524,
                    "num_error_ge2m": 65,
                    "ratio_lt2m": 0.8896,
                },
            }
        )
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(stats_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == (
        "submap3 | geo(thr=120, pairs=5236, tp=4840, fp=396, tp_ratio=0.9244) | "
        "pnp(sampled=1336, success=589, se3_inliers=512, <2m=524, >=2m=65, ratio_lt2m=0.8896)"
    )


def test_summarize_merge_stats_supports_geometric_verification_block(tmp_path: Path) -> None:
    stats_path = tmp_path / "submap3_merge_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "submap_idx": 3,
                "num_pnp_sampled": 1336,
                "num_pnp_success": 589,
                "num_se3_inliers": 512,
                "geometric_verification": {
                    "min_inliers": 120,
                    "geo_verify_tp_fp": {
                        "threshold_f_inliers": 120,
                        "num_pairs_above_thresh": 5236,
                        "num_tp": 4840,
                        "num_fp": 396,
                        "tp_ratio": 0.9244,
                    },
                },
                "pnp_error_stats": {
                    "num_error_lt2m": 524,
                    "num_error_ge2m": 65,
                    "ratio_lt2m": 0.8896,
                },
            }
        )
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(stats_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == (
        "submap3 | geo(thr=120, pairs=5236, tp=4840, fp=396, tp_ratio=0.9244) | "
        "pnp(sampled=1336, success=589, se3_inliers=512, <2m=524, >=2m=65, ratio_lt2m=0.8896)"
    )
