#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _format_value(value: Optional[Any]) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _summary_name(stats_path: Path, data: Dict[str, Any]) -> str:
    submap_idx = data.get("submap_idx")
    if submap_idx is not None:
        return f"submap{submap_idx}"
    return stats_path.stem


def _build_summary_line(stats_path: Path, data: Dict[str, Any]) -> str:
    geometric_verification = data.get("geometric_verification") or {}
    image_matching = data.get("image_matching") or {}
    geo_verify_tp_fp = (
        geometric_verification.get("geo_verify_tp_fp")
        or data.get("geo_verify_tp_fp")
        or image_matching.get("geo_verify_tp_fp")
        or {}
    )
    pnp_error_stats = data.get("pnp_error_stats") or {}

    geo_thr = geo_verify_tp_fp.get(
        "threshold_f_inliers",
        geometric_verification.get(
            "min_inliers",
            data.get("min_inliers", image_matching.get("min_inliers")),
        ),
    )
    geo_pairs = geo_verify_tp_fp.get("num_pairs_above_thresh")
    geo_tp = geo_verify_tp_fp.get("num_tp")
    geo_fp = geo_verify_tp_fp.get("num_fp")
    geo_tp_ratio = geo_verify_tp_fp.get("tp_ratio")

    pnp_sampled = data.get("num_pnp_sampled")
    pnp_success = data.get("num_pnp_success")
    se3_inliers = data.get("num_se3_inliers")
    pnp_lt_2m = pnp_error_stats.get("num_error_lt2m")
    pnp_ge_2m = pnp_error_stats.get("num_error_ge2m")
    pnp_ratio_lt_2m = pnp_error_stats.get("ratio_lt2m")

    name = _summary_name(stats_path, data)
    return (
        f"{name} | "
        f"geo(thr={_format_value(geo_thr)}, pairs={_format_value(geo_pairs)}, "
        f"tp={_format_value(geo_tp)}, fp={_format_value(geo_fp)}, "
        f"tp_ratio={_format_value(geo_tp_ratio)}) | "
        f"pnp(sampled={_format_value(pnp_sampled)}, success={_format_value(pnp_success)}, "
        f"se3_inliers={_format_value(se3_inliers)}, <2m={_format_value(pnp_lt_2m)}, "
        f">=2m={_format_value(pnp_ge_2m)}, ratio_lt2m={_format_value(pnp_ratio_lt_2m)})"
    )


def _load_stats(stats_path: Path) -> Dict[str, Any]:
    return json.loads(stats_path.read_text())


def summarize_merge_stats(paths: Iterable[Path]) -> List[str]:
    lines: List[str] = []
    for stats_path in paths:
        data = _load_stats(stats_path)
        lines.append(_build_summary_line(stats_path, data))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize geo_verify_tp_fp and PnP stats from merge_stats JSON files."
    )
    parser.add_argument(
        "stats_paths",
        nargs="+",
        type=Path,
        help="One or more submap*_merge_stats.json files.",
    )
    args = parser.parse_args()

    for line in summarize_merge_stats(args.stats_paths):
        print(line)


if __name__ == "__main__":
    main()
