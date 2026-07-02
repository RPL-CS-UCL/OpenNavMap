#!/usr/bin/env python
"""Apply bundle adjustment to pre-built per-submap SfM reconstructions.

Loads existing COLMAP .bin models, runs pycolmap.bundle_adjustment(),
and writes optimized models to a new directory -- without re-running
feature extraction, matching, or triangulation.

Usage:
    PYTHONPATH=<opennavmap/python> \
    python python/benchmark_map_merge/scripts/apply_ba_to_prebuilt_sfm.py \
        --input-sfm-root /path/to/s00000_sfm_netvlad_splg_025 \
        --output-sfm-root /path/to/s00000_sfm_netvlad_splg_025_ba10 \
        --ba-iter 10 \
        --jobs 4
"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pycolmap

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))


def _has_colmap_files(model_dir: Path) -> bool:
    return all(
        (model_dir / fn).is_file()
        for fn in ("cameras.bin", "images.bin", "points3D.bin")
    )


def _get_submap_ids(sfm_root: Path) -> List[str]:
    submaps_dir = sfm_root / "submaps_sfm"
    if not submaps_dir.is_dir():
        raise FileNotFoundError(f"submaps_sfm not found in {sfm_root}")
    return sorted(
        [d.name for d in submaps_dir.iterdir() if d.is_dir() and (d / "sfm").is_dir()],
        key=int,
    )


def _get_intrinsics_from_model(
    model: pycolmap.Reconstruction,
) -> Optional[Tuple[float, float, float, float, int, int]]:
    if not model.cameras:
        return None
    cam = next(iter(model.cameras.values()))
    params = cam.params
    if len(params) >= 4:
        return (
            float(params[0]), float(params[1]),
            float(params[2]), float(params[3]),
            int(cam.width), int(cam.height),
        )
    return None


def _configure_ba_options(
    ba_iter: int,
    refine_extrinsics: bool,
) -> pycolmap.BundleAdjustmentOptions:
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_points3D = True
    options.refine_rig_from_world = refine_extrinsics
    options.refine_sensor_from_rig = False
    options.constant_rig_from_world_rotation = False
    options.print_summary = False
    options.ceres.solver_options.max_num_iterations = ba_iter
    return options


def apply_ba_single(
    input_sfm_dir: str,
    output_sfm_dir: str,
    ba_iter: int,
    refine_extrinsics: bool,
    submap_id: str,
) -> Dict:
    """Load, BA, write for a single submap.  Returns stats dict."""
    input_sfm_dir = Path(input_sfm_dir)
    output_sfm_dir = Path(output_sfm_dir)

    model = pycolmap.Reconstruction()
    model.read_binary(str(input_sfm_dir))

    before_images = len(model.images)
    before_points = len(model.points3D)

    options = _configure_ba_options(ba_iter, refine_extrinsics)

    t0 = time.time()
    pycolmap.bundle_adjustment(model, options)
    elapsed = time.time() - t0

    after_images = len(model.images)
    after_points = len(model.points3D)

    output_sfm_dir.mkdir(parents=True, exist_ok=True)
    model.write_binary(str(output_sfm_dir))

    return {
        "submap_id": submap_id,
        "ba_iter": ba_iter,
        "refine_extrinsics": refine_extrinsics,
        "before": {"num_images": before_images, "num_points3D": before_points},
        "after": {"num_images": after_images, "num_points3D": after_points},
        "elapsed_seconds": round(elapsed, 2),
    }


def _regenerate_rrd(
    sfm_dir: Path,
    submap_dir: Path,
    submap_id: str,
    ba_iter: int,
) -> None:
    from benchmark_map_merge.vis_utils import save_sfm_vis

    model = pycolmap.Reconstruction()
    model.read_binary(str(sfm_dir))
    intr = _get_intrinsics_from_model(model)
    save_sfm_vis(
        model,
        submap_dir / "sfm_reconstruction.rrd",
        title=(
            f"SfM submap{submap_id} (BA iter={ba_iter}): "
            f"{len(model.images)} reg / {len(model.points3D)} pts"
        ),
        intrinsics=intr,
    )


def _write_submap_summary(submap_dir: Path, stats: Dict) -> None:
    summary = {
        "submap_id": stats["submap_id"],
        "ba_applied": True,
        "ba_max_iter": stats["ba_iter"],
        "refine_extrinsics": stats["refine_extrinsics"],
        "num_registered_images": stats["after"]["num_images"],
        "num_points3D": stats["after"]["num_points3D"],
        "before_ba": {
            "num_registered_images": stats["before"]["num_images"],
            "num_points3D": stats["before"]["num_points3D"],
        },
        "elapsed_seconds": stats["elapsed_seconds"],
    }
    path = submap_dir / "sfm_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def _finalize_submap(
    input_submap_dir: Path,
    output_submap_dir: Path,
    submap_id: str,
    ba_iter: int,
    stats: Dict,
) -> None:
    """Regenerate rrd, copy png, write summary for one submap."""
    output_sfm_dir = output_submap_dir / "sfm"

    try:
        _regenerate_rrd(output_sfm_dir, output_submap_dir, submap_id, ba_iter)
    except Exception as e:
        print(f"[apply_ba] sub{submap_id}: rrd regeneration failed: {e}")

    src_png = input_submap_dir / "topdown_poses.png"
    if src_png.exists():
        shutil.copy2(str(src_png), str(output_submap_dir / "topdown_poses.png"))

    _write_submap_summary(output_submap_dir, stats)


def apply_ba_to_prebuilt_sfm(
    input_root: Path,
    output_root: Path,
    ba_iter: int,
    refine_extrinsics: bool = True,
    jobs: int = 1,
) -> List[Dict]:
    submap_ids = _get_submap_ids(input_root)
    print(
        f"[apply_ba] {len(submap_ids)} submaps, "
        f"ba_iter={ba_iter}, refine_extrinsics={refine_extrinsics}, jobs={jobs}"
    )

    output_submaps_root = output_root / "submaps_sfm"
    output_submaps_root.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[Path, Path, str]] = []
    for sid in submap_ids:
        inp = input_root / "submaps_sfm" / sid / "sfm"
        out = output_submaps_root / sid / "sfm"
        if not _has_colmap_files(inp):
            print(f"[apply_ba] sub{sid}: SKIP (missing .bin files)")
            continue
        tasks.append((inp, out, sid))

    results: List[Dict] = []

    if jobs <= 1:
        for inp, out, sid in tasks:
            print(f"[apply_ba] sub{sid}: loading -> BA({ba_iter}) -> writing...")
            stats = apply_ba_single(
                str(inp), str(out), ba_iter, refine_extrinsics, sid
            )
            results.append(stats)
            print(
                f"[apply_ba] sub{sid}: "
                f"{stats['before']['num_points3D']} -> "
                f"{stats['after']['num_points3D']} pts, "
                f"{stats['elapsed_seconds']}s"
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            future_map = {}
            for inp, out, sid in tasks:
                fut = pool.submit(
                    apply_ba_single,
                    str(inp), str(out), ba_iter, refine_extrinsics, sid,
                )
                future_map[fut] = sid
            for fut in as_completed(future_map):
                sid = future_map[fut]
                try:
                    stats = fut.result()
                    results.append(stats)
                    print(
                        f"[apply_ba] sub{sid}: "
                        f"{stats['before']['num_points3D']} -> "
                        f"{stats['after']['num_points3D']} pts, "
                        f"{stats['elapsed_seconds']}s"
                    )
                except Exception as e:
                    print(f"[apply_ba] sub{sid}: FAILED: {e}")

    for inp_sfm, out_sfm, sid in tasks:
        stats = next((s for s in results if s["submap_id"] == sid), None)
        if stats is None:
            continue
        _finalize_submap(inp_sfm.parent, out_sfm.parent, sid, ba_iter, stats)

    for src_name in ("metrics", "logs"):
        src = input_root / src_name
        if src.is_dir():
            shutil.copytree(str(src), str(output_root / src_name), dirs_exist_ok=True)

    ba_log = output_root / "logs" / "ba_applied.log"
    ba_log.parent.mkdir(parents=True, exist_ok=True)
    with open(ba_log, "w") as f:
        f.write(
            f"BA: max_iter={ba_iter}, refine_extrinsics={refine_extrinsics}\n"
            f"Input:  {input_root}\n"
            f"Output: {output_root}\n"
            f"Submaps processed: {len(results)}/{len(submap_ids)}\n\n"
        )
        for s in results:
            f.write(json.dumps(s) + "\n")

    print(f"\n[apply_ba] Done. {len(results)}/{len(submap_ids)} submaps processed.")
    print(f"[apply_ba] Output: {output_root}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply bundle adjustment to pre-built per-submap SfM."
    )
    parser.add_argument("--input-sfm-root", type=Path, required=True)
    parser.add_argument("--output-sfm-root", type=Path, required=True)
    parser.add_argument("--ba-iter", type=int, required=True)
    parser.add_argument(
        "--no-refine-extrinsics",
        action="store_true",
        help="Only refine 3D points, keep poses fixed (default: refine poses)",
    )
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    apply_ba_to_prebuilt_sfm(
        input_root=args.input_sfm_root,
        output_root=args.output_sfm_root,
        ba_iter=args.ba_iter,
        refine_extrinsics=not args.no_refine_extrinsics,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
