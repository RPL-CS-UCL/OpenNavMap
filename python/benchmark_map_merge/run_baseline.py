#!/usr/bin/env python
"""Run HLoc-based multi-session map merging baseline experiment.

Usage (first test, order 0, first 3 submaps):
    PYTHONPATH=<opennavmap/python>:<opennavmap/third_party/pose_estimation_models/estimator> \
    python python/benchmark_map_merge/run_baseline.py \
        --dataset-root /Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria \
        --method hloc_superpoint_splg \
        --order-index 0 \
        --max-submaps 3

Usage (subset, order 0, first 5 submaps):
    python ... --method hloc_superpoint_splg --order-index 0 --max-submaps 5
    python ... --method hloc_disk_dilg --order-index 0 --max-submaps 5

Usage (full, all orders):
    for i in $(seq 0 9); do
        python ... --method hloc_superpoint_splg --order-index $i
    done
"""

import sys
import time
import logging
import shutil
import json
from pathlib import Path

import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_map_merge.hloc_merger import HlocMapMerger, _get_image_list
from benchmark_map_merge.hloc_sfm_merger import (
    HlocSfmMapMerger,
    _GEO_VERIFY_MIN_MATCHES,
    _PNP_MIN_INLIERS,
    _FEATURE_CONF,
    _RETRIEVAL_CONF,
)
from benchmark_map_merge.merge_writer import (
    read_poses, read_timestamps,
    apply_transform, merge_poses, write_poses_txt, write_timestamps_txt,
    write_summary_json, create_merge_dir, create_finalmap_symlink,
    read_intrinsics, read_gps, read_edges_odom,
    write_intrinsics_txt, write_gps_txt, write_edges_odom_txt,
    merge_edges_with_offset,
)
from benchmark_map_merge.export_eval_data import export_to_eval_structure
from benchmark_map_merge.vis_utils import save_topdown_pose_viz, save_sfm_vis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_baseline")

ORDER_TAGS = ["in", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"]


def _has_colmap_model_files(model_dir: Path) -> bool:
    return all((model_dir / file_name).is_file() for file_name in (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    ))


def _is_registered(model: pycolmap.Reconstruction, image_id: int, img) -> bool:
    if hasattr(img, "registered"):
        return bool(img.registered)
    if hasattr(model, "reg_image_ids"):
        return int(image_id) in set(model.reg_image_ids())
    return True


def _log(msg: str, log_file: Path = None):
    logger.info(msg)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")


def _build_sfm_summary(
    model: object,
    sampled_frames: int,
    total_frames: int,
    sfm_pairs_path: Path,
    sfm_rrd_path: Path,
    topdown_path: Path,
) -> dict:
    """Build a compact JSON-serializable summary of reference SfM outputs."""
    num_sfm_pairs = 0
    if sfm_pairs_path.exists():
        with open(sfm_pairs_path) as f:
            num_sfm_pairs = sum(1 for line in f if line.strip())

    return {
        "num_total_ref_frames": int(total_frames),
        "num_sampled_frames": int(sampled_frames),
        "num_registered_images": len(model.images),
        "num_points3D": len(model.points3D),
        "num_sfm_pairs": num_sfm_pairs,
        "sfm_reconstruction_rrd": {
            "path": str(sfm_rrd_path),
            "size_bytes": sfm_rrd_path.stat().st_size if sfm_rrd_path.exists() else 0,
        },
        "topdown_poses_png": {
            "path": str(topdown_path),
            "size_bytes": topdown_path.stat().st_size if topdown_path.exists() else 0,
        },
    }


def _write_sfm_summary(summary: dict, output_path: Path) -> None:
    """Write SfM/BA diagnostic summary as indented JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def _ensure_ba_summary_defaults(
    summary: dict,
    submap_idx: int,
    num_pnp_frames_sampled: int,
    num_pnp_success: int,
) -> dict:
    """Fill missing BA summary fields used by logging and diagnostics."""
    summary.setdefault("num_pnp_success", int(num_pnp_success))
    summary.setdefault("num_pose_refined", 0)
    summary.setdefault("num_registered_to_reconstruction", 0)
    summary.setdefault("global_ba_iterations", 0)
    summary.setdefault("submap_idx", int(submap_idx))
    summary.setdefault("geometric_verification", {})
    summary.setdefault("num_pnp_frames_sampled", int(num_pnp_frames_sampled))
    summary.setdefault("num_vio_fallback_frames", 0)
    return summary


def _build_result_root(
    dataset_root: Path,
    order_tag: str,
    method: str,
    data_dir: str,
    sfm_ba_iter: int = 0,
    sfm_only: bool = False,
) -> Path:
    if sfm_only:
        data_label = data_dir.replace("s00000_aria_", "")
        return dataset_root / f"s00000_sfm_{data_label}_sba{sfm_ba_iter}"
    data_suffix = f"_{data_dir.replace('s00000_aria_', '')}" if data_dir != "s00000_aria_data" else ""
    sba_suffix = f"_sba{sfm_ba_iter}" if method == "hloc_sfm_netvlad_splg" else ""
    return dataset_root / f"s00000_results_{order_tag}_{method}{data_suffix}{sba_suffix}"


def _read_intr_tuple(submap_dir: Path) -> tuple:
    with open(submap_dir / "intrinsics.txt") as intrinsics_file:
        tokens = intrinsics_file.readline().strip().split()
    return (
        float(tokens[1]), float(tokens[2]),
        float(tokens[3]), float(tokens[4]),
        int(tokens[5]), int(tokens[6]),
    )


def _save_submap_sfm_outputs(
    sfm_merger: HlocSfmMapMerger,
    model: pycolmap.Reconstruction,
    output_dir: Path,
    submap_images: list,
    gt_poses: dict,
    intrinsics: tuple,
    sfm_pairs_path: Path,
    submap_label: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sfm").mkdir(parents=True, exist_ok=True)
    model.write_binary(str(output_dir / "sfm"))

    rrd_path = output_dir / "sfm_reconstruction.rrd"
    topdown_path = output_dir / "topdown_poses.png"
    save_sfm_vis(
        model,
        rrd_path,
        title=f"SfM submap {submap_label}: {len(model.images)} reg / {len(model.points3D)} pts",
        intrinsics=intrinsics,
    )

    registered_images = sorted(
        [img.name for img_id, img in model.images.items()
         if _is_registered(model, img_id, img)]
    )
    sfm_poses = {
        img.name: sfm_merger.extract_w2c_vec_from_image(img)
        for img_id, img in model.images.items()
        if _is_registered(model, img_id, img)
    }
    save_topdown_pose_viz(
        [sfm_poses],
        [{img: gt_poses[img] for img in registered_images if img in gt_poses}],
        [registered_images],
        topdown_path,
        title=f"submap_{submap_label}",
    )

    summary = _build_sfm_summary(
        model,
        sampled_frames=len(registered_images),
        total_frames=len(submap_images),
        sfm_pairs_path=sfm_pairs_path,
        sfm_rrd_path=rrd_path,
        topdown_path=topdown_path,
    )
    _write_sfm_summary(summary, output_dir / "sfm_summary.json")
    return summary


def _run_only_build_sfm(
    submap_ids: list,
    submap_dirs: list,
    sfm_merger: HlocSfmMapMerger,
    sfm_sample_dist: float,
    sfm_ba_iter: int,
    overwrite: bool,
    result_root: Path,
    log_file: Path,
) -> None:
    summaries = []
    failures = []
    total_start = time.time()
    _log("\n=== SfM-only mode: build each submap independently ===", log_file)

    for index, (submap_id, submap_dir) in enumerate(zip(submap_ids, submap_dirs)):
        if not submap_dir.exists():
            _log(f"  SKIP: submap {submap_id} directory not found: {submap_dir}", log_file)
            failures.append({"submap": submap_id, "error": "directory not found"})
            continue

        images = _get_image_list(submap_dir)
        intrinsics = _read_intr_tuple(submap_dir)
        vio_poses = read_poses(str(submap_dir / "poses.txt"))
        gt_poses = read_poses(str(submap_dir / "poses_abs_gt.txt"))
        tag = f"sub{index}"

        _log(f"\n--- Building SfM submap {submap_id}: {len(images)} images ---", log_file)
        start = time.time()
        model = sfm_merger.build_submap_sfm(
            submap_dir,
            images,
            intrinsics,
            vio_poses=vio_poses,
            sfm_sample_dist=sfm_sample_dist,
            sfm_ba_iter=sfm_ba_iter,
            overwrite=overwrite,
            submap_tag=tag,
        )
        if model is None:
            _log(f"  FAILED: submap {submap_id} SfM failed", log_file)
            failures.append({"submap": submap_id, "error": "SfM failed"})
            continue

        output_dir = result_root / "submaps_sfm" / str(submap_id)
        summary = _save_submap_sfm_outputs(
            sfm_merger,
            model,
            output_dir,
            images,
            gt_poses,
            intrinsics,
            result_root / "_work" / tag / "pairs-sfm.txt",
            str(submap_id),
        )
        summary["submap"] = str(submap_id)
        summary["time_sec"] = time.time() - start
        summaries.append(summary)
        _log(
            f"  submap {submap_id}: {summary['num_registered_images']} registered, "
            f"{summary['num_points3D']} points, {summary['num_sfm_pairs']} pairs, "
            f"{summary['time_sec']:.1f}s",
            log_file,
        )
        # clean up h5 intermediate files to free disk space
        tag_work_dir = result_root / "_work" / tag
        for h5_file in tag_work_dir.glob("*.h5"):
            h5_file.unlink(missing_ok=True)
        _log(f"  cleaned h5 intermediates for {tag}", log_file)

    total_time = time.time() - total_start
    write_summary_json(
        {
            "mode": "sfm_only",
            "num_requested_submaps": len(submap_ids),
            "num_built_success": len(summaries),
            "num_failed": len(failures),
            "sfm_sample_dist": sfm_sample_dist,
            "sfm_ba_iter": sfm_ba_iter,
            "total_time_sec": total_time,
            "submaps": summaries,
            "failures": failures,
        },
        result_root / "metrics" / "summary.json",
    )
    _log("\n=== SfM-only Summary ===", log_file)
    _log(f"Total time: {total_time:.0f}s", log_file)
    _log(f"Successfully built: {len(summaries)}", log_file)
    _log(f"Failures: {len(failures)}", log_file)


def run_order(
    dataset_root: Path,
    method: str,
    order_index: int,
    max_submaps: int = None,
    traj_eval_data_root: Path = None,
    skip_eval_export: bool = False,
    data_dir: str = "s00000_aria_full_data",
    pnp_sample_dist: float = 0.5,
    sfm_sample_dist: float = 0.25,
    sfm_ba_iter: int = 0,
    overwrite: bool = False,
    submap_sfm: bool = False,
    submap_merge: bool = False,
    dataset_name: str = None,
    prebuilt_sfm_root: Path = None,
):
    if not submap_sfm and not submap_merge:
        raise ValueError("Specify exactly one of --submap-sfm or --submap-merge")
    if submap_sfm and submap_merge:
        raise ValueError("--submap-sfm and --submap-merge are mutually exclusive")

    orders_file = dataset_root / "s00000_orders.txt"
    with open(orders_file) as f:
        lines = f.readlines()
    if order_index >= len(lines):
        raise ValueError(f"order_index {order_index} out of range (0-{len(lines)-1})")

    submap_ids = lines[order_index].strip().split()
    order_tag = ORDER_TAGS[order_index]
    if max_submaps:
        submap_ids = submap_ids[:max_submaps]
        order_tag = f"{order_tag}_{max_submaps}sub"

    result_root = _build_result_root(
        dataset_root, order_tag, method, data_dir, sfm_ba_iter,
        sfm_only=submap_sfm,
    )
    if overwrite and result_root.exists():
        shutil.rmtree(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    log_file = result_root / "logs" / "pipeline.log"

    _log(f"=== Multi-Session Map Merging Baseline ===", log_file)
    _log(f"Method: {method}", log_file)
    _log(f"Order: {order_index} ({ORDER_TAGS[order_index]}), Tag: {order_tag}", log_file)
    _log(f"Submaps: {submap_ids}", log_file)
    _log(f"Result dir: {result_root}", log_file)
    _log(f"Data dir: {data_dir}", log_file)
    _log(f"PnP sample distance: {pnp_sample_dist}", log_file)
    _log(f"SfM sample distance: {sfm_sample_dist}", log_file)
    _log(f"SfM BA iterations: {sfm_ba_iter}", log_file)
    _log(f"Mode: {'submap-sfm' if submap_sfm else 'submap-merge'}", log_file)

    submap_base = dataset_root / data_dir
    submap_dirs = [submap_base / sid for sid in submap_ids]
    for d in submap_dirs:
        if not d.exists() and not submap_sfm:
            raise FileNotFoundError(f"Submap directory not found: {d}")

    work_dir = result_root / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    _SFM_METHODS = {"hloc_sfm_netvlad_splg"}
    if method in _SFM_METHODS:
        sfm_merger = HlocSfmMapMerger(work_dir)
        merger = None
    else:
        sfm_merger = None
        merger = HlocMapMerger(method, work_dir)

    if submap_sfm:
        if method not in _SFM_METHODS:
            raise ValueError("--submap-sfm is only supported with hloc_sfm_netvlad_splg")
        _run_only_build_sfm(
            submap_ids,
            submap_dirs,
            sfm_merger,
            sfm_sample_dist,
            sfm_ba_iter,
            overwrite,
            result_root,
            log_file,
        )
        return

    merge_ids = [submap_ids[0]]
    merge_name = f"merge_{'_'.join(merge_ids)}"
    merge_dir = create_merge_dir(result_root, merge_name)
    create_finalmap_symlink(result_root, merge_name)

    _log(f"\n--- Merging submap 0 (reference): {submap_ids[0]} ---", log_file)
    ref_images = _get_image_list(submap_dirs[0])
    ref_dir = submap_dirs[0]

    ref_poses = read_poses(str(ref_dir / "poses.txt"))
    ref_gt    = read_poses(str(ref_dir / "poses_abs_gt.txt"))
    ref_ts    = read_timestamps(str(ref_dir / "timestamps.txt"))

    ref_intr  = read_intrinsics(str(ref_dir / "intrinsics.txt"))
    ref_gps   = read_gps(str(ref_dir / "gps_data.txt"))
    _ref_odom_path = ref_dir / "edges_odom.txt"
    ref_edges = read_edges_odom(str(_ref_odom_path)) if _ref_odom_path.exists() else []
    if not _ref_odom_path.exists():
        print(f"[run_baseline] edges_odom.txt not found in ref submap, skipping odometry edges")

    if method not in _SFM_METHODS:
        ref_poses_local = {img: ref_poses[img] for img in ref_images if img in ref_poses}
        ref_gt_local    = {img: ref_gt[img]    for img in ref_images if img in ref_gt}
        ref_ts_local    = {img: ref_ts[img]    for img in ref_images if img in ref_ts}

        submap_poses          = [ref_poses_local]
        submap_gt             = [ref_gt_local]
        submap_ts             = [ref_ts_local]
        submap_intrinsics     = [{img: ref_intr[img] for img in ref_images if img in ref_intr}]
        submap_gps            = [{img: ref_gps[img]  for img in ref_images if img in ref_gps}]
        submap_images_ordered = [ref_images]
        merged_edges          = list(ref_edges)

        flat_poses, flat_gt, flat_ts, flat_intr, flat_gps = {}, {}, {}, {}, {}
        _off = 0
        for _imgs, _po, _gt, _ts, _intr, _gps in zip(
            submap_images_ordered, submap_poses, submap_gt,
            submap_ts, submap_intrinsics, submap_gps,
        ):
            for _j, _img in enumerate(_imgs):
                _k = f"seq/{_off + _j:06d}.color.jpg"
                if _img in _po:   flat_poses[_k] = _po[_img]
                if _img in _gt:   flat_gt[_k]    = _gt[_img]
                if _img in _ts:   flat_ts[_k]    = _ts[_img]
                if _img in _intr: flat_intr[_k]  = _intr[_img]
                if _img in _gps:  flat_gps[_k]   = _gps[_img]
            _off += len(_imgs)
        write_poses_txt(flat_poses, merge_dir / "poses.txt")
        write_poses_txt(flat_gt,    merge_dir / "poses_abs_gt.txt")
        write_timestamps_txt(flat_ts, merge_dir / "timestamps.txt")
        write_intrinsics_txt(flat_intr, merge_dir / "intrinsics.txt")
        write_gps_txt(flat_gps,         merge_dir / "gps_data.txt")
        write_edges_odom_txt(merged_edges, merge_dir / "edges_odom.txt")
    else:
        submap_poses          = []
        submap_gt             = []
        submap_ts             = []
        submap_intrinsics     = []
        submap_gps            = []
        submap_images_ordered = []
        merged_edges          = []
    _log(f"  reference submap: {len(ref_images)} images", log_file)

    # ---- SfM path: build reference map and prepare incremental BA ----
    if method in _SFM_METHODS:
        _sfm_root = prebuilt_sfm_root if prebuilt_sfm_root is not None else result_root
        try:
            with open(ref_dir / "intrinsics.txt") as _f:
                _tok = _f.readline().strip().split()
            intr_tuple = (float(_tok[1]), float(_tok[2]),
                          float(_tok[3]), float(_tok[4]),
                          int(_tok[5]),   int(_tok[6]))
        except Exception:
            intr_tuple = (444.492708, 444.492708, 511.5, 287.5, 1024, 576)

        prebuilt_sub0 = _sfm_root / "submaps_sfm" / submap_ids[0] / "sfm"
        if submap_merge and _has_colmap_model_files(prebuilt_sub0):
            _log(f"  loading pre-built SfM for sub0 from {prebuilt_sub0}", log_file)
            sfm_model = pycolmap.Reconstruction()
            sfm_model.read_binary(str(prebuilt_sub0))
            # copy features if present (may have been cleaned up after --submap-sfm)
            _prebuilt_tag = result_root / "_work" / "sub0"
            for _feat_name, _dst_name in [
                ("feats-sp.h5", "feats-ref.h5"),
                ("feats-netvlad.h5", "global-feats-netvlad.h5"),
            ]:
                _src = _prebuilt_tag / _feat_name
                _dst = work_dir / _dst_name
                if _src.exists() and not _dst.exists():
                    shutil.copy2(str(_src), str(_dst))
            # re-extract features if missing (e.g. cleaned up after --submap-sfm)
            from hloc import extract_features as _ef
            _feats_ref = work_dir / "feats-ref.h5"
            _feats_global = work_dir / "global-feats-netvlad.h5"
            if not _feats_ref.exists():
                _log("  feats-ref.h5 missing, re-extracting SuperPoint for sub0...", log_file)
                _ef.main(_FEATURE_CONF, ref_dir, image_list=ref_images,
                         feature_path=_feats_ref, overwrite=False)
            if not _feats_global.exists():
                _log("  global-feats-netvlad.h5 missing, re-extracting NetVLAD for sub0...", log_file)
                _ef.main(_RETRIEVAL_CONF, ref_dir, image_list=ref_images,
                         feature_path=_feats_global, overwrite=False)
        else:
            _log(f"  building SfM map from submap 0...", log_file)
            ref_vio = read_poses(str(ref_dir / "poses.txt"))
            sfm_model = sfm_merger.build_submap_sfm(
                ref_dir, ref_images, intr_tuple, vio_poses=ref_vio,
                sfm_sample_dist=sfm_sample_dist,
                sfm_ba_iter=sfm_ba_iter,
                overwrite=overwrite,
                submap_tag="sub0",
            )
            if sfm_model is None:
                _log("  ERROR: SfM failed for reference submap, aborting.", log_file)
                write_summary_json(
                    {"stage": "build_sfm_sub0", "error": "SfM failed or too few 3D points"},
                    result_root / "metrics" / "failed.json",
                )
                return

            _copy_src_sp = work_dir / "sub0" / "feats-sp.h5"
            _copy_src_global = work_dir / "sub0" / "feats-netvlad.h5"
            if _copy_src_sp.exists():
                shutil.copy2(str(_copy_src_sp), str(work_dir / "feats-ref.h5"))
            if _copy_src_global.exists():
                shutil.copy2(str(_copy_src_global), str(work_dir / "global-feats-netvlad.h5"))

        sampled_ref_images = sorted(
            [img.name for img_id, img in sfm_model.images.items()
             if _is_registered(sfm_model, img_id, img)]
        )
        ref_sfm_poses = {
            img.name: sfm_merger.extract_w2c_vec_from_image(img)
            for img_id, img in sfm_model.images.items()
            if _is_registered(sfm_model, img_id, img)
        }
        submap_poses          = [ref_sfm_poses]
        submap_gt             = [{img: ref_gt[img]   for img in sampled_ref_images if img in ref_gt}]
        submap_ts             = [{img: ref_ts[img]   for img in sampled_ref_images if img in ref_ts}]
        submap_intrinsics     = [{img: ref_intr[img] for img in sampled_ref_images if img in ref_intr}]
        submap_gps            = [{img: ref_gps[img]  for img in sampled_ref_images if img in ref_gps}]
        submap_images_ordered = [sampled_ref_images]
        merged_edges          = []
        # model_name -> (submap_idx, local_name) for pose refinement tracking
        model_name_to_submap_local = {name: (0, name) for name in sampled_ref_images}

        # save SfM reconstruction visualization
        sub0_sfm_dir = result_root / "submaps_sfm" / submap_ids[0]
        sub0_sfm_dir.mkdir(parents=True, exist_ok=True)
        (sub0_sfm_dir / "sfm").mkdir(parents=True, exist_ok=True)
        sfm_model.write_binary(str(sub0_sfm_dir / "sfm"))
        save_sfm_vis(
            sfm_model,
            sub0_sfm_dir / "sfm_reconstruction.rrd",
            title=f"SfM submap0: {len(sfm_model.images)} reg / {len(sfm_model.points3D)} pts",
            intrinsics=intr_tuple,
        )
        save_topdown_pose_viz(
            submap_poses,
            submap_gt,
            submap_images_ordered,
            sub0_sfm_dir / "topdown_poses.png",
            title=f"submap_{submap_ids[0]}",
        )

        current_model = sfm_model
        merge_sfm_dir = merge_dir / "sfm"
        merge_sfm_dir.mkdir(parents=True, exist_ok=True)
        current_model.write_binary(str(merge_sfm_dir))

        flat_poses, flat_gt, flat_ts, flat_intr, flat_gps = {}, {}, {}, {}, {}
        _off = 0
        for _imgs, _po, _gt, _ts, _intr, _gps in zip(
            submap_images_ordered, submap_poses, submap_gt,
            submap_ts, submap_intrinsics, submap_gps,
        ):
            for _j, _img in enumerate(_imgs):
                _k = f"seq/{_off + _j:06d}.color.jpg"
                if _img in _po:   flat_poses[_k] = _po[_img]
                if _img in _gt:   flat_gt[_k]    = _gt[_img]
                if _img in _ts:   flat_ts[_k]    = _ts[_img]
                if _img in _intr: flat_intr[_k]  = _intr[_img]
                if _img in _gps:  flat_gps[_k]   = _gps[_img]
            _off += len(_imgs)
        write_poses_txt(flat_poses, merge_dir / "poses.txt")
        write_poses_txt(flat_gt,    merge_dir / "poses_abs_gt.txt")
        write_timestamps_txt(flat_ts, merge_dir / "timestamps.txt")
        write_intrinsics_txt(flat_intr, merge_dir / "intrinsics.txt")
        write_gps_txt(flat_gps,         merge_dir / "gps_data.txt")
        write_edges_odom_txt(merged_edges, merge_dir / "edges_odom.txt")
        _log(f"  SfM model: {len(sfm_model.images)} images, "
             f"{len(sfm_model.points3D)} 3D points", log_file)

        # write merge_0 preds
        _preds0_dir = merge_dir.parent / "preds"
        _preds0_dir.mkdir(parents=True, exist_ok=True)
        write_poses_txt(ref_sfm_poses, _preds0_dir / "s0_pred.txt")
        save_topdown_pose_viz(
            submap_poses,
            submap_gt,
            submap_images_ordered,
            _preds0_dir / "topdown_merge_0.png",
            title="merge_0",
        )
        sfm_summary = _build_sfm_summary(
            sfm_model,
            sampled_frames=sfm_merger.last_sfm_sampled_frames,
            total_frames=len(ref_images),
            sfm_pairs_path=work_dir / "sub0" / "pairs-sfm.txt",
            sfm_rrd_path=sub0_sfm_dir / "sfm_reconstruction.rrd",
            topdown_path=_preds0_dir / "topdown_merge_0.png",
        )
        _write_sfm_summary(sfm_summary, result_root / "logs" / "sfm_summary.json")
        _log(
            f"  SfM summary: {sfm_summary['num_sampled_frames']}/"
            f"{sfm_summary['num_total_ref_frames']} sampled, "
            f"{sfm_summary['num_sfm_pairs']} pairs, "
            f"RRD {sfm_summary['sfm_reconstruction_rrd']['size_bytes']} bytes",
            log_file,
        )

    failures = []
    total_start = time.time()

    for i in range(1, len(submap_dirs)):
        sid = submap_ids[i]
        sdir = submap_dirs[i]
        incoming_images = _get_image_list(sdir)

        _log(f"\n--- Merging submap {i}: {sid} ({len(incoming_images)} images) ---", log_file)
        t_start = time.time()

        # ---- SfM path ----
        if method in _SFM_METHODS:
            next_merge_ids = merge_ids + [sid]
            next_merge_name = f"merge_{'_'.join(next_merge_ids)}"
            next_merge_dir = result_root / next_merge_name / "submap_disc_0"
            next_sfm_dir = next_merge_dir / "sfm"
            next_poses_path = next_merge_dir / "poses.txt"
            merge_stats_path = result_root / "logs" / f"submap{i}_merge_stats.json"

            if (not overwrite and _has_colmap_model_files(next_sfm_dir)
                    and next_poses_path.exists() and merge_stats_path.exists()):
                current_model = pycolmap.Reconstruction()
                current_model.read_binary(str(next_sfm_dir))
                cached_poses = read_poses(str(next_poses_path))
                with open(merge_stats_path) as f:
                    merge_stats = json.load(f)
                sampled_inc_images = merge_stats.get("sampled_inc_images", [])
                _cache_offset = sum(len(x) for x in submap_images_ordered)
                inc_poses_w2c = {
                    img: cached_poses[f"seq/{_cache_offset + idx:06d}.color.jpg"]
                    for idx, img in enumerate(sampled_inc_images)
                    if f"seq/{_cache_offset + idx:06d}.color.jpg" in cached_poses
                }
                model_name_to_orig = merge_stats.get("model_name_to_orig", {})
                _log(f"  loaded cached SfM reconstruction: {next_sfm_dir}", log_file)
            else:
                # check pre-built SfM from --submap-sfm step
                prebuilt_sub_i = _sfm_root / "submaps_sfm" / sid / "sfm"
                if submap_merge and _has_colmap_model_files(prebuilt_sub_i):
                    _log(f"  loading pre-built SfM for sub{i} from {prebuilt_sub_i}", log_file)
                    sfm_model_i = pycolmap.Reconstruction()
                    sfm_model_i.read_binary(str(prebuilt_sub_i))
                else:
                    inc_vio = read_poses(str(sdir / "poses.txt"))
                    _log(f"  building SfM map from submap {i}...", log_file)
                    sfm_model_i = sfm_merger.build_submap_sfm(
                        sdir,
                        incoming_images,
                        intr_tuple,
                        vio_poses=inc_vio,
                        sfm_sample_dist=sfm_sample_dist,
                        sfm_ba_iter=sfm_ba_iter,
                        overwrite=overwrite,
                        submap_tag=f"sub{i}",
                    )
                if sfm_model_i is None:
                    _log(f"  ERROR: SfM failed for submap {i}, skipping.", log_file)
                    failures.append({
                        "submap": sid,
                        "stage": "build_sfm_inc",
                        "error": "SfM failed or too few 3D points",
                    })
                    continue

                sampled_inc_images = sorted(
                    [img.name for img_id, img in sfm_model_i.images.items()
                     if _is_registered(sfm_model_i, img_id, img)]
                )
                sub_i_sfm_dir = result_root / "submaps_sfm" / sid
                sub_i_sfm_dir.mkdir(parents=True, exist_ok=True)
                (sub_i_sfm_dir / "sfm").mkdir(parents=True, exist_ok=True)
                sfm_model_i.write_binary(str(sub_i_sfm_dir / "sfm"))
                save_sfm_vis(
                    sfm_model_i,
                    sub_i_sfm_dir / "sfm_reconstruction.rrd",
                    title=(
                        f"SfM submap{i}: {len(sfm_model_i.images)} reg / "
                        f"{len(sfm_model_i.points3D)} pts"
                    ),
                    intrinsics=intr_tuple,
                )
                inc_gt_for_topdown = read_poses(str(sdir / "poses_abs_gt.txt"))
                sfm_poses_i = {
                    img.name: sfm_merger.extract_w2c_vec_from_image(img)
                    for img_id, img in sfm_model_i.images.items()
                    if _is_registered(sfm_model_i, img_id, img)
                }
                save_topdown_pose_viz(
                    [sfm_poses_i],
                    [{img: inc_gt_for_topdown[img] for img in sampled_inc_images
                      if img in inc_gt_for_topdown}],
                    [sampled_inc_images],
                    sub_i_sfm_dir / "topdown_poses.png",
                    title=f"submap_{sid}",
                )

                current_model, inc_poses_w2c, model_name_to_orig, merge_stats = sfm_merger.merge_model_with_se3(
                    current_model,
                    sfm_model_i,
                     ref_dir,
                     ref_images,
                     sdir,
                     sampled_inc_images,
                     intr_tuple,
                     submap_idx=i,
                     submap_tag=f"sub{i}",
                     gt_poses_ref=ref_gt,
                     gt_poses_inc=read_poses(str(sdir / "poses_abs_gt.txt"))
                     if (sdir / "poses_abs_gt.txt").exists() else None,
                 )
                if merge_stats is None:
                    merge_stats = {}
                merge_stats["sampled_inc_images"] = sampled_inc_images
                merge_stats["model_name_to_orig"] = model_name_to_orig
                _write_sfm_summary(merge_stats, merge_stats_path)

            if merge_stats:
                rv = merge_stats.get("retrieval", {})
                gv = merge_stats.get("geometric_verification", {})
                if log_file:
                    with open(log_file, "a") as log_handle:
                        log_handle.write(
                            f"  [Retrieval] queries={rv.get('num_queries', 0)}, "
                            f"db={rv.get('num_db', 0)}, "
                            f"top_k={rv.get('top_k', 0)}, "
                            f"raw_pairs={rv.get('num_pairs', 0)} | "
                            f"[FeatMatch] total_matches={gv.get('num_total_matches', 0)} | "
                            f"[GeoVerify] queries_kept={gv.get('num_query_kept', 0)}/"
                            f"{gv.get('num_query_total', 0)} "
                            f"(>={_GEO_VERIFY_MIN_MATCHES} F-inliers), "
                            f"pairs_written={gv.get('num_pairs_written', 0)}/"
                            f"{gv.get('num_pairs_total', 0)}\n"
                        )
                        for pair_item in gv.get("pairs_detail", []):
                            log_handle.write(
                                f"  [Pair] {pair_item['query']} <-> {pair_item['db']} "
                                f"feat_matches={pair_item['feat_matches']}, "
                                f"f_inliers={pair_item['f_inliers']}\n"
                            )
                _log(
                    f"  [PnP] sampled: {merge_stats.get('num_pnp_sampled', 0)}, "
                    f"success: {merge_stats.get('num_pnp_success', 0)}, "
                    f"threshold: inliers>={_PNP_MIN_INLIERS}",
                    log_file,
                )
                if log_file:
                    with open(log_file, "a") as _lh:
                        for pf in merge_stats.get("pnp_per_frame", []):
                            if pf["status"] == "SUCCESS":
                                _lh.write(
                                    f"  [PnP-match] {pf['frame']} <- db: {pf.get('best_db', '')} | "
                                    f"num_2d3d: {pf.get('num_2d3d', 0)} | "
                                    f"inliers: {pf.get('inliers', 0)} | "
                                    f"status: SUCCESS\n"
                                )
                            else:
                                _lh.write(
                                    f"  [PnP-match] {pf['frame']} | "
                                    f"num_db: {pf.get('num_db', 0)} | "
                                    f"inliers: {pf.get('num_inliers', 0)} | "
                                    f"status: {pf['status']}\n"
                                )
                _log(
                    f"  [SE(3)] inliers: {merge_stats.get('num_se3_inliers', 0)}, "
                    f"residual mean: {merge_stats.get('se3_residual_mean_m', 0.0):.3f}m, "
                    f"max: {merge_stats.get('se3_residual_max_m', 0.0):.3f}m",
                    log_file,
                )
                _log(
                    f"  [Merge] images: {merge_stats.get('num_images_merged', 0)}, "
                    f"points3D: {merge_stats.get('num_points3D_merged', 0)}",
                    log_file,
                )

            if current_model is None or inc_poses_w2c is None or len(inc_poses_w2c) < 1:
                _log(f"  FAILED: submap {sid} has no PnP overlap with merged map", log_file)
                failures.append({
                    "submap": sid,
                    "stage": "localization",
                    "error": "no PnP success or SE(3) estimation failed",
                })
                continue

            _log(
                f"  SfM merge summary: {merge_stats.get('num_pnp_success', 0)} PnP success, "
                f"{merge_stats.get('num_se3_inliers', 0)} SE(3) inliers, "
                f"{merge_stats.get('num_images_merged', 0)} images merged, "
                f"{merge_stats.get('num_points3D_merged', 0)} points merged",
                log_file,
            )

            # update model_name_to_submap_local for inc submap frames
            for model_name, orig_name in model_name_to_orig.items():
                if orig_name in sampled_inc_images:
                    model_name_to_submap_local[model_name] = (i, orig_name)

            # refine poses per-submap (Bug 1 fix: no cross-submap pollution)
            for img in current_model.images.values():
                if not _is_registered(current_model, img.image_id, img):
                    continue
                mapping = model_name_to_submap_local.get(img.name)
                if mapping is not None:
                    sub_idx, local_name = mapping
                    submap_poses[sub_idx][local_name] = sfm_merger.extract_w2c_vec_from_image(img)

            # append inc submap data (no reindex_dict, keep local names)
            submap_poses.append(inc_poses_w2c)
            submap_images_ordered.append(sampled_inc_images)

            inc_gt_dict = read_poses(str(sdir / "poses_abs_gt.txt"))
            submap_gt.append({img: inc_gt_dict[img] for img in sampled_inc_images
                               if img in inc_gt_dict})

            inc_ts_dict = read_timestamps(str(sdir / "timestamps.txt"))
            submap_ts.append({img: inc_ts_dict[img] for img in sampled_inc_images
                               if img in inc_ts_dict})

            inc_intr = read_intrinsics(str(sdir / "intrinsics.txt"))
            inc_gps  = read_gps(str(sdir / "gps_data.txt"))
            submap_intrinsics.append({img: inc_intr[img] for img in sampled_inc_images
                                      if img in inc_intr})
            submap_gps.append({img: inc_gps[img] for img in sampled_inc_images
                                if img in inc_gps})

            merge_ids.append(sid)
            merge_name = next_merge_name
            merge_dir = create_merge_dir(result_root, merge_name)
            create_finalmap_symlink(result_root, merge_name)
            merge_sfm_dir = merge_dir / "sfm"
            merge_sfm_dir.mkdir(parents=True, exist_ok=True)
            current_model.write_binary(str(merge_sfm_dir))

            flat_poses, flat_gt, flat_ts, flat_intr, flat_gps = {}, {}, {}, {}, {}
            _off = 0
            for _imgs, _po, _gt, _ts, _intr, _gps in zip(
                submap_images_ordered, submap_poses, submap_gt,
                submap_ts, submap_intrinsics, submap_gps,
            ):
                for _j, _img in enumerate(_imgs):
                    _k = f"seq/{_off + _j:06d}.color.jpg"
                    if _img in _po:   flat_poses[_k] = _po[_img]
                    if _img in _gt:   flat_gt[_k]    = _gt[_img]
                    if _img in _ts:   flat_ts[_k]    = _ts[_img]
                    if _img in _intr: flat_intr[_k]  = _intr[_img]
                    if _img in _gps:  flat_gps[_k]   = _gps[_img]
                _off += len(_imgs)
            write_poses_txt(flat_poses, merge_dir / "poses.txt")
            write_poses_txt(flat_gt,    merge_dir / "poses_abs_gt.txt")
            write_timestamps_txt(flat_ts, merge_dir / "timestamps.txt")
            write_intrinsics_txt(flat_intr, merge_dir / "intrinsics.txt")
            write_gps_txt(flat_gps,         merge_dir / "gps_data.txt")
            write_edges_odom_txt(merged_edges, merge_dir / "edges_odom.txt")

            # write per-submap predicted poses
            preds_dir = merge_dir.parent / "preds"
            preds_dir.mkdir(parents=True, exist_ok=True)
            write_poses_txt(flat_poses, preds_dir / "merged_pred.txt")
            save_topdown_pose_viz(
                submap_poses,
                submap_gt,
                submap_images_ordered,
                preds_dir / f"topdown_{merge_name}.png",
                title=merge_name,
            )
            sfm_vis_dir = merge_dir.parent / "sfm_vis"
            sfm_vis_dir.mkdir(parents=True, exist_ok=True)
            sfm_rrd_path = sfm_vis_dir / f"sfm_{merge_name}.rrd"
            save_sfm_vis(
                current_model,
                sfm_rrd_path,
                title=(
                    f"SfM {merge_name}: {len(current_model.images)} reg / "
                    f"{len(current_model.points3D)} pts"
                ),
                intrinsics=intr_tuple,
            )
            _log(
                f"  SfM visualization: {sfm_rrd_path} "
                f"({sfm_rrd_path.stat().st_size} bytes)",
                log_file,
            )

            elapsed = time.time() - t_start
            _log(f"  merged in {elapsed:.0f}s, total {sum(len(x) for x in submap_images_ordered)} poses", log_file)
            continue   # skip the existing v1 path below

        ref_poses_dict = read_poses(str(ref_dir / "poses.txt"))
        inc_poses_dict = read_poses(str(sdir / "poses.txt"))

        try:
            valid_pairs = merger.match_pairs(ref_dir, ref_images, sdir, incoming_images)
        except Exception as e:
            _log(f"  FAILED: matching error: {e}", log_file)
            failures.append({"submap": sid, "stage": "matching", "error": str(e)})
            continue

        if len(valid_pairs) < 5:
            _log(f"  FAILED: only {len(valid_pairs)} valid pairs (need >=5)", log_file)
            failures.append({
                "submap": sid, "stage": "matching",
                "error": f"only {len(valid_pairs)} valid pairs",
            })
            continue

        try:
            with open(ref_dir / "intrinsics.txt") as f:
                tokens = f.readline().strip().split()
            intr = (float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]))
        except Exception:
            intr = (444.492708, 444.492708, 511.5, 287.5)

        T_ref_incoming = merger.estimate_submap_transform(
            valid_pairs, ref_poses_dict, inc_poses_dict, intr,
        )
        if T_ref_incoming is None:
            _log(f"  FAILED: transform estimation failed", log_file)
            failures.append({
                "submap": sid, "stage": "transform",
                "error": f"could not estimate transform from {len(valid_pairs)} pairs",
            })
            continue

        inc_poses_local = {img: inc_poses_dict[img] for img in incoming_images
                           if img in inc_poses_dict}
        transformed_local = apply_transform(inc_poses_local, T_ref_incoming)

        submap_poses.append(transformed_local)
        submap_images_ordered.append(incoming_images)

        inc_gt_dict = read_poses(str(sdir / "poses_abs_gt.txt"))
        submap_gt.append({img: inc_gt_dict[img] for img in incoming_images
                          if img in inc_gt_dict})

        inc_ts_dict = read_timestamps(str(sdir / "timestamps.txt"))
        submap_ts.append({img: inc_ts_dict[img] for img in incoming_images
                          if img in inc_ts_dict})

        inc_intr  = read_intrinsics(str(sdir / "intrinsics.txt"))
        inc_gps   = read_gps(str(sdir / "gps_data.txt"))
        _inc_odom_path2 = sdir / "edges_odom.txt"
        inc_edges = read_edges_odom(str(_inc_odom_path2)) if _inc_odom_path2.exists() else []
        submap_intrinsics.append({img: inc_intr[img] for img in incoming_images
                                  if img in inc_intr})
        submap_gps.append({img: inc_gps[img] for img in incoming_images
                           if img in inc_gps})
        _edges_offset = sum(len(x) for x in submap_images_ordered[:-1])
        merged_edges = merge_edges_with_offset(merged_edges, inc_edges, offset=_edges_offset)

        merge_ids.append(sid)
        merge_name = f"merge_{'_'.join(merge_ids)}"
        merge_dir = create_merge_dir(result_root, merge_name)
        create_finalmap_symlink(result_root, merge_name)

        flat_poses, flat_gt, flat_ts, flat_intr, flat_gps = {}, {}, {}, {}, {}
        _off = 0
        for _imgs, _po, _gt, _ts, _intr, _gps in zip(
            submap_images_ordered, submap_poses, submap_gt,
            submap_ts, submap_intrinsics, submap_gps,
        ):
            for _j, _img in enumerate(_imgs):
                _k = f"seq/{_off + _j:06d}.color.jpg"
                if _img in _po:   flat_poses[_k] = _po[_img]
                if _img in _gt:   flat_gt[_k]    = _gt[_img]
                if _img in _ts:   flat_ts[_k]    = _ts[_img]
                if _img in _intr: flat_intr[_k]  = _intr[_img]
                if _img in _gps:  flat_gps[_k]   = _gps[_img]
            _off += len(_imgs)
        write_poses_txt(flat_poses,   merge_dir / "poses.txt")
        write_poses_txt(flat_gt,      merge_dir / "poses_abs_gt.txt")
        write_timestamps_txt(flat_ts, merge_dir / "timestamps.txt")
        write_intrinsics_txt(flat_intr, merge_dir / "intrinsics.txt")
        write_gps_txt(flat_gps,         merge_dir / "gps_data.txt")
        write_edges_odom_txt(merged_edges, merge_dir / "edges_odom.txt")

        elapsed = time.time() - t_start
        _log(f"  merged successfully in {elapsed:.0f}s, "
             f"total {sum(len(x) for x in submap_images_ordered)} poses", log_file)

    total_elapsed = time.time() - total_start
    num_success = len(submap_ids) - 1 - len(failures)
    num_requested = len(submap_ids) - 1

    _log(f"\n=== Summary ===", log_file)
    _log(f"Total time: {total_elapsed:.0f}s", log_file)
    _log(f"Requested incoming submaps: {num_requested}", log_file)
    _log(f"Successfully merged: {num_success}", log_file)
    _log(f"Failures: {len(failures)}", log_file)
    if failures:
        _log(f"  {failures}", log_file)
    _log(f"Final merged poses: {sum(len(x) for x in submap_images_ordered)}", log_file)

    summary = {
        "method": method,
        "order_index": order_index,
        "order_tag": order_tag,
        "data_dir": data_dir,
        "pnp_sample_dist": pnp_sample_dist,
        "sfm_ba_iter": sfm_ba_iter,
        "submaps": submap_ids,
        "num_requested_incoming": num_requested,
        "num_merged_success": num_success,
        "num_failed": len(failures),
        "submap_success_rate": num_success / num_requested if num_requested > 0 else 0,
        "total_poses": sum(len(x) for x in submap_images_ordered),
        "total_time_sec": total_elapsed,
        "failures": failures,
    }
    write_summary_json(summary, result_root / "metrics" / "summary.json")
    unmerged_ids = [f["submap"] for f in failures if f.get("stage") == "localization"]
    if unmerged_ids:
        write_summary_json(
            {"unmerged_submap_ids": unmerged_ids},
            result_root / "metrics" / "unmerged_submaps.json",
        )

    if not skip_eval_export and traj_eval_data_root:
        _dataset_name = dataset_name or f"{dataset_root.name}_s00000"
        # if dataset_name is explicitly given, use it as-is; otherwise append order tag
        dataset_order_name = _dataset_name if dataset_name else f"{_dataset_name}_{ORDER_TAGS[order_index]}"
        # eval algorithm directory name mirrors result_root naming: method + data_suffix + sba_suffix
        _data_suffix = f"_{data_dir.replace('s00000_aria_', '')}" if data_dir != "s00000_aria_data" else ""
        _sba_suffix = f"_sba{sfm_ba_iter}" if method == "hloc_sfm_netvlad_splg" else ""
        eval_method_name = f"{method}{_data_suffix}{_sba_suffix}"
        try:
            gt_path, est_path = export_to_eval_structure(
                merge_dir, traj_eval_data_root, dataset_order_name, eval_method_name
            )
            _log(f"\nEvaluation trajectories exported:", log_file)
            _log(f"  GT: {gt_path}", log_file)
            _log(f"  EST: {est_path}", log_file)
        except Exception as e:
            _log(f"  warning: eval export failed: {e}", log_file)

    return summary
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="HLoc multi-session map merging baseline")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path to dataset root, e.g. .../ucl_campus_aria")
    p.add_argument("--method", type=str, required=True,
                   choices=["hloc_superpoint_splg", "hloc_disk_dilg",
                            "hloc_sfm_netvlad_splg"],
                   help="HLoc feature+matcher combination")
    p.add_argument("--order-index", type=int, required=True,
                   help="Order index in orders file (0=in, 1=r0, ...)")
    p.add_argument("--max-submaps", type=int, default=None,
                   help="Limit to first N submaps (for subset/first-test)")
    p.add_argument("--traj-eval-data-root", type=Path,
                   default=Path("/Titan/dataset/data_opennavmap/traj_eval_data/map_merge_eval_data"),
                   help="Root for slam_trajectory_evaluation output")
    p.add_argument("--skip-eval-export", action="store_true",
                    help="Skip exporting TUM trajectories for evaluation")
    p.add_argument("--data-dir", type=str, default="s00000_aria_full_data",
                   help="Submap data directory name under dataset-root "
                        "(e.g. s00000_aria_full_data)")
    p.add_argument("--pnp-sample-dist", type=float, default=0.5,
                   help="VIO distance sampling for PnP localization frames "
                        "(e.g. 1.0 = one frame per meter). 0 = all frames.")
    p.add_argument("--sfm-sample-dist", type=float, default=0.25,
                   help="VIO distance sampling for per-submap SfM frames "
                        "(e.g. 0.25 = one frame per 0.25m). 0 = all frames.")
    p.add_argument("--sfm-ba-iter", type=int, default=0,
                   help="BA iterations after VIO-prior triangulation (point-only, pose fixed). "
                        "0 = pure triangulation, no BA.")
    p.add_argument("--submap-sfm", action="store_true",
                   help="Build independent SfM for each submap. "
                        "Output: result_root/submaps_sfm/{submap_id}/sfm/")
    p.add_argument("--submap-merge", action="store_true",
                   help="Merge pre-built submap SfM models. "
                        "Reads from result_root/submaps_sfm/{submap_id}/sfm/")
    p.add_argument("--overwrite", action="store_true",
                   help="Remove the result directory before running")
    p.add_argument("--prebuilt-sfm-root", type=Path, default=None,
                   help="Directory containing pre-built submap SfM results "
                        "(submaps_sfm/{submap_id}/sfm/) from a prior --submap-sfm run. "
                        "When set, --submap-merge loads SfM from here instead of "
                        "result_root, avoiding redundant SfM reconstruction.")
    p.add_argument("--dataset-name", type=str, default=None,
                   help="Dataset name for TUM eval export, e.g. 'vineyard_s00000'. "
                        "Defaults to <dataset_root.name>_s00000")
    args = p.parse_args()

    run_order(
        dataset_root=args.dataset_root,
        method=args.method,
        order_index=args.order_index,
        max_submaps=args.max_submaps,
        traj_eval_data_root=args.traj_eval_data_root,
        skip_eval_export=args.skip_eval_export,
        data_dir=args.data_dir,
        pnp_sample_dist=args.pnp_sample_dist,
        sfm_sample_dist=args.sfm_sample_dist,
        sfm_ba_iter=args.sfm_ba_iter,
        overwrite=args.overwrite,
        submap_sfm=args.submap_sfm,
        submap_merge=args.submap_merge,
        dataset_name=args.dataset_name,
        prebuilt_sfm_root=args.prebuilt_sfm_root,
    )
