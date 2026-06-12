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
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_map_merge.hloc_merger import HlocMapMerger, _get_image_list
from benchmark_map_merge.hloc_sfm_merger import HlocSfmMapMerger
from benchmark_map_merge.pose_graph_optimizer import PoseGraphOptimizer, _vec_to_T, _T_to_vec
from benchmark_map_merge.merge_writer import (
    read_poses, read_timestamps, estimate_umeyama,
    apply_transform, merge_poses, write_poses_txt, write_timestamps_txt,
    write_summary_json, create_merge_dir, create_finalmap_symlink,
    reindex_dict, read_intrinsics, read_gps, read_edges_odom,
    write_intrinsics_txt, write_gps_txt, write_edges_odom_txt,
    merge_edges_with_offset,
)
from benchmark_map_merge.export_eval_data import export_to_eval_structure

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_baseline")

ORDER_TAGS = ["in", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"]


def _log(msg: str, log_file: Path = None):
    logger.info(msg)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")


def run_order(
    dataset_root: Path,
    method: str,
    order_index: int,
    max_submaps: int = None,
    traj_eval_data_root: Path = None,
    skip_eval_export: bool = False,
):
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

    result_root = dataset_root / f"s00000_results_{order_tag}_{method}"
    result_root.mkdir(parents=True, exist_ok=True)
    log_file = result_root / "logs" / "pipeline.log"

    _log(f"=== Multi-Session Map Merging Baseline ===", log_file)
    _log(f"Method: {method}", log_file)
    _log(f"Order: {order_index} ({ORDER_TAGS[order_index]}), Tag: {order_tag}", log_file)
    _log(f"Submaps: {submap_ids}", log_file)
    _log(f"Result dir: {result_root}", log_file)

    submap_base = dataset_root / "s00000_aria_data"
    submap_dirs = [submap_base / sid for sid in submap_ids]
    for d in submap_dirs:
        if not d.exists():
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

    merge_ids = [submap_ids[0]]
    merge_name = f"merge_{'_'.join(merge_ids)}"
    merge_dir = create_merge_dir(result_root, merge_name)
    create_finalmap_symlink(result_root, merge_name)

    _log(f"\n--- Merging submap 0 (reference): {submap_ids[0]} ---", log_file)
    ref_images = _get_image_list(submap_dirs[0])
    ref_dir = submap_dirs[0]

    global_offset = 0

    ref_poses = read_poses(str(ref_dir / "poses.txt"))
    ref_gt    = read_poses(str(ref_dir / "poses_abs_gt.txt"))
    ref_ts    = read_timestamps(str(ref_dir / "timestamps.txt"))
    ref_poses_local = {img: ref_poses[img] for img in ref_images if img in ref_poses}
    ref_gt_local    = {img: ref_gt[img]    for img in ref_images if img in ref_gt}
    ref_ts_local    = {img: ref_ts[img]    for img in ref_images if img in ref_ts}

    merged_poses = reindex_dict(ref_poses_local, ref_images, global_offset)
    merged_gt    = reindex_dict(ref_gt_local,    ref_images, global_offset)
    merged_ts    = reindex_dict(ref_ts_local,    ref_images, global_offset)

    write_poses_txt(merged_poses, merge_dir / "poses.txt")
    write_poses_txt(merged_gt,    merge_dir / "poses_abs_gt.txt")
    write_timestamps_txt(merged_ts, merge_dir / "timestamps.txt")

    ref_intr  = read_intrinsics(str(ref_dir / "intrinsics.txt"))
    ref_gps   = read_gps(str(ref_dir / "gps_data.txt"))
    ref_edges = read_edges_odom(str(ref_dir / "edges_odom.txt"))

    merged_intrinsics = reindex_dict(ref_intr, ref_images, global_offset)
    merged_gps        = reindex_dict(ref_gps,  ref_images, global_offset)
    merged_edges      = list(ref_edges)

    write_intrinsics_txt(merged_intrinsics, merge_dir / "intrinsics.txt")
    write_gps_txt(merged_gps,              merge_dir / "gps_data.txt")
    write_edges_odom_txt(merged_edges,     merge_dir / "edges_odom.txt")

    global_offset += len(ref_images)
    _log(f"  reference submap: {len(ref_images)} images", log_file)

    # ---- SfM path: build reference map and init pose graph ----
    if method in _SFM_METHODS:
        try:
            with open(ref_dir / "intrinsics.txt") as _f:
                _tok = _f.readline().strip().split()
            intr_tuple = (float(_tok[1]), float(_tok[2]),
                          float(_tok[3]), float(_tok[4]),
                          int(_tok[5]),   int(_tok[6]))
        except Exception:
            intr_tuple = (444.492708, 444.492708, 511.5, 287.5, 1024, 576)

        _log(f"  building SfM map from submap 0...", log_file)
        sfm_model = sfm_merger.build_ref_map(ref_dir, ref_images, intr_tuple)
        if sfm_model is None:
            _log("  ERROR: SfM failed for reference submap, aborting.", log_file)
            return

        optimizer = PoseGraphOptimizer()
        _sfm_all_frames: list = []
        ref_vio = read_poses(str(ref_dir / "poses.txt"))
        ref_offset = optimizer.add_submap(ref_vio, ref_images, is_reference=True)
        _sfm_all_frames += [(ref_offset + j, img) for j, img in enumerate(ref_images)]
        _log(f"  SfM model: {len(sfm_model.images)} images, "
             f"{len(sfm_model.points3D)} 3D points", log_file)

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
            pnp_poses = sfm_merger.localize_submap(
                sfm_model, ref_dir, ref_images,
                sdir, incoming_images, intr_tuple, submap_idx=i,
            )
            _log(f"  PnP: {len(pnp_poses)}/{len(incoming_images)} localized", log_file)

            if len(pnp_poses) < 5:
                _log(f"  FAILED: too few PnP successes ({len(pnp_poses)} < 5)", log_file)
                failures.append({"submap": sid, "stage": "pnp",
                                  "error": f"only {len(pnp_poses)} PnP successes"})
                global_offset += len(incoming_images)
                continue

            inc_vio = read_poses(str(sdir / "poses.txt"))
            inc_offset = optimizer.add_submap(
                inc_vio, incoming_images, abs_poses_w0=pnp_poses,
            )
            _sfm_all_frames += [(inc_offset + j, img)
                                 for j, img in enumerate(incoming_images)]

            optimized = optimizer.optimize(_sfm_all_frames)

            # Update merged_poses with optimized global-key poses
            for j, img in enumerate(ref_images):
                gk = ref_offset + j
                gname = f"seq/{j:06d}.color.jpg"
                if gk in optimized:
                    merged_poses[gname] = optimized[gk]

            for j, img in enumerate(incoming_images):
                gk = inc_offset + j
                gname = f"seq/{(global_offset + j):06d}.color.jpg"
                if gk in optimized:
                    merged_poses[gname] = optimized[gk]
                elif img in pnp_poses:
                    merged_poses[gname] = _T_to_vec(pnp_poses[img])

            # GT / timestamps / intrinsics / gps / edges (same as v1 path)
            inc_gt_dict = read_poses(str(sdir / "poses_abs_gt.txt"))
            inc_gt_local = {img: inc_gt_dict[img] for img in incoming_images
                            if img in inc_gt_dict}
            merged_gt.update(reindex_dict(inc_gt_local, incoming_images, global_offset))

            inc_ts_dict = read_timestamps(str(sdir / "timestamps.txt"))
            inc_ts_local = {img: inc_ts_dict[img] for img in incoming_images
                            if img in inc_ts_dict}
            merged_ts.update(reindex_dict(inc_ts_local, incoming_images, global_offset))

            inc_intr  = read_intrinsics(str(sdir / "intrinsics.txt"))
            inc_gps   = read_gps(str(sdir / "gps_data.txt"))
            inc_edges = read_edges_odom(str(sdir / "edges_odom.txt"))
            merged_intrinsics.update(reindex_dict(inc_intr, incoming_images, global_offset))
            merged_gps.update(reindex_dict(inc_gps, incoming_images, global_offset))
            merged_edges = merge_edges_with_offset(merged_edges, inc_edges, offset=global_offset)

            global_offset += len(incoming_images)

            merge_ids.append(sid)
            merge_name = f"merge_{'_'.join(merge_ids)}"
            merge_dir = create_merge_dir(result_root, merge_name)
            create_finalmap_symlink(result_root, merge_name)
            write_poses_txt(merged_poses,           merge_dir / "poses.txt")
            write_poses_txt(merged_gt,              merge_dir / "poses_abs_gt.txt")
            write_timestamps_txt(merged_ts,         merge_dir / "timestamps.txt")
            write_intrinsics_txt(merged_intrinsics, merge_dir / "intrinsics.txt")
            write_gps_txt(merged_gps,               merge_dir / "gps_data.txt")
            write_edges_odom_txt(merged_edges,      merge_dir / "edges_odom.txt")

            elapsed = time.time() - t_start
            _log(f"  merged in {elapsed:.0f}s, total {len(merged_poses)} poses", log_file)
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
            intr = (float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3]))
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
        transformed = reindex_dict(transformed_local, incoming_images, global_offset)
        merged_poses = merge_poses(merged_poses, transformed)

        inc_gt_dict = read_poses(str(sdir / "poses_abs_gt.txt"))
        inc_gt_local = {img: inc_gt_dict[img] for img in incoming_images
                        if img in inc_gt_dict}
        inc_gt = reindex_dict(inc_gt_local, incoming_images, global_offset)
        merged_gt = merge_poses(merged_gt, inc_gt)

        inc_ts_dict = read_timestamps(str(sdir / "timestamps.txt"))
        inc_ts_local = {img: inc_ts_dict[img] for img in incoming_images
                        if img in inc_ts_dict}
        inc_ts = reindex_dict(inc_ts_local, incoming_images, global_offset)
        merged_ts = merge_poses(merged_ts, inc_ts)

        inc_intr  = read_intrinsics(str(sdir / "intrinsics.txt"))
        inc_gps   = read_gps(str(sdir / "gps_data.txt"))
        inc_edges = read_edges_odom(str(sdir / "edges_odom.txt"))

        inc_intr_reindexed = reindex_dict(inc_intr, incoming_images, global_offset)
        inc_gps_reindexed  = reindex_dict(inc_gps,  incoming_images, global_offset)

        merged_intrinsics.update(inc_intr_reindexed)
        merged_gps.update(inc_gps_reindexed)
        merged_edges = merge_edges_with_offset(merged_edges, inc_edges, offset=global_offset)

        global_offset += len(incoming_images)

        merge_ids.append(sid)
        merge_name = f"merge_{'_'.join(merge_ids)}"
        merge_dir = create_merge_dir(result_root, merge_name)
        create_finalmap_symlink(result_root, merge_name)

        write_poses_txt(merged_poses,     merge_dir / "poses.txt")
        write_poses_txt(merged_gt,        merge_dir / "poses_abs_gt.txt")
        write_timestamps_txt(merged_ts,   merge_dir / "timestamps.txt")
        write_intrinsics_txt(merged_intrinsics, merge_dir / "intrinsics.txt")
        write_gps_txt(merged_gps,               merge_dir / "gps_data.txt")
        write_edges_odom_txt(merged_edges,       merge_dir / "edges_odom.txt")

        elapsed = time.time() - t_start
        _log(f"  merged successfully in {elapsed:.0f}s, "
             f"total {len(merged_poses)} poses", log_file)

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
    _log(f"Final merged poses: {len(merged_poses)}", log_file)

    summary = {
        "method": method,
        "order_index": order_index,
        "order_tag": order_tag,
        "submaps": submap_ids,
        "num_requested_incoming": num_requested,
        "num_merged_success": num_success,
        "num_failed": len(failures),
        "submap_success_rate": num_success / num_requested if num_requested > 0 else 0,
        "total_poses": len(merged_poses),
        "total_time_sec": total_elapsed,
        "failures": failures,
    }
    write_summary_json(summary, result_root / "metrics" / "summary.json")

    if not skip_eval_export and traj_eval_data_root:
        dataset_order_name = f"ucl_campus_aria_s00000_{ORDER_TAGS[order_index]}"
        try:
            gt_path, est_path = export_to_eval_structure(
                merge_dir, traj_eval_data_root, dataset_order_name, method
            )
            _log(f"\nEvaluation trajectories exported:", log_file)
            _log(f"  GT: {gt_path}", log_file)
            _log(f"  EST: {est_path}", log_file)
        except Exception as e:
            _log(f"  warning: eval export failed: {e}", log_file)

    return summary


def _vvec_to_matrix(vec: np.ndarray) -> np.ndarray:
    """Convert [qw,qx,qy,qz,tx,ty,tz] to 4x4 matrix."""
    from scipy.spatial.transform import Rotation
    q_wxyz = vec[0:4]
    t = vec[4:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


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
    p.add_argument("--use-sfm", action="store_true", default=False,
                   help="Try using COLMAP SfM for reference model (may crash)")
    args = p.parse_args()

    run_order(
        dataset_root=args.dataset_root,
        method=args.method,
        order_index=args.order_index,
        max_submaps=args.max_submaps,
        traj_eval_data_root=args.traj_eval_data_root,
        skip_eval_export=args.skip_eval_export,
    )
