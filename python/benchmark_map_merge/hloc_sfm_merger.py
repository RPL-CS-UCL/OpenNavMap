"""HLoc SfM-based submap merging: SfM reconstruction + NetVLAD retrieval + PnP localization.

Provides HlocSfmMapMerger with two public methods:
  build_ref_map()    – build a pycolmap.Reconstruction from reference submap images
  localize_submap()  – localize incoming submap images against the reference SfM map
                       using NetVLAD retrieval + local features + LightGlue + PnP
"""
import sys
import time
import logging
import cv2
import h5py
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ESTIMATOR_DIR = str(_REPO_ROOT / "third_party" / "pose_estimation_models" / "estimator")
_HLOC_DIR = str(Path(_ESTIMATOR_DIR) / "third_party" / "Hierarchical-Localization")
_LIGHTGLUE_DIR = str(
    _REPO_ROOT / "third_party" / "vismatch" / "vismatch" / "third_party" / "LightGlue"
)
for _p in [_LIGHTGLUE_DIR, _HLOC_DIR, _ESTIMATOR_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_retrieval
from hloc import reconstruction as hloc_reconstruction
from hloc import triangulation as hloc_triangulation
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

logger = logging.getLogger(__name__)

_FEATURE_CONF   = extract_features.confs["superpoint_max"]
_RETRIEVAL_CONF = extract_features.confs["netvlad"]
_MATCHER_CONF   = match_features.confs["superpoint+lightglue"]
_LOCAL_FEATURE_NAME = "feats-sp.h5"
_LOC_CONF = {
    "estimation": {},   # use pycolmap defaults; abs_pose_min_num_inliers not in 0.6.0
    "refinement": {"refine_focal_length": False, "refine_extra_params": False},
}
_NUM_RETRIEVAL = 10  # top-10 ref frames per query (vs 20; SfM DB is rich enough)
_GEO_VERIFY_MIN_MATCHES = 150
_PNP_MIN_INLIERS = 50


def _build_pnp_per_frame_log(
    submap_idx: int,
    sfm_sampled_i: List[str],
    pnp_results: Dict[str, np.ndarray],
    pnp_ref_frames: Dict[str, str],
    pnp_logs: Dict[str, dict],
    failure_samples: List[dict],
    gt_poses_ref: Optional[Dict[str, np.ndarray]] = None,
    gt_poses_inc: Optional[Dict[str, np.ndarray]] = None,
    model0_c2w: Optional[Dict[str, np.ndarray]] = None,
) -> List[dict]:
    failure_by_frame = {failure.get("frame"): failure for failure in failure_samples}
    pnp_per_frame: List[dict] = []

    for orig_name in sfm_sampled_i:
        prefixed = f"inc{submap_idx}/{orig_name}"
        if orig_name in pnp_results:
            log = pnp_logs.get(orig_name, {})
            best_db = pnp_ref_frames.get(orig_name, "")
            trans_error_m = None
            if (gt_poses_ref is not None and gt_poses_inc is not None
                    and model0_c2w is not None
                    and orig_name in gt_poses_inc
                    and best_db in model0_c2w
                    and best_db in gt_poses_ref):
                T_query_w0 = pnp_results[orig_name]
                T_db_w0 = model0_c2w[best_db]
                T_query_gt = _w2c_vec_to_c2w_matrix(gt_poses_inc[orig_name])
                T_db_gt = _w2c_vec_to_c2w_matrix(gt_poses_ref[best_db])
                delta_T_est = np.linalg.inv(T_db_w0) @ T_query_w0
                delta_T_gt = np.linalg.inv(T_db_gt) @ T_query_gt
                delta_T_err = np.linalg.inv(delta_T_gt) @ delta_T_est
                trans_error_m = float(np.linalg.norm(delta_T_err[:3, 3]))

            pnp_per_frame.append({
                "frame": prefixed,
                "best_db": best_db,
                "num_2d3d": int(len(log.get("points3D_ids", []))),
                "num_inliers": int(log.get("num_inliers", 0)),
                "status": "SUCCESS",
                "trans_error_m": trans_error_m,
            })
        else:
            failure = failure_by_frame.get(
                orig_name,
                {"reason": "unknown", "num_inliers": 0, "num_db": 0},
            )
            pnp_per_frame.append({
                "frame": prefixed,
                "num_db": int(failure.get("num_db", 0)),
                "num_inliers": int(failure.get("num_inliers", 0)),
                "status": f"FAIL({failure.get('reason', 'unknown')})",
            })

    return pnp_per_frame


def _h5_image_group(h5_file: h5py.File, image_name: str) -> h5py.Group:
    return h5_file[image_name]


def _match_group(h5_file: h5py.File, query_name: str, db_name: str) -> h5py.Group:
    query_key = query_name.replace("/", "-")
    db_key = db_name.replace("/", "-")
    if query_key in h5_file and db_key in h5_file[query_key]:
        return h5_file[query_key][db_key]
    if db_key in h5_file and query_key in h5_file[db_key]:
        return h5_file[db_key][query_key]
    return h5_file[query_key][db_key]


def _copy_points2d_without_tracks(source_points2d: list) -> list:
    copied_points2d = []
    for point2d in source_points2d:
        copied_points2d.append(pycolmap.Point2D(xy=point2d.xy))
    return copied_points2d


def _fundamental_inlier_count(
    query_keypoints: np.ndarray,
    db_keypoints: np.ndarray,
    matches0: np.ndarray,
) -> int:
    valid = (matches0 >= 0) & (matches0 < len(db_keypoints))
    if int(np.count_nonzero(valid)) < 8:
        return 0

    query_points = np.ascontiguousarray(query_keypoints[valid], dtype=np.float32)
    db_points = np.ascontiguousarray(db_keypoints[matches0[valid]], dtype=np.float32)
    try:
        _, inlier_mask = cv2.findFundamentalMat(
            query_points,
            db_points,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99,
            maxIters=2000,
        )
    except cv2.error:
        return 0
    if inlier_mask is None:
        return 0
    return int(np.count_nonzero(inlier_mask))


def _geometric_verify_pairs(
    loc_pairs: Path,
    features: Path,
    matches: Path,
    out_pairs: Path,
    min_inliers: int = _GEO_VERIFY_MIN_MATCHES,
) -> Dict[str, int]:
    pairs: List[Tuple[str, str]] = []
    with open(loc_pairs) as pairs_file:
        for line in pairs_file:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))

    query_to_scored_pairs: Dict[
        str,
        List[Tuple[int, int, int, Tuple[str, str]]],
    ] = {}
    all_pair_details: List[dict] = []
    total_matches = 0
    with h5py.File(features, "r") as features_h5, h5py.File(matches, "r") as matches_h5:
        for pair_index, (query_name, db_name) in enumerate(pairs):
            query_group = _h5_image_group(features_h5, query_name)
            db_group = _h5_image_group(features_h5, db_name)
            match_group = _match_group(matches_h5, query_name, db_name)
            matches0 = np.asarray(match_group["matches0"])
            valid_count = int(np.count_nonzero(matches0 >= 0))
            total_matches += valid_count
            if valid_count < min_inliers:
                all_pair_details.append({
                    "query": query_name,
                    "db": db_name,
                    "feat_matches": valid_count,
                    "f_inliers": None,
                    "status": "FAIL",
                })
                continue

            inlier_count = _fundamental_inlier_count(
                np.asarray(query_group["keypoints"]),
                np.asarray(db_group["keypoints"]),
                matches0,
            )
            status = "SUCCESS" if inlier_count >= min_inliers else "FAIL"
            all_pair_details.append({
                "query": query_name,
                "db": db_name,
                "feat_matches": valid_count,
                "f_inliers": inlier_count,
                "status": status,
            })
            if status == "SUCCESS":
                query_to_scored_pairs.setdefault(query_name, []).append(
                    (inlier_count, valid_count, -pair_index, (query_name, db_name))
                )

    queries = list(dict.fromkeys(query for query, _ in pairs))
    written_pairs: List[Tuple[str, str]] = []
    num_query_kept = 0
    for query_name in dict.fromkeys(query for query, _ in pairs):
        scored_pairs = query_to_scored_pairs.get(query_name, [])
        scored_pairs.sort(reverse=True)
        if not scored_pairs:
            continue
        num_query_kept += 1
        for _, _, _, pair in scored_pairs:
            written_pairs.append(pair)

    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pairs, "w") as out_file:
        for query_name, db_name in written_pairs:
            out_file.write(f"{query_name} {db_name}\n")

    return {
        "num_query_total": len(queries),
        "num_query_kept": num_query_kept,
        "num_query_dropped": len(queries) - num_query_kept,
        "num_pairs_total": len(pairs),
        "num_pairs_written": len(written_pairs),
        "num_total_matches": total_matches,
        "pairs_detail": all_pair_details,
        "min_inliers": min_inliers,
    }


def _vec_to_T(vec: np.ndarray) -> np.ndarray:
    """Convert dataset W2C vec7 [qw,qx,qy,qz,tx,ty,tz] to a 4x4 matrix."""
    from scipy.spatial.transform import Rotation
    q_wxyz = vec[0:4]
    t = vec[4:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


def _cam_pos_from_vec(vec: np.ndarray) -> np.ndarray:
    """Return camera position in world from a dataset W2C vec7.

    C2W stores the camera center directly in t. This dataset stores W2C, where
    t is not the camera center; the center is recovered as -R^T @ t. TUM files
    store the camera center and C2W rotation.
    """
    from scipy.spatial.transform import Rotation
    q_wxyz = vec[0:4]
    t_w2c = vec[4:7]
    r_w2c = Rotation.from_quat(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    ).as_matrix()
    return -r_w2c.T @ t_w2c


def _r_c2w_from_vec(vec: np.ndarray) -> np.ndarray:
    """Return 3x3 C2W rotation matrix from a dataset W2C vec7 [qw,qx,qy,qz,tx,ty,tz]."""
    from scipy.spatial.transform import Rotation
    q_wxyz = vec[0:4]
    r_w2c = Rotation.from_quat(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    ).as_matrix()
    return r_w2c.T  # C2W rotation = R_w2c^T


def _c2w_to_rigid3d(T_c2w: np.ndarray) -> pycolmap.Rigid3d:
    """Convert a 4x4 C2W pose matrix to pycolmap's W2C cam_from_world."""
    r_w2c = T_c2w[:3, :3].T
    t_w2c = -r_w2c @ T_c2w[:3, 3]
    return pycolmap.Rigid3d(np.hstack([r_w2c, t_w2c.reshape(3, 1)]))


def _image_cam_from_world(img) -> pycolmap.Rigid3d:
    cfw = img.cam_from_world
    return cfw() if callable(cfw) else cfw


def _extract_w2c_vec_from_image(img) -> np.ndarray:
    """Extract dataset-format W2C vec7 from a pycolmap image.

    pycolmap stores camera poses as cam_from_world (W2C). This returns the same
    W2C convention as poses.txt, not TUM/C2W.
    """
    from scipy.spatial.transform import Rotation

    cfw = _image_cam_from_world(img)
    r_w2c = cfw.rotation.matrix()
    t_w2c = cfw.translation
    q_xyzw = Rotation.from_matrix(r_w2c).as_quat()
    return np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2],
         t_w2c[0], t_w2c[1], t_w2c[2]],
        dtype=np.float64,
    )


def _c2w_matrix_to_w2c_vec(T_c2w: np.ndarray) -> np.ndarray:
    """Convert a 4x4 C2W matrix to dataset-format W2C vec7."""
    from scipy.spatial.transform import Rotation

    r_w2c = T_c2w[:3, :3].T
    t_w2c = -r_w2c @ T_c2w[:3, 3]
    q_xyzw = Rotation.from_matrix(r_w2c).as_quat()
    return np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2],
         t_w2c[0], t_w2c[1], t_w2c[2]],
        dtype=np.float64,
    )


def _w2c_vec_to_c2w_matrix(vec: np.ndarray) -> np.ndarray:
    """Convert dataset-format W2C vec7 to a 4x4 C2W matrix."""
    T_w2c = _vec_to_T(vec)
    return np.linalg.inv(T_w2c)


def _image_c2w_matrix(img) -> np.ndarray:
    """Convert a pycolmap image's W2C cam_from_world to a 4x4 C2W matrix."""
    cfw = _image_cam_from_world(img)
    r_w2c = cfw.rotation.matrix()
    t_w2c = cfw.translation
    r_c2w = r_w2c.T
    t_c2w = -r_c2w @ t_w2c
    T_c2w = np.eye(4, dtype=np.float64)
    T_c2w[:3, :3] = r_c2w
    T_c2w[:3, 3] = t_c2w
    return T_c2w


def _build_vio_reference_model(
    sfm_images: List[str],
    vio_poses: Dict[str, np.ndarray],
    intrinsics: Tuple[float, float, float, float, int, int],
) -> pycolmap.Reconstruction:
    """Build a registered COLMAP model from dataset-format VIO W2C poses."""
    fx, fy, cx, cy, width, height = intrinsics
    model = pycolmap.Reconstruction()
    camera_id = 1

    camera = pycolmap.Camera(
        model="PINHOLE",
        width=int(width),
        height=int(height),
        params=[float(fx), float(fy), float(cx), float(cy)],
    )
    camera.camera_id = camera_id

    if hasattr(model, "add_camera_with_trivial_rig"):
        model.add_camera_with_trivial_rig(camera)
    else:
        model.add_camera(camera)

    if hasattr(pycolmap, "Rig") and hasattr(model, "exists_rig") and not model.exists_rig(camera_id):
        rig = pycolmap.Rig()
        rig.rig_id = camera_id
        sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)
        rig.add_ref_sensor(sensor)
        model.add_rig(rig)

    image_id = 1
    for image_name in sfm_images:
        if image_name not in vio_poses:
            continue
        T_w2c = _vec_to_T(vio_poses[image_name])
        cam_from_world = pycolmap.Rigid3d(
            np.hstack([T_w2c[:3, :3], T_w2c[:3, 3].reshape(3, 1)])
        )

        image = pycolmap.Image()
        image.image_id = image_id
        image.name = image_name
        image.camera_id = camera_id
        if hasattr(model, "add_image_with_trivial_frame"):
            model.add_image_with_trivial_frame(image, cam_from_world)
        else:
            image.cam_from_world = cam_from_world
            model.add_image(image)
            if hasattr(model, "register_image"):
                model.register_image(image_id)
        image_id += 1

    return model


def _set_ba_max_iterations(options: pycolmap.BundleAdjustmentOptions, max_iter: int) -> None:
    if hasattr(options, "solver_options"):
        options.solver_options.max_num_iterations = int(max_iter)
    else:
        options.ceres.solver_options.max_num_iterations = int(max_iter)


def _run_light_bundle_adjustment(
    model: pycolmap.Reconstruction,
    max_iter: int,
) -> None:
    if max_iter <= 0:
        return
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.print_summary = False
    if hasattr(options, "refine_extrinsics"):
        options.refine_extrinsics = False
    if hasattr(options, "refine_points3D"):
        options.refine_points3D = True
    if hasattr(options, "refine_rig_from_world"):
        options.refine_rig_from_world = False
    if hasattr(options, "refine_sensor_from_rig"):
        options.refine_sensor_from_rig = False
    _set_ba_max_iterations(options, max_iter)
    pycolmap.bundle_adjustment(model, options)


def _triangulate_with_vio_prior(
    reference_model: pycolmap.Reconstruction,
    sfm_dir: Path,
    submap_dir: Path,
    sfm_pairs: Path,
    features_ref: Path,
    sfm_matches: Path,
    sfm_ba_iter: int = 0,
) -> Optional[pycolmap.Reconstruction]:
    """Triangulate 3D points from known VIO poses, then optionally run light BA."""
    ref_model_dir = sfm_dir / "vio_reference_model"
    tri_dir = sfm_dir / "triangulated"
    if ref_model_dir.exists():
        shutil.rmtree(ref_model_dir)
    if tri_dir.exists():
        shutil.rmtree(tri_dir)
    ref_model_dir.mkdir(parents=True, exist_ok=True)
    reference_model.write_binary(str(ref_model_dir))

    model = hloc_triangulation.main(
        tri_dir,
        ref_model_dir,
        submap_dir,
        sfm_pairs,
        features_ref,
        sfm_matches,
        skip_geometric_verification=True,
        estimate_two_view_geometries=False,
        mapper_options={},
    )
    if model is None:
        return None

    _run_light_bundle_adjustment(model, sfm_ba_iter)
    for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
        src = tri_dir / filename
        dst = sfm_dir / filename
        if src.exists():
            if dst.exists():
                dst.unlink()
            shutil.copy2(str(src), str(dst))
    return model


def _is_registered(model: pycolmap.Reconstruction, image_id: int, img) -> bool:
    if hasattr(img, "registered"):
        return bool(img.registered)
    if hasattr(model, "reg_image_ids"):
        return int(image_id) in set(model.reg_image_ids())
    return True


def sample_frames_by_vio_distance(
    vio_poses: Dict[str, np.ndarray],
    image_list: List[str],
    min_dist: float = 0.5,
) -> List[str]:
    """Downsample frames by cumulative VIO camera-center distance.

    Always keeps the first and last frame; intermediate frames are kept only
    when their recovered W2C camera center differs from the previous kept frame by at
    least min_dist (meters).

    Args:
        vio_poses: {img_name: [qw,qx,qy,qz,tx,ty,tz]} VIO poses (local frame)
        image_list: ordered image names
        min_dist: minimum translation distance (m) between kept frames

    Returns:
        Subset of image_list, preserving original order.
    """
    selected: List[str] = [image_list[0]]
    last_pos = _cam_pos_from_vec(vio_poses[image_list[0]])
    for img in image_list[1:]:
        if img not in vio_poses:
            continue
        pos = _cam_pos_from_vec(vio_poses[img])
        if np.linalg.norm(pos - last_pos) >= min_dist:
            selected.append(img)
            last_pos = pos
    if image_list[-1] not in selected:
        selected.append(image_list[-1])
    print(f"[sample_frames_by_vio_distance] {len(image_list)} → {len(selected)} frames "
          f"(min_dist={min_dist}m)")
    return selected


def _make_vio_guided_pairs(
    vio_poses: Dict[str, np.ndarray],
    image_list: List[str],
    output_path: Path,
    num_neighbors: int = 10,
    seq_window: int = 3,
) -> None:
    """Generate SfM matching pairs using VIO camera proximity + sequential neighbours.

    Replaces pairs_from_exhaustive with a sparser, higher-overlap pair set
    guided by VIO odometry translations, improving SfM reconstruction coverage.

    Args:
        vio_poses: {img_name: [qw,qx,qy,qz,tx,ty,tz]} in local VIO frame
        image_list: ordered image names, e.g. ['seq/000000.color.jpg', ...]
        output_path: where to write the pair list (one pair per line)
        num_neighbors: top-K spatial neighbours per image
        seq_window: sequential window (±seq_window frames)
    """
    if not vio_poses:
        logger.warning("VIO poses empty, falling back to exhaustive pairs")
        pairs_from_exhaustive.main(output_path, image_list=image_list)
        return

    img_to_idx = {img: i for i, img in enumerate(image_list)}
    valid_images = [img for img in image_list if img in vio_poses]
    if len(valid_images) < 3:
        logger.warning(f"Too few VIO poses ({len(valid_images)}), falling back to exhaustive")
        pairs_from_exhaustive.main(output_path, image_list=image_list)
        return

    cam_positions = np.array(
        [_cam_pos_from_vec(vio_poses[img]) for img in valid_images],
        dtype=np.float64,
    )
    dists = np.linalg.norm(cam_positions[:, None] - cam_positions[None, :], axis=-1)

    pairs_set: set = set()

    for i in range(len(valid_images)):
        order = np.argsort(dists[i])
        count = 0
        for j in order:
            if j == i:
                continue
            if count >= num_neighbors:
                break
            a, b = valid_images[i], valid_images[j]
            pairs_set.add((a, b) if img_to_idx[a] < img_to_idx[b] else (b, a))
            count += 1

    for i in range(len(image_list)):
        for d in range(1, seq_window + 1):
            j = i + d
            if j >= len(image_list):
                break
            pairs_set.add((image_list[i], image_list[j]))

    logger.info(f"VIO-guided pairs: {len(pairs_set)} (pose_neighbors={num_neighbors}, seq_window={seq_window})")
    with open(output_path, "w") as f:
        for a, b in sorted(pairs_set):
            f.write(f"{a} {b}\n")


def _estimate_se3_umeyama(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_threshold: float = 0.5,
    min_inliers: int = 3,
    num_ransac_iter: int = 200,
) -> Tuple[Optional[np.ndarray], List[int]]:
    """Estimate SE(3) from source to target camera centers using Umeyama."""

    def _fit_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        from numpy.linalg import svd

        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        covariance = dst_centered.T @ src_centered / src.shape[0]
        U, _, Vt = svd(covariance)
        rotation = U @ Vt
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = U @ Vt
        translation = dst_mean - rotation @ src_mean

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    if src_pts.shape != dst_pts.shape or src_pts.ndim != 2 or src_pts.shape[1] != 3:
        return None, []
    num_points = src_pts.shape[0]
    if num_points < min_inliers:
        return None, []

    def _inliers(transform: np.ndarray) -> List[int]:
        transformed = (transform[:3, :3] @ src_pts.T + transform[:3, 3:4]).T
        residuals = np.linalg.norm(transformed - dst_pts, axis=1)
        return [idx for idx, residual in enumerate(residuals) if residual <= ransac_threshold]

    if num_points <= min_inliers:
        transform = _fit_se3(src_pts, dst_pts)
        inliers = _inliers(transform)
        if len(inliers) < min_inliers:
            return None, []
        return transform, inliers

    rng = np.random.default_rng(42)
    best_inliers: List[int] = []
    for _ in range(num_ransac_iter):
        sample_indices = rng.choice(num_points, min_inliers, replace=False)
        candidate = _fit_se3(src_pts[sample_indices], dst_pts[sample_indices])
        candidate_inliers = _inliers(candidate)
        if len(candidate_inliers) > len(best_inliers):
            best_inliers = candidate_inliers

    if len(best_inliers) < min_inliers:
        return None, []
    inlier_indices = np.array(best_inliers, dtype=np.int64)
    transform = _fit_se3(src_pts[inlier_indices], dst_pts[inlier_indices])
    return transform, best_inliers


class HlocSfmMapMerger:
    """SfM-based submap merging using NetVLAD retrieval and PnP localization."""

    def __init__(
        self,
        out_dir: Path,
        feature_conf: Optional[dict] = None,
        retrieval_conf: Optional[dict] = None,
        matcher_conf: Optional[dict] = None,
        local_feature_name: str = _LOCAL_FEATURE_NAME,
        num_retrieval: int = _NUM_RETRIEVAL,
        geo_verify_min_matches: int = _GEO_VERIFY_MIN_MATCHES,
        pnp_min_inliers: int = _PNP_MIN_INLIERS,
    ) -> None:
        """
        Args:
            out_dir: scratch directory for all hloc intermediate files (_work/)
            feature_conf: local feature extractor config.
            retrieval_conf: global retrieval feature config.
            matcher_conf: local feature matcher config.
            local_feature_name: h5 filename for cached submap local features.
        """
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.feature_conf = feature_conf or _FEATURE_CONF
        self.retrieval_conf = retrieval_conf or _RETRIEVAL_CONF
        self.matcher_conf = matcher_conf or _MATCHER_CONF
        self.local_feature_name = local_feature_name
        self.num_retrieval = num_retrieval
        self.geo_verify_min_matches = geo_verify_min_matches
        self.pnp_min_inliers = pnp_min_inliers
        self.last_sfm_sampled_frames = 0

    @staticmethod
    def extract_w2c_vec_from_image(img) -> np.ndarray:
        """Extract dataset-format W2C vec7 from a pycolmap image."""
        return _extract_w2c_vec_from_image(img)

    def build_submap_sfm(
        self,
        submap_dir: Path,
        submap_images: List[str],
        intrinsics: Tuple[float, float, float, float, int, int],
        vio_poses: Optional[Dict[str, np.ndarray]] = None,
        num_pose_neighbors: int = 15,
        seq_window: int = 5,
        sfm_sample_dist: float = 0.0,
        use_vio_prior: bool = True,
        sfm_ba_iter: int = 0,
        overwrite: bool = False,
        submap_tag: str = "sub0",
    ) -> Optional[pycolmap.Reconstruction]:
        """Build a SfM 3D map from a submap in an isolated work directory.

        Args:
            submap_dir: submap root directory (contains seq/*.color.jpg)
            submap_images: ordered image names, e.g. ['seq/000000.color.jpg', ...]
            intrinsics: (fx, fy, cx, cy, width, height)
            vio_poses: optional VIO poses {img_name: [qw,qx,qy,qz,tx,ty,tz]}
                       for pose-guided pair generation instead of exhaustive.
            num_pose_neighbors: top-K spatial neighbours per image when using VIO pairs
            seq_window: additional sequential window (±N frames) to supplement pairs
            sfm_sample_dist: if > 0 and vio_poses is given, downsample ref_images
                             by VIO translation distance before SfM reconstruction.
                             NetVLAD features are still extracted on full ref_images.
            use_vio_prior: if True and vio_poses is given, use VIO poses as known
                           COLMAP poses, triangulate points, then optionally run BA.
                           If False or no VIO exists, use free incremental SfM.
            sfm_ba_iter: BA iterations after VIO-prior triangulation. 0 keeps poses fixed.
            overwrite: if True, ignore cached SfM files and rebuild.

        Returns:
            pycolmap.Reconstruction on success, None on failure.
        """
        tag_dir      = self.out_dir / submap_tag
        sfm_dir      = tag_dir / "sfm"
        features_ref = tag_dir / self.local_feature_name
        sfm_pairs    = tag_dir / "pairs-sfm.txt"
        sfm_matches  = tag_dir / "matches-sfm.h5"
        global_feats = tag_dir / "feats-netvlad.h5"
        sfm_model_files = ["cameras.bin", "images.bin", "points3D.bin"]
        if not overwrite and all((sfm_dir / fn).is_file() for fn in sfm_model_files):
            model = pycolmap.Reconstruction()
            model.read_binary(str(sfm_dir))
            self.last_sfm_sampled_frames = sum(
                1 for img_id, img in model.images.items()
                if _is_registered(model, img_id, img)
            )
            print(f"[build_submap_sfm] cache hit: loaded SfM model from {sfm_dir} "
                  f"({self.last_sfm_sampled_frames} registered images, "
                  f"{len(model.points3D)} 3D points)")
            return model

        if sfm_dir.exists():
            shutil.rmtree(sfm_dir)
        sfm_dir.mkdir(parents=True, exist_ok=True)

        # SfM reconstruction only uses sampled frames; NetVLAD stays on full set
        if sfm_sample_dist > 0.0 and vio_poses is not None:
            sfm_images = sample_frames_by_vio_distance(vio_poses, submap_images, min_dist=sfm_sample_dist)
        else:
            sfm_images = submap_images
        self.last_sfm_sampled_frames = len(sfm_images)

        print(f"[build_submap_sfm] extracting local features for "
              f"{len(submap_images)} images (SfM on {len(sfm_images)} sampled)...")
        extract_features.main(
            self.feature_conf, submap_dir, image_list=submap_images,
            feature_path=features_ref, overwrite=False,
        )

        if vio_poses is not None:
            print(f"[build_submap_sfm] generating VIO-guided SfM pairs "
                  f"(k={num_pose_neighbors}, seq_window={seq_window}) "
                  f"on {len(sfm_images)} sampled frames...")
            _make_vio_guided_pairs(vio_poses, sfm_images, sfm_pairs,
                                    num_neighbors=num_pose_neighbors,
                                    seq_window=seq_window)
        else:
            print(f"[build_submap_sfm] generating exhaustive SfM pairs "
                  f"on {len(sfm_images)} frames...")
            pairs_from_exhaustive.main(sfm_pairs, image_list=sfm_images)

        print(f"[build_submap_sfm] running LightGlue matching...")
        match_features.main(
            self.matcher_conf, sfm_pairs,
            features=features_ref, matches=sfm_matches, overwrite=False,
        )

        fx, fy, cx, cy, w, h = intrinsics
        if vio_poses is not None and use_vio_prior:
            print(f"[build_submap_sfm] running VIO-prior triangulation "
                  f"on {len(sfm_images)} frames (ba_iter={sfm_ba_iter})...")
            reference_model = _build_vio_reference_model(sfm_images, vio_poses, intrinsics)
            model = _triangulate_with_vio_prior(
                reference_model, sfm_dir, submap_dir,
                sfm_pairs, features_ref, sfm_matches,
                sfm_ba_iter=sfm_ba_iter,
            )
        else:
            print(f"[build_submap_sfm] running COLMAP incremental SfM on {len(sfm_images)} frames...")
            model = hloc_reconstruction.main(
                sfm_dir, submap_dir, sfm_pairs, features_ref, sfm_matches,
                camera_mode=pycolmap.CameraMode.SINGLE,
                image_list=sfm_images,
                image_options={"camera_model": "PINHOLE",
                               "camera_params": f"{fx},{fy},{cx},{cy}"},
                mapper_options={"min_num_matches": 15, "init_min_num_inliers": 15,
                                "ba_local_num_images": 10},
            )

        if model is None or len(model.points3D) < 20:
            n = 0 if model is None else len(model.points3D)
            print(f"[build_submap_sfm] SfM failed or too few 3D points ({n})")
            logger.warning(f"SfM failed or too few 3D points ({n})")
            return None

        print(f"[build_submap_sfm] SfM succeeded: {len(model.images)} images registered, "
              f"{len(model.points3D)} 3D points")
        logger.info(f"SfM: {len(model.images)} images triangulated, "
                    f"{len(model.points3D)} 3D points")

        # Ensure root sfm_dir has the correct model files (run_reconstruction may have
        # failed to overwrite them if the database already wrote sparse binary files).
        for _fn in ["images.bin", "cameras.bin", "points3D.bin"]:
            _dst = sfm_dir / _fn
            _src = sfm_dir / "models" / "0" / _fn
            if _src.exists():
                if _dst.exists():
                    _dst.unlink()
                shutil.move(str(_src), str(_dst))

        print(f"[build_submap_sfm] extracting NetVLAD global features for retrieval...")
        extract_features.main(
            self.retrieval_conf, submap_dir, image_list=submap_images,
            feature_path=global_feats, overwrite=False,
        )
        print(f"[build_submap_sfm] done.")
        return model

    @staticmethod
    def _append_features(src_path: Path, dst_path: Path) -> None:
        """Append image feature groups from src h5 into dst h5 for future map queries."""
        with h5py.File(src_path, "r") as src, h5py.File(dst_path, "a") as dst:
            for key in src.keys():
                if key not in dst:
                    src.copy(key, dst)

    def merge_model_with_se3(
        self,
        model0: pycolmap.Reconstruction,
        model_i: pycolmap.Reconstruction,
        ref_dir: Path,
        ref_images: List[str],
        inc_dir: Path,
        sfm_sampled_i: List[str],
        intrinsics: Tuple[float, float, float, float, int, int],
        submap_idx: int,
        submap_tag: str,
        gt_poses_ref: Optional[Dict[str, np.ndarray]] = None,
        gt_poses_inc: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[
        Optional[pycolmap.Reconstruction],
        Optional[Dict[str, np.ndarray]],
        Dict[str, str],
        dict,
    ]:
        """Merge an independently reconstructed submap into model0 via PnP + SE(3).

        Pipeline:
          1. NetVLAD retrieval: find cross-session candidate pairs (inc→ref).
          2. SuperPoint+LightGlue matching + geometric verification.
          3. PnP: localize sampled inc frames in model0 frame (per-frame diagnostics logged).
          4. SE(3) estimation: RANSAC over (model_i camera centers → PnP camera centers).
          5. Rigid transform: ALL model_i images+points transformed by T_i_to_0 and merged
             into model0 with full track linkage preserved. No BA.
        """
        loc_dir = self.out_dir / f"merge_sub{submap_idx}"
        feats_inc = loc_dir / "feats-inc.h5"
        global_feats = self.out_dir / "global-feats-netvlad.h5"
        global_feats_inc = loc_dir / "global-feats-netvlad-inc.h5"
        loc_pairs = loc_dir / "pairs-loc.txt"
        feats_merged = loc_dir / "feats-merged.h5"
        loc_matches = loc_dir / "matches-loc.h5"
        loc_dir.mkdir(parents=True, exist_ok=True)

        inc_prefix = f"inc{submap_idx}"
        inc_prefixed = [f"{inc_prefix}/{img}" for img in sfm_sampled_i]
        prefixed_to_orig = dict(zip(inc_prefixed, sfm_sampled_i))

        tmp_inc_dir = loc_dir / "_tmp_inc"
        tmp_inc_dir.mkdir(parents=True, exist_ok=True)
        for img, prefixed in zip(sfm_sampled_i, inc_prefixed):
            src = (inc_dir / img).resolve()
            dst = tmp_inc_dir / prefixed
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(src)

        extract_features.main(
            self.feature_conf, tmp_inc_dir, image_list=inc_prefixed,
            feature_path=feats_inc, overwrite=False,
        )
        extract_features.main(
            self.retrieval_conf, tmp_inc_dir, image_list=inc_prefixed,
            feature_path=global_feats_inc, overwrite=False,
        )

        db_images = [
            img.name for img_id, img in model0.images.items()
            if _is_registered(model0, img_id, img)
        ]
        k = min(self.num_retrieval, len(db_images))
        if k == 0:
            return None, None, {}, {
                "submap_idx": int(submap_idx),
                "error": "no registered database images",
            }

        pairs_from_retrieval.main(
            global_feats_inc, loc_pairs, k,
            query_list=inc_prefixed,
            db_list=db_images,
            db_descriptors=global_feats,
        )
        with open(loc_pairs) as retrieval_file:
            num_retrieval_pairs = sum(1 for line in retrieval_file if line.strip())
        retrieval_stats = {
            "num_queries": len(inc_prefixed),
            "num_db": len(db_images),
            "top_k": k,
            "num_pairs": num_retrieval_pairs,
        }

        feats_merged.unlink(missing_ok=True)
        shutil.copy(self.out_dir / "feats-ref.h5", feats_merged)
        with h5py.File(feats_inc, "r") as src, h5py.File(feats_merged, "a") as dst:
            for key in src.keys():
                if key not in dst:
                    src.copy(key, dst)

        match_features.main(
            self.matcher_conf, loc_pairs,
            features=feats_merged, matches=loc_matches, overwrite=False,
        )
        verified_pairs = loc_dir / "pairs-loc-verified.txt"
        gv_stats = _geometric_verify_pairs(
            loc_pairs, feats_merged, loc_matches, verified_pairs,
            min_inliers=self.geo_verify_min_matches,
        )

        # --- Annotate GV pairs_detail with GT TP/FP (pair level, f_inliers >= threshold) ---
        if gt_poses_ref is not None and gt_poses_inc is not None:
            _tp = _fp = _no_gt = 0
            for _pair in gv_stats["pairs_detail"]:
                _f_inliers = _pair["f_inliers"]
                if _f_inliers is None or _f_inliers < self.geo_verify_min_matches:
                    continue
                _q_orig = prefixed_to_orig.get(_pair["query"], _pair["query"])
                _db_name = _pair["db"]
                if _q_orig not in gt_poses_inc or _db_name not in gt_poses_ref:
                    _pair["gv_label"] = "NO_GT"
                    _no_gt += 1
                    continue
                _c_inc = _cam_pos_from_vec(gt_poses_inc[_q_orig])
                _c_ref = _cam_pos_from_vec(gt_poses_ref[_db_name])
                _t_rel = float(np.linalg.norm(_c_inc - _c_ref))
                # relative rotation (C2W frames): R_rel = R_inc_c2w.T @ R_ref_c2w
                _R_inc_c2w = _r_c2w_from_vec(gt_poses_inc[_q_orig])
                _R_ref_c2w = _r_c2w_from_vec(gt_poses_ref[_db_name])
                _R_rel = _R_inc_c2w.T @ _R_ref_c2w
                _theta = float(np.degrees(np.arccos(np.clip((np.trace(_R_rel) - 1.0) / 2.0, -1.0, 1.0))))
                _pair["gt_trans_m"] = round(_t_rel, 4)
                _pair["gt_rot_deg"] = round(_theta, 2)
                if _t_rel < 7.0 and _theta < 90.0:
                    _pair["gv_label"] = "TP"
                    _tp += 1
                else:
                    _pair["gv_label"] = "FP"
                    _fp += 1
            _n_above = _tp + _fp + _no_gt
            gv_stats["geo_verify_tp_fp"] = {
                "threshold_f_inliers": self.geo_verify_min_matches,
                "num_pairs_above_thresh": _n_above,
                "num_tp": _tp,
                "num_fp": _fp,
                "num_no_gt": _no_gt,
                "tp_ratio": round(_tp / _n_above, 4) if _n_above else None,
                "fp_ratio": round(_fp / _n_above, 4) if _n_above else None,
            }

        pnp_results, pnp_ref_frames, pnp_logs, failure_samples = self._run_pnp(
            model0, inc_prefixed, prefixed_to_orig,
            verified_pairs, feats_merged, loc_matches, intrinsics,
        )
        refinement_stats = {"num_pose_refined": 0, "pose_refinement_change_mm": {"mean": 0.0, "max": 0.0}}

        model0_c2w: Dict[str, np.ndarray] = {}
        if gt_poses_ref is not None and gt_poses_inc is not None:
            for _img in model0.images.values():
                model0_c2w[_img.name] = _image_c2w_matrix(_img)
        pnp_per_frame = _build_pnp_per_frame_log(
            submap_idx=submap_idx,
            sfm_sampled_i=sfm_sampled_i,
            pnp_results=pnp_results,
            pnp_ref_frames=pnp_ref_frames,
            pnp_logs=pnp_logs,
            failure_samples=failure_samples,
            gt_poses_ref=gt_poses_ref,
            gt_poses_inc=gt_poses_inc,
            model0_c2w=model0_c2w,
        )

        model_i_by_name = {img.name: img for img in model_i.images.values()}
        src_pts: List[np.ndarray] = []
        dst_pts: List[np.ndarray] = []
        anchor_names: List[str] = []
        for orig_name, T_c2w_w0 in pnp_results.items():
            model_img = model_i_by_name.get(orig_name)
            if model_img is not None:
                src_pts.append(_image_c2w_matrix(model_img)[:3, 3])
                dst_pts.append(T_c2w_w0[:3, 3])
                anchor_names.append(orig_name)

        T_i_to_0, inlier_indices = _estimate_se3_umeyama(
            np.asarray(src_pts, dtype=np.float64),
            np.asarray(dst_pts, dtype=np.float64),
        )
        if T_i_to_0 is None:
            return None, None, {}, {
                "submap_idx": int(submap_idx),
                "error": "SE(3) estimation failed",
                "num_pnp_sampled": len(sfm_sampled_i),
                "num_pnp_success": len(pnp_results),
                "num_se3_inliers": 0,
                "num_images_merged": 0,
                "num_points3D_merged": 0,
                "retrieval": retrieval_stats,
                "geometric_verification": gv_stats,
                "num_pose_refined": refinement_stats["num_pose_refined"],
                "pose_refinement_change_mm": refinement_stats["pose_refinement_change_mm"],
                "sampled_inc_images": sfm_sampled_i,
                "sample_pnp_failures": failure_samples,
                "pnp_per_frame": pnp_per_frame,
            }

        transformed_anchors = (T_i_to_0[:3, :3] @ np.asarray(src_pts).T + T_i_to_0[:3, 3:4]).T
        residuals = np.linalg.norm(transformed_anchors - np.asarray(dst_pts), axis=1)
        inlier_residuals = residuals[inlier_indices] if inlier_indices else np.array([], dtype=np.float64)

        # --- Rigidly transform entire model_i into model0 coordinate frame ---
        # Step 1: transform all 3D points, record old->new point3D id mapping
        R_i0 = T_i_to_0[:3, :3]
        t_i0 = T_i_to_0[:3, 3]
        old_to_new_pid: Dict[int, int] = {}
        for old_pid, point in model_i.points3D.items():
            if point.track.length() == 0:
                continue
            xyz_w0 = R_i0 @ point.xyz + t_i0
            new_pid = model0.add_point3D(xyz_w0, pycolmap.Track(), point.color)
            old_to_new_pid[old_pid] = new_pid

        # Step 2: transform all images and re-link track observations
        model_name_to_orig: Dict[str, str] = {}
        inc_poses_w2c: Dict[str, np.ndarray] = {}
        # ensure camera and rig exist in model0
        camera_id = next(iter(model0.cameras.keys()))
        if hasattr(model0, "exists_rig") and not model0.exists_rig(int(camera_id)):
            rig = pycolmap.Rig()
            rig.rig_id = int(camera_id)
            sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, int(camera_id))
            rig.add_ref_sensor(sensor)
            model0.add_rig(rig)
        existing_ids = set(model0.images.keys())
        next_image_id = (max(existing_ids) + 1) if existing_ids else 1

        for old_img in model_i.images.values():
            orig_name = old_img.name
            new_name = f"inc{submap_idx}/{orig_name}"
            # transform pose: T_c2w_wi -> T_c2w_w0
            T_c2w_wi = _image_c2w_matrix(old_img)
            T_c2w_w0 = T_i_to_0 @ T_c2w_wi
            cam_from_world = _c2w_to_rigid3d(T_c2w_w0)

            new_img = pycolmap.Image()
            new_img.image_id = next_image_id
            new_img.name = new_name
            new_img.camera_id = camera_id
            # copy points2D with point3D_id cleared (will be re-linked via add_observation)
            cleared_p2d = _copy_points2d_without_tracks(old_img.points2D)
            if cleared_p2d:
                new_img.points2D = cleared_p2d
            if hasattr(model0, "add_image_with_trivial_frame"):
                model0.add_image_with_trivial_frame(new_img, cam_from_world)
            else:
                new_img.cam_from_world = cam_from_world
                model0.add_image(new_img)

            # re-link track observations using old->new point3D id
            for p2d_idx, p2d in enumerate(old_img.points2D):
                if not p2d.has_point3D():
                    continue
                old_pid = p2d.point3D_id
                new_pid = old_to_new_pid.get(old_pid)
                if new_pid is not None:
                    te = pycolmap.TrackElement()
                    te.image_id = new_img.image_id
                    te.point2D_idx = p2d_idx
                    model0.add_observation(new_pid, te)

            inc_poses_w2c[orig_name] = _c2w_matrix_to_w2c_vec(T_c2w_w0)
            model_name_to_orig[new_name] = orig_name
            next_image_id += 1

        pnp_errors = [
            e["trans_error_m"] for e in pnp_per_frame
            if e["status"] == "SUCCESS" and e.get("trans_error_m") is not None
        ]
        # pnp_error_stats: only count frames with inliers >= 10 as real success
        _inliers_thresh = 10
        real_success = [
            e for e in pnp_per_frame
            if e["status"] == "SUCCESS" and e.get("num_inliers", 0) >= _inliers_thresh
        ]
        _n_real = len(real_success)
        _n_lt2 = sum(
            1 for e in real_success
            if e.get("trans_error_m") is not None and e["trans_error_m"] < 2.0
        )
        _n_ge2 = sum(
            1 for e in real_success
            if e.get("trans_error_m") is not None and e["trans_error_m"] >= 2.0
        )
        merge_stats = {
            "submap_idx": int(submap_idx),
            "num_pnp_sampled": len(sfm_sampled_i),
            "num_pnp_success": len(pnp_results),
            "num_se3_inliers": len(inlier_indices),
            "se3_anchor_names": [anchor_names[idx] for idx in inlier_indices],
            "se3_residual_mean_m": float(np.mean(inlier_residuals)) if len(inlier_residuals) else 0.0,
            "se3_residual_max_m": float(np.max(inlier_residuals)) if len(inlier_residuals) else 0.0,
            "num_images_merged": len(model_i.images),
            "num_points3D_merged": len(old_to_new_pid),
            "retrieval": retrieval_stats,
            "geometric_verification": gv_stats,
            "num_pose_refined": refinement_stats["num_pose_refined"],
            "pose_refinement_change_mm": refinement_stats["pose_refinement_change_mm"],
            "sampled_inc_images": sfm_sampled_i,
            "sample_pnp_failures": failure_samples,
            "pnp_per_frame": pnp_per_frame,
            "pnp_trans_error_m": {
                "mean": float(np.mean(pnp_errors)) if pnp_errors else None,
                "median": float(np.median(pnp_errors)) if pnp_errors else None,
                "max": float(np.max(pnp_errors)) if pnp_errors else None,
                "num_valid": len(pnp_errors),
            },
            "pnp_error_stats": {
                "inliers_threshold": _inliers_thresh,
                "num_success_total": len(pnp_results),
                "num_real_success": _n_real,
                "num_error_lt2m": _n_lt2,
                "num_error_ge2m": _n_ge2,
                "ratio_lt2m": round(_n_lt2 / _n_real, 4) if _n_real else None,
                "ratio_ge2m": round(_n_ge2 / _n_real, 4) if _n_real else None,
            },
        }
        self._append_features(feats_inc, self.out_dir / "feats-ref.h5")
        self._append_features(global_feats_inc, global_feats)
        return model0, inc_poses_w2c, model_name_to_orig, merge_stats

    def _run_pnp(
        self,
        model: pycolmap.Reconstruction,
        inc_prefixed: List[str],
        prefixed_to_orig: Dict[str, str],
        loc_pairs: Path,
        features: Path,
        matches: Path,
        intrinsics: Tuple[float, float, float, float, int, int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str], Dict[str, dict], List[dict]]:
        """Run PnP for each incoming image using prefixed names.

        Args:
            inc_prefixed: list of 'inc_seq/XXXXXX.color.jpg' names
            prefixed_to_orig: mapping from prefixed name to original local name

        Returns:
            Tuple of:
              - {original_inc_img_name: 4x4 C2W} in W0 frame
              - {original_inc_img_name: ref_img_name} primary reference frame per query
                (retained for diagnostics)
              - {original_inc_img_name: pose_from_cluster log} with 2D-3D matches
              - list of all failed PnP frames with diagnostics
        """
        if not loc_pairs.exists() or loc_pairs.stat().st_size == 0:
            logger.warning("No retrieval pairs found for PnP, returning empty results")
            return {}, {}, {}, [
                {
                    "frame": prefixed_to_orig[prefixed],
                    "reason": "no_retrieval_pairs_file",
                    "num_inliers": 0,
                    "num_db": 0,
                }
                for prefixed in inc_prefixed
            ]

        fx, fy, cx, cy, w, h = intrinsics
        query_cam = pycolmap.Camera(
            model="PINHOLE", width=int(w), height=int(h),
            params=[fx, fy, cx, cy],
        )
        localizer = QueryLocalizer(model, _LOC_CONF)
        name_to_id = {img.name: img_id for img_id, img in model.images.items()}
        id_to_name = {img_id: img.name for img_id, img in model.images.items()}

        query_to_dbs: Dict[str, List[int]] = {img: [] for img in inc_prefixed}
        with open(loc_pairs) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] in query_to_dbs:
                    if parts[1] in name_to_id:
                        query_to_dbs[parts[0]].append(name_to_id[parts[1]])

        results: Dict[str, np.ndarray] = {}
        pnp_ref_frames: Dict[str, str] = {}  # orig_img -> primary ref image name
        pnp_logs: Dict[str, dict] = {}
        failure_samples: List[dict] = []
        n_failed = 0
        for idx, prefixed in enumerate(inc_prefixed):
            if (idx + 1) % 100 == 0:
                print(f"[PnP] progress: {idx+1}/{len(inc_prefixed)} frames processed, "
                      f"{len(results)} localized so far...")
            orig_name = prefixed_to_orig[prefixed]
            db_ids = query_to_dbs.get(prefixed, [])
            if not db_ids:
                failure_samples.append({
                    "frame": orig_name,
                    "reason": "no_retrieval",
                    "num_inliers": 0,
                    "num_db": 0,
                })
                n_failed += 1
                continue
            try:
                ret, log = pose_from_cluster(
                    localizer, prefixed, query_cam, db_ids, features, matches,
                )
            except Exception as e:
                logger.debug(f"PnP exception for {prefixed}: {e}")
                failure_samples.append({
                    "frame": orig_name,
                    "reason": "pnp_exception",
                    "num_inliers": 0,
                    "num_db": len(db_ids),
                })
                n_failed += 1
                continue

            if ret is None or ret.get("num_inliers", 0) < self.pnp_min_inliers:
                failure_samples.append({
                    "frame": orig_name,
                    "reason": "insufficient_inliers",
                    "num_inliers": int(ret.get("num_inliers", 0)) if ret is not None else 0,
                    "num_db": len(db_ids),
                })
                n_failed += 1
                continue

            w2c = np.vstack([ret["cam_from_world"].matrix(), [0.0, 0.0, 0.0, 1.0]])
            results[orig_name] = np.linalg.inv(w2c)
            # record primary reference frame (first db image = top retrieval result)
            pnp_ref_frames[orig_name] = id_to_name[db_ids[0]]
            pnp_logs[orig_name] = ret  # store ret (contains num_inliers) not hloc log

        print(f"[PnP] done: {len(results)}/{len(inc_prefixed)} localized, {n_failed} failed")
        logger.info(f"PnP: {len(results)}/{len(inc_prefixed)} localized, {n_failed} failed")
        return results, pnp_ref_frames, pnp_logs, failure_samples
