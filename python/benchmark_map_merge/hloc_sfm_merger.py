"""HLoc SfM-based submap merging: SfM reconstruction + NetVLAD retrieval + PnP localization.

Provides HlocSfmMapMerger with two public methods:
  build_ref_map()    – build a pycolmap.Reconstruction from reference submap images
  localize_submap()  – localize incoming submap images against the reference SfM map
                       using NetVLAD retrieval + SuperPoint+LightGlue + PnP
"""
import sys
import logging
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
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

logger = logging.getLogger(__name__)

_FEATURE_CONF   = extract_features.confs["superpoint_max"]
_RETRIEVAL_CONF = extract_features.confs["netvlad"]
_MATCHER_CONF   = match_features.confs["superpoint+lightglue"]
_LOC_CONF = {
    "estimation": {},   # use pycolmap defaults; abs_pose_min_num_inliers not in 0.6.0
    "refinement": {"refine_focal_length": False, "refine_extra_params": False},
}
_NUM_RETRIEVAL = 10  # top-10 ref frames per query (vs 20; SfM DB is rich enough)


def _vec_to_T(vec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    q_wxyz = vec[0:4]
    t = vec[4:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


def sample_frames_by_vio_distance(
    vio_poses: Dict[str, np.ndarray],
    image_list: List[str],
    min_dist: float = 0.5,
) -> List[str]:
    """Downsample frames by cumulative VIO translation distance.

    Always keeps the first and last frame; intermediate frames are kept only
    when their VIO translation differs from the previous kept frame by at
    least min_dist (meters).

    Args:
        vio_poses: {img_name: [qw,qx,qy,qz,tx,ty,tz]} VIO poses (local frame)
        image_list: ordered image names
        min_dist: minimum translation distance (m) between kept frames

    Returns:
        Subset of image_list, preserving original order.
    """
    selected: List[str] = [image_list[0]]
    last_t = np.array(vio_poses[image_list[0]][4:7], dtype=np.float64)
    for img in image_list[1:]:
        if img not in vio_poses:
            continue
        t = np.array(vio_poses[img][4:7], dtype=np.float64)
        if np.linalg.norm(t - last_t) >= min_dist:
            selected.append(img)
            last_t = t
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
    """Generate SfM matching pairs using VIO pose proximity + sequential neighbours.

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

    translations = np.array([vio_poses[img][4:7] for img in valid_images], dtype=np.float64)
    dists = np.linalg.norm(translations[:, None] - translations[None, :], axis=-1)

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


def _propagate_vio_fallback(
    pnp_results: Dict[str, np.ndarray],
    inc_vio: Dict[str, np.ndarray],
    inc_images: List[str],
) -> Dict[str, np.ndarray]:
    """Fill PnP-failed frames by propagating VIO odometry from nearest successful frame.

    For each failed frame, finds the closest successful frame by index distance,
    then computes:
        T_fail_world = T_anchor_world @ T_anchor_vio^{-1} @ T_fail_vio

    Args:
        pnp_results: {img_name: 4x4 C2W matrix} from successful PnP
        inc_vio: {img_name: [qw,qx,qy,qz,tx,ty,tz]} VIO poses in local frame
        inc_images: ordered list of inc image names

    Returns:
        dict with all frames filled (successful PnP + VIO-propagated estimates).
    """
    n_total = len(inc_images)
    n_pnp = len(pnp_results)
    if n_pnp == 0:
        logger.warning("No PnP successes, cannot propagate VIO fallback")
        return pnp_results

    filled: Dict[str, np.ndarray] = dict(pnp_results)

    success_idxs = [i for i, img in enumerate(inc_images) if img in pnp_results]
    for i, img in enumerate(inc_images):
        if img in filled:
            continue
        nearest_idx = min(success_idxs, key=lambda j: abs(j - i))
        anchor_img = inc_images[nearest_idx]

        T_anchor_world = pnp_results[anchor_img]
        T_anchor_vio = _vec_to_T(inc_vio[anchor_img])
        T_fail_vio = _vec_to_T(inc_vio[img])

        T_fail_world = T_anchor_world @ np.linalg.inv(T_anchor_vio) @ T_fail_vio
        filled[img] = T_fail_world

    logger.info(f"VIO fallback: filled {len(filled) - n_pnp}/{n_total - n_pnp} missing frames")
    return filled


class HlocSfmMapMerger:
    """SfM-based submap merging using NetVLAD retrieval and PnP localization."""

    def __init__(self, out_dir: Path) -> None:
        """
        Args:
            out_dir: scratch directory for all hloc intermediate files (_work/)
        """
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def build_ref_map(
        self,
        ref_dir: Path,
        ref_images: List[str],
        intrinsics: Tuple[float, float, float, float, int, int],
        vio_poses: Optional[Dict[str, np.ndarray]] = None,
        num_pose_neighbors: int = 10,
        seq_window: int = 3,
        sfm_sample_dist: float = 0.0,
    ) -> Optional[pycolmap.Reconstruction]:
        """Build a SfM 3D map from the reference submap.

        Args:
            ref_dir: reference submap root directory (contains seq/*.color.jpg)
            ref_images: ordered list of image names, e.g. ['seq/000000.color.jpg', ...]
            intrinsics: (fx, fy, cx, cy, width, height)
            vio_poses: optional VIO poses {img_name: [qw,qx,qy,qz,tx,ty,tz]}
                       for pose-guided pair generation instead of exhaustive.
            num_pose_neighbors: top-K spatial neighbours per image when using VIO pairs
            seq_window: additional sequential window (±N frames) to supplement pairs
            sfm_sample_dist: if > 0 and vio_poses is given, downsample ref_images
                             by VIO translation distance before SfM reconstruction.
                             NetVLAD features are still extracted on full ref_images.

        Returns:
            pycolmap.Reconstruction on success, None on failure.
        """
        sfm_dir      = self.out_dir / "sfm"
        features_ref = self.out_dir / "feats-ref.h5"
        sfm_pairs    = self.out_dir / "pairs-sfm.txt"
        sfm_matches  = self.out_dir / "matches-sfm.h5"
        global_feats = self.out_dir / "global-feats-netvlad.h5"
        if sfm_dir.exists():
            shutil.rmtree(sfm_dir)
        sfm_dir.mkdir(parents=True, exist_ok=True)

        # SfM reconstruction only uses sampled frames; NetVLAD stays on full set
        if sfm_sample_dist > 0.0 and vio_poses is not None:
            sfm_images = sample_frames_by_vio_distance(vio_poses, ref_images, min_dist=sfm_sample_dist)
        else:
            sfm_images = ref_images

        print(f"[build_ref_map] extracting SuperPoint features for "
              f"{len(ref_images)} ref images (SfM on {len(sfm_images)} sampled)...")
        extract_features.main(
            _FEATURE_CONF, ref_dir, image_list=ref_images,
            feature_path=features_ref, overwrite=False,
        )

        if vio_poses is not None:
            print(f"[build_ref_map] generating VIO-guided SfM pairs "
                  f"(k={num_pose_neighbors}, seq_window={seq_window}) "
                  f"on {len(sfm_images)} sampled frames...")
            _make_vio_guided_pairs(vio_poses, sfm_images, sfm_pairs,
                                    num_neighbors=num_pose_neighbors,
                                    seq_window=seq_window)
        else:
            print(f"[build_ref_map] generating exhaustive SfM pairs "
                  f"on {len(sfm_images)} frames...")
            pairs_from_exhaustive.main(sfm_pairs, image_list=sfm_images)

        print(f"[build_ref_map] running LightGlue matching...")
        match_features.main(
            _MATCHER_CONF, sfm_pairs,
            features=features_ref, matches=sfm_matches, overwrite=False,
        )

        print(f"[build_ref_map] running COLMAP incremental SfM on {len(sfm_images)} frames...")
        fx, fy, cx, cy, w, h = intrinsics
        model = hloc_reconstruction.main(
            sfm_dir, ref_dir, sfm_pairs, features_ref, sfm_matches,
            camera_mode=pycolmap.CameraMode.SINGLE,
            image_list=sfm_images,
            image_options={"camera_model": "PINHOLE",
                           "camera_params": f"{fx},{fy},{cx},{cy}"},
            mapper_options={"min_num_matches": 5, "init_min_num_inliers": 5,
                            "ba_local_num_images": 6},
        )

        if model is None or len(model.points3D) < 20:
            n = 0 if model is None else len(model.points3D)
            print(f"[build_ref_map] SfM failed or too few 3D points ({n})")
            logger.warning(f"SfM failed or too few 3D points ({n})")
            return None

        print(f"[build_ref_map] SfM succeeded: {len(model.images)} images registered, "
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

        print(f"[build_ref_map] extracting NetVLAD global features for retrieval...")
        extract_features.main(
            _RETRIEVAL_CONF, ref_dir, image_list=ref_images,
            feature_path=global_feats, overwrite=False,
        )
        print(f"[build_ref_map] done.")
        return model

    def localize_submap(
        self,
        model: pycolmap.Reconstruction,
        ref_dir: Path,
        ref_images: List[str],
        inc_dir: Path,
        inc_images: List[str],
        intrinsics: Tuple[float, float, float, float, int, int],
        submap_idx: int,
        inc_vio_poses: Optional[Dict[str, np.ndarray]] = None,
        pnp_sample_dist: float = 0.0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """Localize incoming submap images against the SfM map via NetVLAD + PnP.

        Inc images are extracted with an 'inc_' prefix to avoid h5 key collisions
        with reference images that share the same local seq/XXXXXX naming.

        Args:
            model: pycolmap.Reconstruction from build_ref_map()
            ref_dir: reference submap directory
            ref_images: reference image name list
            inc_dir: incoming submap directory
            inc_images: incoming image name list (local seq/XXXXXX.color.jpg)
            intrinsics: (fx, fy, cx, cy, width, height)
            submap_idx: index used for unique per-submap scratch file naming
            inc_vio_poses: optional VIO poses {img_name: [qw,qx,qy,qz,tx,ty,tz]}
                           used to propagate odometry for PnP-failed frames
                           AND to fill unlocalized frames when pnp_sample_dist > 0.
            pnp_sample_dist: if > 0, downsample inc_images by VIO distance before
                             SuperPoint+NetVLAD+LightGlue+PnP, then propagate VIO
                             odometry to fill the full set.  Dramatically reduces
                             feature extraction and matching cost (e.g. 1.0m → ~65%
                             fewer loc pairs).  Set to 0 to localize all frames.

        Returns:
            Tuple of:
              - {inc_img_name: 4x4 C2W numpy float64} in W0 frame
                (all images, VIO fallback fills PnP failures + unsampled frames)
              - {inc_img_name: ref_img_name} primary reference frame per localized image
                (only PnP-successful frames; used to build cross-submap BetweenFactors)
        """
        loc_dir          = self.out_dir / f"loc_sub{submap_idx}"
        feats_inc        = loc_dir / "feats-inc.h5"
        global_feats     = self.out_dir / "global-feats-netvlad.h5"
        global_feats_inc = loc_dir / "global-feats-netvlad-inc.h5"
        loc_pairs        = loc_dir / "pairs-loc.txt"
        feats_merged     = loc_dir / "feats-merged.h5"
        loc_matches      = loc_dir / "matches-loc.h5"
        loc_dir.mkdir(parents=True, exist_ok=True)

        full_inc_images = inc_images
        if pnp_sample_dist > 0.0 and inc_vio_poses is not None:
            pnp_images = sample_frames_by_vio_distance(
                inc_vio_poses, inc_images, min_dist=pnp_sample_dist,
            )
            print(f"[localize_submap] submap{submap_idx}: PnP sampling "
                  f"{len(inc_images)} → {len(pnp_images)} frames "
                  f"(dist={pnp_sample_dist}m)")
        else:
            pnp_images = inc_images

        # Use 'inc_' prefix to avoid key collision with ref images in merged h5.
        # Both ref and inc use seq/XXXXXX.color.jpg locally; the prefix makes
        # them distinguishable when stored in the same h5 file.
        inc_prefixed = [f"inc_{img}" for img in pnp_images]
        orig_to_prefixed = dict(zip(pnp_images, inc_prefixed))
        prefixed_to_orig = dict(zip(inc_prefixed, pnp_images))

        # Symlink inc images to a temp dir with prefixed names so hloc can read them
        tmp_inc_dir = loc_dir / "_tmp_inc"
        tmp_inc_dir.mkdir(parents=True, exist_ok=True)
        for img, prefixed in orig_to_prefixed.items():
            src = (inc_dir / img).resolve()
            dst = tmp_inc_dir / prefixed
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(src)

        print(f"[localize_submap] submap{submap_idx}: extracting SuperPoint features "
              f"for {len(pnp_images)} inc images...")
        extract_features.main(
            _FEATURE_CONF, tmp_inc_dir, image_list=inc_prefixed,
            feature_path=feats_inc, overwrite=False,
        )
        print(f"[localize_submap] submap{submap_idx}: extracting NetVLAD features...")
        extract_features.main(
            _RETRIEVAL_CONF, tmp_inc_dir, image_list=inc_prefixed,
            feature_path=global_feats_inc, overwrite=False,
        )

        # Retrieval: query = inc_prefixed names, db = ref names
        k = min(_NUM_RETRIEVAL, len(ref_images))
        print(f"[localize_submap] submap{submap_idx}: NetVLAD retrieval top-{k} "
              f"({len(inc_prefixed)} queries vs {len(ref_images)} db)...")
        pairs_from_retrieval.main(
            global_feats_inc, loc_pairs, k,
            query_list=inc_prefixed,
            db_list=ref_images,
            db_descriptors=global_feats,
        )

        # Merge features: ref h5 + inc h5 (no key collision since inc uses 'inc_' prefix)
        feats_merged.unlink(missing_ok=True)
        shutil.copy(self.out_dir / "feats-ref.h5", feats_merged)
        with h5py.File(feats_inc, "r") as src, h5py.File(feats_merged, "a") as dst:
            for key in src.keys():
                if key not in dst:
                    src.copy(key, dst)

        match_features.main(
            _MATCHER_CONF, loc_pairs,
            features=feats_merged, matches=loc_matches, overwrite=False,
        )
        print(f"[localize_submap] submap{submap_idx}: running PnP on "
              f"{len(pnp_images)} sampled images against {len(ref_images)} ref frames...")
        pnp_results, pnp_ref_frames = self._run_pnp(
            model, inc_prefixed, prefixed_to_orig,
            loc_pairs, feats_merged, loc_matches, intrinsics,
        )

        if inc_vio_poses is not None and len(pnp_results) < len(full_inc_images):
            n_before = len(pnp_results)
            pnp_results = _propagate_vio_fallback(pnp_results, inc_vio_poses, full_inc_images)
            n_filled = len(pnp_results) - n_before
            n_miss = len(full_inc_images) - n_before
            print(f"[localize_submap] submap{submap_idx}: VIO fallback filled "
                  f"{n_filled}/{n_miss} frames "
                  f"(PnP on {len(pnp_images)} sampled, {len(full_inc_images)} total)")
        return pnp_results, pnp_ref_frames

    def _run_pnp(
        self,
        model: pycolmap.Reconstruction,
        inc_prefixed: List[str],
        prefixed_to_orig: Dict[str, str],
        loc_pairs: Path,
        features: Path,
        matches: Path,
        intrinsics: Tuple[float, float, float, float, int, int],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """Run PnP for each incoming image using prefixed names.

        Args:
            inc_prefixed: list of 'inc_seq/XXXXXX.color.jpg' names
            prefixed_to_orig: mapping from prefixed name to original local name

        Returns:
            Tuple of:
              - {original_inc_img_name: 4x4 C2W} in W0 frame
              - {original_inc_img_name: ref_img_name} primary reference frame per query
                (used to build cross-submap BetweenFactors in the pose graph)
        """
        if not loc_pairs.exists() or loc_pairs.stat().st_size == 0:
            logger.warning("No retrieval pairs found for PnP, returning empty results")
            return {}, {}

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
        n_failed = 0
        for idx, prefixed in enumerate(inc_prefixed):
            if (idx + 1) % 100 == 0:
                print(f"[PnP] progress: {idx+1}/{len(inc_prefixed)} frames processed, "
                      f"{len(results)} localized so far...")
            orig_name = prefixed_to_orig[prefixed]
            db_ids = query_to_dbs.get(prefixed, [])
            if not db_ids:
                n_failed += 1
                continue
            try:
                ret, _ = pose_from_cluster(
                    localizer, prefixed, query_cam, db_ids, features, matches,
                )
            except Exception as e:
                logger.debug(f"PnP exception for {prefixed}: {e}")
                n_failed += 1
                continue

            # pycolmap 0.6 returns no 'success' key; treat num_inliers >= 3 as success
            if ret is None or ret.get("num_inliers", 0) < 3:
                n_failed += 1
                continue

            w2c = np.vstack([ret["cam_from_world"].matrix(), [0.0, 0.0, 0.0, 1.0]])
            results[orig_name] = np.linalg.inv(w2c)
            # record primary reference frame (first db image = top retrieval result)
            pnp_ref_frames[orig_name] = id_to_name[db_ids[0]]

        print(f"[PnP] done: {len(results)}/{len(inc_prefixed)} localized, {n_failed} failed")
        logger.info(f"PnP: {len(results)}/{len(inc_prefixed)} localized, {n_failed} failed")
        return results, pnp_ref_frames
