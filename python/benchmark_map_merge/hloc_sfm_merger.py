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
    "estimation": {"abs_pose_min_num_inliers": 6},
    "refinement": {"refine_focal_length": False, "refine_extra_params": False},
}
_NUM_RETRIEVAL = 20


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
    ) -> Optional[pycolmap.Reconstruction]:
        """Build a SfM 3D map from the reference submap.

        Args:
            ref_dir: reference submap root directory (contains seq/*.color.jpg)
            ref_images: ordered list of image names, e.g. ['seq/000000.color.jpg', ...]
            intrinsics: (fx, fy, cx, cy, width, height)

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

        extract_features.main(
            _FEATURE_CONF, ref_dir, image_list=ref_images,
            feature_path=features_ref, overwrite=True,
        )
        pairs_from_exhaustive.main(sfm_pairs, image_list=ref_images)
        match_features.main(
            _MATCHER_CONF, sfm_pairs,
            features=features_ref, matches=sfm_matches, overwrite=True,
        )

        fx, fy, cx, cy, w, h = intrinsics
        model = hloc_reconstruction.main(
            sfm_dir, ref_dir, sfm_pairs, features_ref, sfm_matches,
            camera_mode=pycolmap.CameraMode.SINGLE,
            image_list=ref_images,
            image_options={"camera_model": "PINHOLE",
                           "camera_params": f"{fx},{fy},{cx},{cy}"},
            mapper_options={"min_num_matches": 10, "init_min_num_inliers": 10},
        )

        if model is None or len(model.points3D) < 50:
            n = 0 if model is None else len(model.points3D)
            logger.warning(f"SfM failed or too few 3D points ({n})")
            return None

        logger.info(f"SfM: {len(model.images)} images triangulated, "
                    f"{len(model.points3D)} 3D points")

        extract_features.main(
            _RETRIEVAL_CONF, ref_dir, image_list=ref_images,
            feature_path=global_feats, overwrite=True,
        )
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
    ) -> Dict[str, np.ndarray]:
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

        Returns:
            {inc_img_name: 4x4 camera-to-world numpy float64} in W0 frame.
            Failed images are absent.
        """
        loc_dir          = self.out_dir / f"loc_sub{submap_idx}"
        feats_inc        = loc_dir / "feats-inc.h5"
        global_feats     = self.out_dir / "global-feats-netvlad.h5"
        global_feats_inc = loc_dir / "global-feats-netvlad-inc.h5"
        loc_pairs        = loc_dir / "pairs-loc.txt"
        feats_merged     = loc_dir / "feats-merged.h5"
        loc_matches      = loc_dir / "matches-loc.h5"
        loc_dir.mkdir(parents=True, exist_ok=True)

        # Use 'inc_' prefix to avoid key collision with ref images in merged h5.
        # Both ref and inc use seq/XXXXXX.color.jpg locally; the prefix makes
        # them distinguishable when stored in the same h5 file.
        inc_prefixed = [f"inc_{img}" for img in inc_images]
        orig_to_prefixed = dict(zip(inc_images, inc_prefixed))
        prefixed_to_orig = dict(zip(inc_prefixed, inc_images))

        # Copy inc images to a temp dir with prefixed names so hloc can read them
        tmp_inc_dir = loc_dir / "_tmp_inc"
        tmp_inc_dir.mkdir(parents=True, exist_ok=True)
        for img, prefixed in orig_to_prefixed.items():
            src = inc_dir / img
            dst = tmp_inc_dir / prefixed
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)

        extract_features.main(
            _FEATURE_CONF, tmp_inc_dir, image_list=inc_prefixed,
            feature_path=feats_inc, overwrite=True,
        )
        extract_features.main(
            _RETRIEVAL_CONF, tmp_inc_dir, image_list=inc_prefixed,
            feature_path=global_feats_inc, overwrite=True,
        )

        # Retrieval: query = inc_prefixed names, db = ref names
        k = min(_NUM_RETRIEVAL, len(ref_images))
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
            features=feats_merged, matches=loc_matches, overwrite=True,
        )
        return self._run_pnp(model, inc_prefixed, prefixed_to_orig,
                             loc_pairs, feats_merged, loc_matches, intrinsics)

    def _run_pnp(
        self,
        model: pycolmap.Reconstruction,
        inc_prefixed: List[str],
        prefixed_to_orig: Dict[str, str],
        loc_pairs: Path,
        features: Path,
        matches: Path,
        intrinsics: Tuple[float, float, float, float, int, int],
    ) -> Dict[str, np.ndarray]:
        """Run PnP for each incoming image using prefixed names.

        Args:
            inc_prefixed: list of 'inc_seq/XXXXXX.color.jpg' names
            prefixed_to_orig: mapping from prefixed name to original local name

        Returns:
            {original_inc_img_name: 4x4 C2W} in W0 frame.
        """
        if not loc_pairs.exists() or loc_pairs.stat().st_size == 0:
            logger.warning("No retrieval pairs found for PnP, returning empty results")
            return {}

        fx, fy, cx, cy, w, h = intrinsics
        query_cam = pycolmap.Camera(
            model="PINHOLE", width=int(w), height=int(h),
            params=[fx, fy, cx, cy],
        )
        localizer = QueryLocalizer(model, _LOC_CONF)
        name_to_id = {img.name: img_id for img_id, img in model.images.items()}

        query_to_dbs: Dict[str, List[int]] = {img: [] for img in inc_prefixed}
        with open(loc_pairs) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] in query_to_dbs:
                    if parts[1] in name_to_id:
                        query_to_dbs[parts[0]].append(name_to_id[parts[1]])

        results: Dict[str, np.ndarray] = {}
        n_failed = 0
        for prefixed in inc_prefixed:
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

            if not ret.get("success", False) or ret.get("num_inliers", 0) < 6:
                n_failed += 1
                continue

            w2c = np.vstack([ret["cam_from_world"].matrix(), [0.0, 0.0, 0.0, 1.0]])
            results[orig_name] = np.linalg.inv(w2c)

        logger.info(f"PnP: {len(results)}/{len(inc_prefixed)} localized, {n_failed} failed")
        return results
