"""Multi-session map merging via HLoc feature matching WITHOUT COLMAP SfM.

Matches images between submap pairs using SuperPoint/DISK + LightGlue,
estimates the essential matrix for each matched pair, decomposes it to
get relative rotation, and aggregates submap-level transforms via
Umeyama alignment of VIO camera centers.

This avoids COLMAP SfM reconstruction entirely (which crashes on this system).
"""

import sys
import copy
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ESTIMATOR_DIR = str(Path(__file__).resolve().parents[2]
                     / "third_party" / "pose_estimation_models" / "estimator")
if _ESTIMATOR_DIR not in sys.path:
    sys.path.insert(0, _ESTIMATOR_DIR)

_HLOC_DIR = str(Path(_ESTIMATOR_DIR) / "third_party" / "Hierarchical-Localization")
if _HLOC_DIR not in sys.path:
    sys.path.insert(0, _HLOC_DIR)

_LIGHTGLUE_DIR = str(Path(__file__).resolve().parents[2]
                     / "third_party" / "vismatch" / "vismatch" / "third_party" / "LightGlue")
if _LIGHTGLUE_DIR not in sys.path:
    sys.path.insert(0, _LIGHTGLUE_DIR)

import pycolmap
from hloc import extract_features, match_features
from hloc.utils.io import get_keypoints, get_matches

pycolmap.logging.minloglevel = 100
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _build_disk_conf() -> dict:
    conf = copy.deepcopy(extract_features.confs["disk"])
    conf["model"]["max_keypoints"] = 2048
    return conf


FEATURE_CONFS = {
    "hloc_superpoint_splg": "superpoint_max",
    "hloc_disk_dilg": _build_disk_conf(),
}
MATCHER_CONFS = {
    "hloc_superpoint_splg": "superpoint+lightglue",
    "hloc_disk_dilg": "disk+lightglue",
}


def _get_feature_match_confs(method: str) -> Tuple[dict, dict]:
    fval = FEATURE_CONFS[method]
    mname = MATCHER_CONFS[method]
    feature_conf = fval if isinstance(fval, dict) else extract_features.confs[fval]
    return feature_conf, match_features.confs[mname]


def _get_image_list(submap_dir: Path) -> List[str]:
    seq_dir = submap_dir / "seq"
    return sorted([f"seq/{p.name}" for p in seq_dir.glob("*.color.jpg")])

class HlocMapMerger:
    def __init__(self, method: str, out_dir: Path):
        self.method = method
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.feature_conf, self.matcher_conf = _get_feature_match_confs(method)
        self._features_path = self.out_dir / "features.h5"

    def _extract_all_features(self, image_dir: Path, image_list: List[str]):
        if self._features_path.exists():
            self._features_path.unlink()
        extract_features.main(
            self.feature_conf, image_dir,
            image_list=image_list, feature_path=self._features_path, overwrite=True,
        )

    def match_pairs(self, ref_dir: Path, ref_images: List[str],
                    inc_dir: Path, inc_images: List[str]) -> List[Tuple[str, str]]:
        all_images = list(ref_images)
        inc_mapped = []
        for img in inc_images:
            mapped_name = f"inc_{img.replace('/', '_')}"
            inc_mapped.append(mapped_name)
            all_images.append(mapped_name)

        import shutil
        tmp_dir = self.out_dir / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "seq").mkdir(exist_ok=True)

        # Copy all images to tmp_dir for unified processing
        # Use symlinks for efficiency
        for img in ref_images:
            src = ref_dir / img
            dst = tmp_dir / img
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
        for img, mapped in zip(inc_images, inc_mapped):
            src = inc_dir / img
            dst = tmp_dir / mapped
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)

        self._extract_all_features(tmp_dir, all_images)

        pairs_path = self.out_dir / "pairs-cross.txt"
        with open(pairs_path, "w") as f:
            for inc_m in inc_mapped:
                for ref_m in ref_images:
                    f.write(f"{inc_m} {ref_m}\n")
        logger.info(f"Generated {len(inc_mapped) * len(ref_images)} cross-submap pairs")

        matches_path = self.out_dir / "matches-cross.h5"
        match_features.main(
            self.matcher_conf, pairs_path,
            features=self._features_path, matches=matches_path, overwrite=True,
        )

        valid_pairs = []
        for inc_m in inc_mapped:
            for ref_m in ref_images:
                try:
                    mkpts = get_matches(matches_path, inc_m, ref_m)
                    if mkpts is not None and len(mkpts[0]) >= 10:
                        orig_inc = inc_images[inc_mapped.index(inc_m)]
                        valid_pairs.append((ref_m, orig_inc))
                except Exception:
                    continue

        logger.info(f"Valid matched pairs: {len(valid_pairs)}")
        return valid_pairs

    def estimate_submap_transform(
        self,
        valid_pairs: List[Tuple[str, str]],
        ref_poses: Dict[str, np.ndarray],
        inc_poses: Dict[str, np.ndarray],
        intrinsics: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Estimate 4x4 submap transform T_ref_inc using matched pairs."""
        from scipy.spatial.transform import Rotation as R
        import cv2

        fx, fy, cx, cy = intrinsics
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        if len(valid_pairs) < 5:
            logger.warning(f"Only {len(valid_pairs)} valid pairs, need >=5")
            return None

        R_submap_votes = []
        t_submap_votes = []

        for ref_img, inc_img in valid_pairs[:min(len(valid_pairs), 200)]:
            if ref_img not in ref_poses or inc_img not in inc_poses:
                continue
            try:
                inc_mapped = f"inc_{inc_img.replace('/', '_')}"
                mkpts = get_matches(
                    self.out_dir / "matches-cross.h5",
                    inc_mapped, ref_img,
                )
                if mkpts is None or len(mkpts[0]) < 10:
                    continue
                match_indices = mkpts[0]

                kp1 = get_keypoints(self._features_path, inc_mapped)
                kp2 = get_keypoints(self._features_path, ref_img)

                pts1 = kp1[match_indices[:, 0]][:, :2]
                pts2 = kp2[match_indices[:, 1]][:, :2]
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0,
                )
                if E is None or mask.sum() < 10:
                    continue
                _, R_rel, t_rel, _ = cv2.recoverPose(E, pts1, pts2, K)
                R_rel = R_rel.astype(np.float64)
            except Exception:
                continue

            ref_vec = ref_poses[ref_img]
            inc_vec = inc_poses[inc_img]
            R_ref = R.from_quat([ref_vec[1], ref_vec[2], ref_vec[3], ref_vec[0]]).as_matrix()
            R_inc = R.from_quat([inc_vec[1], inc_vec[2], inc_vec[3], inc_vec[0]]).as_matrix()
            p_ref = ref_vec[4:7]
            p_inc = inc_vec[4:7]

            R_vote = R_ref @ R_rel.T @ R_inc.T
            R_submap_votes.append(R_vote)

            t_vote = p_ref - R_vote @ p_inc
            t_submap_votes.append(t_vote)

        if len(R_submap_votes) < 3:
            return None

        # Average rotations via SVD on mean
        R_mean = np.mean(R_submap_votes, axis=0)
        U, _, Vt = np.linalg.svd(R_mean)
        R_submap = U @ Vt
        if np.linalg.det(R_submap) < 0:
            Vt[-1, :] *= -1
            R_submap = U @ Vt

        t_submap = np.median(t_submap_votes, axis=0)

        scale_candidates = []
        for ref_img, inc_img in valid_pairs[:min(len(valid_pairs), 100)]:
            if ref_img not in ref_poses or inc_img not in inc_poses:
                continue
            p_ref = ref_poses[ref_img][4:7]
            p_inc = inc_poses[inc_img][4:7]
            ref_inc_vec = p_ref - (R_submap @ p_inc + t_submap)
            scale_candidates.append(np.linalg.norm(ref_inc_vec) / max(1e-6, np.linalg.norm(p_inc)))

        scale = np.median(scale_candidates) if scale_candidates else 1.0

        T = np.eye(4)
        T[:3, :3] = scale * R_submap
        T[:3, 3] = t_submap

        logger.info(f"Estimated submap transform: scale={scale:.3f}, "
                    f"det(R)={np.linalg.det(R_submap):.3f}, "
                    f"from {len(R_submap_votes)} pairs")
        return T
