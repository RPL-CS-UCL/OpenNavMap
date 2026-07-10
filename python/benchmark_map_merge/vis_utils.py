"""Visualization utilities for map merging results.

Provides:
  save_topdown_pose_viz()  – top-down merged trajectory plot, optionally aligned to GT
  save_sfm_vis()           – SfM reconstruction point cloud + camera poses plot

Pose conventions:
  - C2W maps camera points to world; its translation is the camera center.
  - W2C maps world points to camera; its translation is not the camera center.
  - This dataset stores W2C vec7 [qw,qx,qy,qz,tx,ty,tz]. Camera center is -R^T @ t.
  - TUM stores camera center plus C2W rotation: timestamp tx ty tz qx qy qz qw.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _vec_to_cam_pos(pose_vec: np.ndarray) -> np.ndarray:
    """W2C vec7 [qw,qx,qy,qz,tx,ty,tz] -> camera center in world = -R^T @ t."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([pose_vec[1], pose_vec[2], pose_vec[3], pose_vec[0]]).as_matrix()
    return -R.T @ pose_vec[4:7]


def _select_topdown_axes(points: np.ndarray) -> Tuple[int, int, str]:
    """Select X plus the wider of Y/Z for a top-down trajectory plot."""
    y_spread = float(np.ptp(points[:, 1]))
    z_spread = float(np.ptp(points[:, 2]))
    if y_spread >= z_spread:
        return 0, 1, "Y"
    return 0, 2, "Z"


def _estimate_umeyama_se3(
    est_pts: np.ndarray, gt_pts: np.ndarray
) -> np.ndarray:
    """Estimate SE3 Umeyama (no scale) from est to gt using point sets.

    Returns 4×4 transform T such that gt ≈ T @ est.
    """
    from numpy.linalg import svd
    n = est_pts.shape[0]
    e_mean = est_pts.mean(axis=0)
    g_mean = gt_pts.mean(axis=0)
    est_c = est_pts - e_mean
    gt_c  = gt_pts - g_mean
    C = gt_c.T @ est_c / n
    U, _, Vt = svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    t = g_mean - R @ e_mean
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def save_topdown_pose_viz(
    submap_pose_list: List[Dict[str, np.ndarray]],
    submap_gt_list: Optional[List[Dict[str, np.ndarray]]],
    images_ordered_list: List[List[str]],
    output_path: Path,
    title: str = "",
    align_with_gt: bool = False,
) -> None:
    """Plot a merged top-down trajectory view, one segment per submap.

    Each submap is drawn as a separate connected line segment using the local
    image ordering in images_ordered_list, so submap boundaries are not
    connected and no cross-submap jump lines appear.

    Args:
        submap_pose_list: per-submap list of {local_img_name: W2C vec7}
        submap_gt_list: optional per-submap list of {local_img_name: W2C vec7} GT
        images_ordered_list: per-submap ordered frame name lists
        output_path: where to save the PNG
        title: figure title
        align_with_gt: align estimated poses to GT with SE3 Umeyama before plotting
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # collect all estimated camera centers (for axis selection and optional alignment)
    all_est_pts: List[np.ndarray] = []
    all_gt_pts: List[np.ndarray] = []
    for imgs, poses in zip(images_ordered_list, submap_pose_list):
        for img in imgs:
            if img not in poses:
                continue
            all_est_pts.append(_vec_to_cam_pos(poses[img]))
            if align_with_gt and submap_gt_list is not None:
                for gt_dict in submap_gt_list:
                    if img in gt_dict:
                        all_gt_pts.append(_vec_to_cam_pos(gt_dict[img]))
                        break

    if not all_est_pts:
        print("[save_topdown_pose_viz] no merged poses, skipping")
        return

    all_est = np.array(all_est_pts)
    all_gt = np.array(all_gt_pts) if all_gt_pts else None

    _h0, _h1, h1_label = _select_topdown_axes(all_est)

    T_align = None
    if align_with_gt and all_gt is not None and len(all_gt) == len(all_est):
        T_align = _estimate_umeyama_se3(all_est, all_gt)

    fig, ax = plt.subplots(figsize=(10, 10))

    if align_with_gt and all_gt is not None:
        ax.plot(all_gt[:, _h0], all_gt[:, _h1], "--", color="gray",
                alpha=0.5, linewidth=0.8, label="GT")

    first_pt = None
    for sub_idx, (imgs, poses) in enumerate(zip(images_ordered_list, submap_pose_list)):
        pts = [_vec_to_cam_pos(poses[img]) for img in imgs if img in poses]
        if not pts:
            continue
        seg = np.array(pts)
        if T_align is not None:
            seg = (T_align[:3, :3] @ seg.T + T_align[:3, 3:4]).T
        label = "merged" if sub_idx == 0 else "_nolegend_"
        ax.plot(seg[:, _h0], seg[:, _h1], "-", color="#1f77b4",
                linewidth=0.7, label=label, alpha=0.85)
        if first_pt is None:
            first_pt = seg[0]

    if first_pt is not None:
        ax.scatter(first_pt[_h0], first_pt[_h1], marker="*",
                   color="#1f77b4", s=100, zorder=5)

    plane_name = f"X{h1_label}"
    default_title = f"Top-down trajectory (merged poses, {plane_name} plane)"
    if align_with_gt:
        default_title = f"Top-down trajectory (SE3-aligned to GT, {plane_name} plane)"
    ax.set_xlabel("X (m)")
    ax.set_ylabel(f"{h1_label} (m)")
    ax.set_title(title or default_title)
    ax.legend(loc="best", fontsize="small")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("equal")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save_topdown_pose_viz] saved to {output_path}")


def save_sfm_vis(
    model,
    output_path: Path,
    title: str = "SfM Reconstruction",
    intrinsics: Optional[Tuple[float, float, float, float, int, int]] = None,
) -> None:
    """Save SfM reconstruction to a Rerun .rrd file.

    Logs:
      - sfm/points        : Points3D with COLMAP RGB colors
      - sfm/cameras/<id>  : Transform3D (axis_length=0.1; 0.5 for first camera)
      - sfm/cameras/<id>/image: Pinhole camera frustum when intrinsics are given

    Args:
        model: pycolmap.Reconstruction object
        output_path: destination path; suffix is forced to .rrd
        title: Rerun application ID / recording title
        intrinsics: optional (fx, fy, cx, cy, width, height) for camera frustums
    """
    import os
    _conda_prefix = os.environ.get("CONDA_PREFIX", "/root/miniconda3/envs/opennavmap")
    _conda_lib = os.path.join(_conda_prefix, "lib")
    if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = (
            _conda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )

    import rerun as rr
    from scipy.spatial.transform import Rotation as _Rot

    output_path = Path(output_path).with_suffix(".rrd")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(title, recording_id=None, spawn=False)

    # ── point cloud ───────────────────────────────────────────────────────────
    if model.points3D:
        pts = np.array([p.xyz for p in model.points3D.values()], dtype=np.float32)
        colors = np.array(
            [p.color[:3] for p in model.points3D.values()], dtype=np.uint8
        )
        rr.log("sfm/points", rr.Points3D(pts, colors=colors, radii=0.02))

    # ── camera poses ──────────────────────────────────────────────────────────
    if hasattr(model, "reg_image_ids"):
        reg_ids = set(model.reg_image_ids())
        registered = [img for img_id, img in model.images.items() if int(img_id) in reg_ids]
    else:
        registered = [img for img in model.images.values() if img.registered]
    if not registered:
        print("[save_sfm_vis] warning: no registered cameras")
    for k, img in enumerate(registered):
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R_w2c = cfw.rotation.matrix()
        t_w2c = cfw.translation
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        translation = t_c2w.astype(np.float32)
        quat_xyzw = _Rot.from_matrix(R_c2w).as_quat()  # [x,y,z,w]
        axis_len = 0.5 if k == 0 else 0.1
        camera_group = "inc" if img.name.startswith("inc") else "ref"
        rr.log(
            f"sfm/cameras/{camera_group}/{img.image_id}",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
                axis_length=axis_len,
            ),
        )
        if intrinsics is not None:
            fx, fy, cx, cy, iw, ih = intrinsics
            rr.log(
                f"sfm/cameras/{camera_group}/{img.image_id}/image",
                rr.Pinhole(
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                    width=int(iw),
                    height=int(ih),
                    image_plane_distance=float(axis_len * 0.5),
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )
        if camera_group == "inc":
            rr.log(
                f"sfm/cameras/inc_points/{img.image_id}",
                rr.Points3D(
                    np.asarray([translation], dtype=np.float32),
                    colors=np.asarray([[0, 200, 0]], dtype=np.uint8),
                    radii=0.05,
                ),
            )

    rr.save(str(output_path))
    print(f"[save_sfm_vis] saved to {output_path}")
