"""Per-step SE(3)-aligned ATE for two merge runs (baseline vs GNC-TLS).

Reads each merge_*/submap_disc_0/{poses.txt,poses_abs_gt.txt} directly so no
TUM export / eval-framework dataset entry is needed. Alignment matches the
eval config (align_type: se3, align_num_frames: -1).
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def _read_mapfree(path: Path) -> Tuple[List[str], np.ndarray]:
    """mapfree poses.txt -> (frame names, world-frame camera centers)."""
    names, centers = [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        f = line.split()
        qw, qx, qy, qz = (float(v) for v in f[1:5])
        t = np.array([float(v) for v in f[5:8]])
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        names.append(f[0])
        centers.append(-rot.T @ t)  # world-to-camera -> camera center
    return names, np.asarray(centers)


def _read_mapfree_rot(path: Path) -> Dict[str, np.ndarray]:
    """mapfree poses.txt -> {frame name: camera-to-world rotation}."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        f = line.split()
        qw, qx, qy, qz = (float(v) for v in f[1:5])
        out[f[0]] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().T
    return out


def _umeyama_se3(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid transform mapping src onto dst (no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    u, _, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    rot = u @ s @ vt
    return rot, mu_d - rot @ mu_s


def ate(merge_dir: Path) -> Tuple[float, float, int]:
    """(trans RMSE [m], rot RMSE [deg], n) for one merge step directory."""
    names, est_c = _read_mapfree(merge_dir / "poses.txt")
    gt_names, gt_c = _read_mapfree(merge_dir / "poses_abs_gt.txt")
    assert names == gt_names, merge_dir
    rot, trans = _umeyama_se3(est_c, gt_c)

    trans_err = np.linalg.norm((est_c @ rot.T + trans) - gt_c, axis=1)
    est_r = _read_mapfree_rot(merge_dir / "poses.txt")
    gt_r = _read_mapfree_rot(merge_dir / "poses_abs_gt.txt")
    rot_err = np.array([
        np.rad2deg(np.linalg.norm(
            Rotation.from_matrix((rot @ est_r[n]).T @ gt_r[n]).as_rotvec()))
        for n in names
    ])
    return (float(np.sqrt((trans_err ** 2).mean())),
            float(np.sqrt((rot_err ** 2).mean())), len(names))


def _steps(root: Path) -> Dict[int, Path]:
    out = {}
    for d in root.glob("merge_*/submap_disc_0"):
        tail = d.parent.name.split("_")[-1]
        if tail.isdigit():
            out[int(tail)] = d
    return out


def main() -> None:
    base, gnc = Path(sys.argv[1]), Path(sys.argv[2])
    b_steps, g_steps = _steps(base), _steps(gnc)
    print(f"{'step':>4} {'n':>5} | {'base_t':>8} {'gnc_t':>8} | "
          f"{'base_r':>8} {'gnc_r':>8}")
    for step in sorted(set(b_steps) & set(g_steps)):
        bt, br, n = ate(b_steps[step])
        gt_t, gt_r, n2 = ate(g_steps[step])
        flag = "" if n == n2 else f"  (n mismatch {n} vs {n2})"
        print(f"{step:>4} {n:>5} | {bt:>8.3f} {gt_t:>8.3f} | "
              f"{br:>8.3f} {gt_r:>8.3f}{flag}")


if __name__ == "__main__":
    main()
