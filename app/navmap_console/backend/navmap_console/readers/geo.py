"""GPS-aligned trajectory of one step: poses.txt camera centres fitted onto gps_data.txt fixes.

The fit (ENU about the first fix + Kabsch) is litevloc's utils_gps_align, imported lazily so
this package stays importable without the litevloc PYTHONPATH.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pymap3d as pm

from .map_files import read_poses_c2w

MIN_FIXES = 2  # below this the rigid fit is undefined (utils_gps_align returns identity)


def read_gps_rows(path: Path) -> Dict[str, np.ndarray]:
    """frame name -> [lat, lon, alt, speed, accuracy] as float64 (nan kept as nan); {} when missing."""
    rows: Dict[str, np.ndarray] = {}
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows[parts[0]] = np.asarray([float(v) for v in parts[1:6]], dtype=np.float64)
        except ValueError:
            continue
    return rows


def aligned_trajectory(step_dir: Path) -> Dict[str, Any]:
    """Every frame's (lat, lon) after aligning the local trajectory to the GPS fixes.

    rmse_m is the residual between the aligned positions and their fixes over the frames that
    have one (a display hint, not an evaluation metric). reason="no_gps" when fewer than
    MIN_FIXES frames carry a lat/lon.
    """
    from utils.utils_gps_align import collect_gps_pairs, compute_local_to_enu, local_to_geodetic

    poses = read_poses_c2w(step_dir / "poses.txt")
    gps = read_gps_rows(step_dir / "gps_data.txt")
    positions: List[np.ndarray] = [np.asarray(p, dtype=np.float64) for p in poses.pos]
    rows: List[Optional[np.ndarray]] = [gps.get(name) for name in poses.names]
    pairs, origin = collect_gps_pairs(positions, rows)
    empty: Dict[str, Any] = {"origin": None, "n_frames": len(positions), "n_gps": len(pairs),
                             "traj": [], "gps": [], "rmse_m": None, "reason": "no_gps"}
    if origin is None or len(pairs) < MIN_FIXES:
        return empty
    T = compute_local_to_enu(pairs, origin)
    traj = local_to_geodetic(T, positions, origin)
    fitted = local_to_geodetic(T, [p for p, _ in pairs], origin)
    fixes = np.asarray([g for _, g in pairs], dtype=np.float64)
    # residual in metres: compare in the ENU plane about the origin
    d = []
    for f, g in zip(fitted, fixes):
        e1, n1, _ = pm.geodetic2enu(f[0], f[1], f[2], *origin)
        e2, n2, _ = pm.geodetic2enu(g[0], g[1], g[2], *origin)
        d.append((e1 - e2) ** 2 + (n1 - n2) ** 2)
    return {
        "origin": [float(origin[0]), float(origin[1])],
        "n_frames": len(positions),
        "n_gps": len(pairs),
        "traj": [[float(lat), float(lon)] for lat, lon, _ in traj],
        "gps": [[float(g[0]), float(g[1])] for g in fixes],
        "rmse_m": float(np.sqrt(np.mean(d))),
        "reason": None,
    }
