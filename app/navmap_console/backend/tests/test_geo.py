"""GPS-aligned trajectory: poses.txt + gps_data.txt -> lat/lon per frame via litevloc's utils_gps_align."""
from pathlib import Path

import numpy as np
import pymap3d as pm
from scipy.spatial.transform import Rotation

from navmap_console.readers.geo import aligned_trajectory, read_gps_rows
from test_evaluations_service import _make_imported_run, _wait_jobs_done

ORIGIN = (51.5368, -0.0096, 30.0)  # UCL-ish


def _write_step(step: Path, centres: np.ndarray, gps_lines) -> None:
    """Identity-rotation world-to-camera poses: t = -c."""
    step.mkdir(parents=True, exist_ok=True)
    lines = [f"seq/{i:06d}.color.jpg 1 0 0 0 {-c[0]} {-c[1]} {-c[2]}" for i, c in enumerate(centres)]
    (step / "poses.txt").write_text("\n".join(lines) + "\n")
    (step / "gps_data.txt").write_text("\n".join(gps_lines) + "\n")


def _l_shape(n: int = 12) -> np.ndarray:
    a = np.stack([np.linspace(0, 20, n), np.zeros(n), np.zeros(n)], axis=1)
    b = np.stack([np.full(n, 20.0), np.linspace(0, 15, n), np.zeros(n)], axis=1)
    return np.concatenate([a, b[1:]])


def test_aligned_trajectory_recovers_known_transform(tmp_path: Path) -> None:
    centres = _l_shape()
    R = Rotation.from_euler("z", 35, degrees=True).as_matrix()
    t = np.array([8.0, -5.0, 0.0])
    enu = centres @ R.T + t
    gps_lines = []
    for i, (e, n, u) in enumerate(enu):
        lat, lon, alt = pm.enu2geodetic(e, n, u, *ORIGIN)
        # every other frame: no fix (nan lat/lon), like the network-located rows of real data
        gps_lines.append(f"seq/{i:06d}.color.jpg {lat:.8f} {lon:.8f} nan nan 5.0" if i % 2 == 0
                         else f"seq/{i:06d}.color.jpg nan nan nan nan nan")
    _write_step(tmp_path / "merge_0", centres, gps_lines)
    geo = aligned_trajectory(tmp_path / "merge_0")
    assert geo["reason"] is None
    assert geo["n_frames"] == len(centres) and geo["n_gps"] == (len(centres) + 1) // 2
    assert geo["rmse_m"] < 0.01
    assert len(geo["traj"]) == len(centres) and len(geo["gps"]) == geo["n_gps"]
    for i in range(0, len(centres), 2):
        lat, lon, _ = pm.enu2geodetic(*enu[i], *ORIGIN)
        assert abs(geo["traj"][i][0] - lat) < 1e-6 and abs(geo["traj"][i][1] - lon) < 1e-6
    assert abs(geo["origin"][0] - ORIGIN[0]) < 1e-3


def test_aligned_trajectory_without_fixes(tmp_path: Path) -> None:
    centres = _l_shape(4)
    _write_step(tmp_path / "merge_0", centres, [f"seq/{i:06d}.color.jpg nan nan nan nan nan" for i in range(len(centres))])
    geo = aligned_trajectory(tmp_path / "merge_0")
    assert geo["reason"] == "no_gps" and geo["traj"] == [] and geo["gps"] == []
    # a single fix is not enough for the rigid fit either
    (tmp_path / "merge_0" / "gps_data.txt").write_text("seq/000000.color.jpg 51.5 -0.01 nan nan 5\n")
    assert aligned_trajectory(tmp_path / "merge_0")["reason"] == "no_gps"
    (tmp_path / "merge_0" / "gps_data.txt").unlink()
    assert aligned_trajectory(tmp_path / "merge_0")["reason"] == "no_gps"


def test_read_gps_rows_keeps_nan(tmp_path: Path) -> None:
    p = tmp_path / "gps_data.txt"
    p.write_text("seq/000000.color.jpg 51.5 -0.01 nan nan 447\nbad line\n")
    rows = read_gps_rows(p)
    assert list(rows) == ["seq/000000.color.jpg"]
    assert rows["seq/000000.color.jpg"][0] == 51.5 and np.isnan(rows["seq/000000.color.jpg"][2])


def test_geo_endpoint(client, tmp_path: Path) -> None:
    rid, run_id, result = _make_imported_run(client, tmp_path)
    _wait_jobs_done(client, run_id)
    res = client.get(f"/api/regions/{rid}/runs/{run_id}/steps/0/geo.json")
    assert res.status_code == 200
    assert res.json()["reason"] == "no_gps"  # the fixture has no gps_data.txt
    assert client.get(f"/api/regions/{rid}/runs/{run_id}/steps/9/geo.json").status_code == 404
