"""Tests for dynamic grid shape support in multisession_sim_osm."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmark_mms.multisession_sim_osm as sim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_square_free_grid(h: int, w: int) -> np.ndarray:
    """Return an all-free grid of given shape (no obstacles)."""
    return np.zeros((h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Task 1 baseline: module imports without error
# ---------------------------------------------------------------------------

def test_module_imports():
    assert hasattr(sim, "rasterize_buildings")
    assert hasattr(sim, "find_zones")
    assert hasattr(sim, "make_session_route")
    assert hasattr(sim, "fig0_base_map")


# ---------------------------------------------------------------------------
# Task 2: rasterize_buildings dynamic params
# ---------------------------------------------------------------------------

def test_rasterize_buildings_signature_accepts_res_m():
    """rasterize_buildings must accept res_m, grid_w, grid_h kwargs."""
    import inspect
    sig = inspect.signature(sim.rasterize_buildings)
    assert "res_m" in sig.parameters, "res_m parameter missing"
    assert "grid_w" in sig.parameters, "grid_w parameter missing"
    assert "grid_h" in sig.parameters, "grid_h parameter missing"


def test_rasterize_buildings_returns_correct_shape():
    """Returned grid shape must equal (grid_h, grid_w)."""
    import unittest.mock as mock

    fake_utm_geom = mock.MagicMock()
    fake_utm_geom.unary_union.centroid.x = 0.0
    fake_utm_geom.unary_union.centroid.y = 0.0
    fake_gdf = mock.MagicMock()
    fake_gdf.to_crs.return_value = fake_utm_geom
    type(fake_utm_geom).__len__ = mock.MagicMock(return_value=0)
    fake_utm_geom.__iter__ = mock.MagicMock(return_value=iter([]))

    grid, bbox = sim.rasterize_buildings(fake_gdf, res_m=1.0, grid_w=200, grid_h=150)
    assert grid.shape == (150, 200), f"Expected (150, 200), got {grid.shape}"
    assert grid.dtype == np.uint8


# ---------------------------------------------------------------------------
# Task 3: fig0_base_map with non-square grid
# ---------------------------------------------------------------------------

def test_fig0_base_map_does_not_crash_with_non_square_grid(tmp_path, monkeypatch):
    """fig0_base_map must work with any grid shape, not just 1000x1000."""
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.setattr(sim, "OUTPUT_DIR", tmp_path)

    grid = _make_square_free_grid(150, 200)
    sim.fig0_base_map(grid, location_name="Test", obs_ratio=0.3)
    assert (tmp_path / "fig0_osm_base_map.png").exists()
