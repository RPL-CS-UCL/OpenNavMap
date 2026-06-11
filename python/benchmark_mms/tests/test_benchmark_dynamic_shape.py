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
