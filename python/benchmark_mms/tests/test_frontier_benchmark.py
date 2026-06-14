import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmark_mms.frontier_explore_benchmark as feb


def test_fov_half_deg_is_30():
    assert feb.FOV_HALF_DEG == 30.0, f"Expected 30.0, got {feb.FOV_HALF_DEG}"


def test_fov_range_m_is_5():
    assert feb.FOV_RANGE_M == 5.0, f"Expected 5.0, got {feb.FOV_RANGE_M}"


def test_fov_half_rad_consistent_with_deg():
    assert abs(feb.FOV_HALF_RAD - np.radians(30.0)) < 1e-9
