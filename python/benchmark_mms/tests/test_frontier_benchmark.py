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


def test_select_frontier_prefers_goal_direction():
    rng = np.random.default_rng(0)
    frontiers = [(2, 7), (10, 7)]
    free_neighbors = [(3, 7), (9, 7)]
    current = (5, 7)
    goal = (12, 7)
    obs = np.full((15, 15), -1, dtype=np.int8)
    inf_pg = np.zeros((15, 15), dtype=np.uint8)

    toward_count = 0
    for _ in range(200):
        result = feb.select_frontier(
            frontiers, current, obs, rng,
            temperature=0.5, top_n=5, inf_pg=inf_pg,
            frontier_free_neighbors=free_neighbors,
            goal=goal, goal_bias=0.5,
        )
        if result == (9, 7):
            toward_count += 1

    assert toward_count > 140, f"Expected >140/200 toward-goal, got {toward_count}"


def test_select_frontier_no_bias_behaves_as_nearest():
    rng = np.random.default_rng(42)
    frontiers = [(5, 6), (10, 6)]
    free_neighbors = [(5, 7), (10, 7)]
    current = (5, 8)
    obs = np.full((15, 15), -1, dtype=np.int8)
    inf_pg = np.zeros((15, 15), dtype=np.uint8)

    nearest_count = 0
    for _ in range(200):
        result = feb.select_frontier(
            frontiers, current, obs, rng,
            temperature=0.5, top_n=5, inf_pg=inf_pg,
            frontier_free_neighbors=free_neighbors,
            goal=None, goal_bias=0.5,
        )
        if result == (5, 7):
            nearest_count += 1

    assert nearest_count > 150, f"Expected >150/200 nearest, got {nearest_count}"


def test_obs_to_rgb_unknown_is_grey():
    obs = np.zeros((5, 5), dtype=np.int8)
    rgb = feb.obs_to_rgb(obs)
    expected = np.array([107, 114, 128]) / 255.0
    np.testing.assert_allclose(rgb[2, 2], expected, atol=1e-6)


def test_obs_to_rgb_free_is_white():
    obs = np.full((5, 5), -1, dtype=np.int8)
    rgb = feb.obs_to_rgb(obs)
    expected = np.array([243, 244, 246]) / 255.0
    np.testing.assert_allclose(rgb[2, 2], expected, atol=1e-6)


def test_obs_to_rgb_obstacle_is_black():
    obs = np.full((5, 5), 1, dtype=np.int8)
    rgb = feb.obs_to_rgb(obs)
    expected = np.array([31, 41, 55]) / 255.0
    np.testing.assert_allclose(rgb[2, 2], expected, atol=1e-6)


def test_obs_to_rgb_mixed():
    obs = np.zeros((3, 3), dtype=np.int8)
    obs[0, 0] = -1
    obs[1, 1] = 1
    rgb = feb.obs_to_rgb(obs)
    white = np.array([243, 244, 246]) / 255.0
    black = np.array([31, 41, 55]) / 255.0
    grey  = np.array([107, 114, 128]) / 255.0
    np.testing.assert_allclose(rgb[0, 0], white, atol=1e-6)
    np.testing.assert_allclose(rgb[1, 1], black, atol=1e-6)
    np.testing.assert_allclose(rgb[2, 2], grey,  atol=1e-6)
