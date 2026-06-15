import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmark_mms.frontier_explore_benchmark as feb


def test_fov_half_deg_is_45():
    assert feb.FOV_HALF_DEG == 45.0, f"Expected 45.0, got {feb.FOV_HALF_DEG}"


def test_fov_range_m_is_8():
    assert feb.FOV_RANGE_M == 8.0, f"Expected 8.0, got {feb.FOV_RANGE_M}"


def test_fov_half_rad_consistent_with_deg():
    assert abs(feb.FOV_HALF_RAD - np.radians(45.0)) < 1e-9


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


# ── Manhattan / graph-structure fixes ────────────────────────────────

def test_astar_distance_is_manhattan():
    """A* on a clear 10×10 grid must return Manhattan distance (4-dir only)."""
    grid = np.zeros((10, 10), dtype=np.uint8)
    _, dist = feb.astar(grid, (0, 0), (3, 4))
    expected = (3 + 4) * feb.GRID_RES_M
    assert abs(dist - expected) < 1e-9, f"Expected {expected}, got {dist}"


def test_astar_no_diagonal_movement():
    """A* must not cut diagonals — every step must be axis-aligned."""
    grid = np.zeros((10, 10), dtype=np.uint8)
    path, _ = feb.astar(grid, (0, 0), (2, 2))
    assert path is not None
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        assert (r0 == r1) or (c0 == c1), f"Diagonal step from ({r0},{c0}) to ({r1},{c1})"


def test_topo_subgraph_edge_weight_is_manhattan():
    """build_topometric_subgraph edge weight: Manhattan, not Euclidean."""
    import networkx as nx
    poses = [(0, 0, 0.0), (3, 4, 0.0)]
    G = feb.build_topometric_subgraph(poses, res=feb.GRID_RES_M)
    assert G.number_of_edges() == 1
    weight = list(G.edges(data=True))[0][2]["weight"]
    expected = (3 + 4) * feb.GRID_RES_M
    assert abs(weight - expected) < 1e-9, f"Expected {expected}, got {weight}"


def test_merge_connects_nodes_through_obstacle():
    """merge_topometric_graphs must NOT connect nodes if A* path is obstructed."""
    import networkx as nx
    base_grid = np.zeros((5, 5), dtype=np.uint8)
    base_grid[2, 2] = 1
    G1 = nx.Graph(); G1.add_node(0, x=0.5, y=0.5, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=1.5, y=1.5, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], base_grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() == 0, "Nodes across obstacle must NOT be connected via A*"


def test_merge_distance_threshold_is_5m():
    """merge_topometric_graphs connects nodes up to 5 m Manhattan distance."""
    import networkx as nx
    base_grid = np.zeros((20, 20), dtype=np.uint8)
    G1 = nx.Graph(); G1.add_node(0, x=0.0, y=0.0, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=4.5, y=0.0, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], base_grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() >= 1, "Nodes 4.5 m apart must connect (threshold=5 m)"


def test_merge_does_not_connect_beyond_5m():
    """merge_topometric_graphs rejects nodes > 5 m Manhattan distance."""
    import networkx as nx
    base_grid = np.zeros((20, 20), dtype=np.uint8)
    G1 = nx.Graph(); G1.add_node(0, x=0.0, y=0.0, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=5.5, y=0.0, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], base_grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() == 0, "Nodes > 5 m apart must NOT be connected"


def test_pcd_dilate_is_0():
    """PCD_DILATE must be 0 (raw grid, inflate for planning)."""
    assert feb.PCD_DILATE == 0, f"Expected 0, got {feb.PCD_DILATE}"


def test_inflate_radius_is_1():
    """INFLATE_RADIUS must be 1 (safety margin for planning)."""
    assert feb.INFLATE_RADIUS == 1, f"Expected 1, got {feb.INFLATE_RADIUS}"


def test_topo_subgraph_edge_weight_uses_astar():
    """Edge weight with base_grid provided must reflect A* path, not straight Manhattan."""
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[0:4, 2] = 1   # vertical wall at col 2, rows 0-3
    poses = [(0, 0, 0.0), (0, 6, 0.0)]
    G_no_grid = feb.build_topometric_subgraph(poses, res=feb.GRID_RES_M)
    G_with_grid = feb.build_topometric_subgraph(poses, res=feb.GRID_RES_M, base_grid=grid)
    w_no = list(G_no_grid.edges(data=True))[0][2]["weight"]
    w_with = list(G_with_grid.edges(data=True))[0][2]["weight"]
    assert w_no == 6 * feb.GRID_RES_M, f"Expected 3.0, got {w_no}"
    assert w_with > w_no, f"Expected A* weight {w_with} > Manhattan {w_no}"


def test_merge_edge_weight_uses_astar():
    """Cross-session edge weight must use A* path, not straight Manhattan."""
    import networkx as nx
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[0:8, 4] = 1   # vertical wall at col 4, rows 0-7 (row 8-9 free for detour)
    G1 = nx.Graph(); G1.add_node(0, x=0.5, y=0.5, yaw=0.0)   # grid(r=1,c=1)
    G2 = nx.Graph(); G2.add_node(0, x=3.5, y=0.5, yaw=0.0)   # grid(r=1,c=7)
    merged = feb.merge_topometric_graphs([G1, G2], grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() >= 1
    edge_w = list(merged.edges(data=True))[0][2]["weight"]
    # Manhattan = 3.0m, but wall forces detour → A* > 3.0m
    assert edge_w > 3.0, f"Expected A* weight > 3.0, got {edge_w}"


def test_merge_no_edge_if_unreachable():
    """Unreachable cross-session node pairs must not be connected."""
    import networkx as nx
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 4] = 1   # full vertical wall
    G1 = nx.Graph(); G1.add_node(0, x=1.5, y=1.5, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=3.5, y=1.5, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() == 0, "Unreachable nodes must not be connected"
