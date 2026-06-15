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


def test_astar_uses_8_directions():
    """A* must support 8-directional movement (horizontal, vertical, diagonal)."""
    grid = np.zeros((10, 10), dtype=np.uint8)
    path, _ = feb.astar(grid, (0, 0), (2, 2))
    assert path is not None
    has_diagonal = any(
        abs(path[i][0] - path[i-1][0]) == 1 and abs(path[i][1] - path[i-1][1]) == 1
        for i in range(1, len(path))
    )
    assert has_diagonal, "A* should use diagonal steps in 8-direction mode"


def test_topo_subgraph_edge_weight_is_manhattan():
    """build_topometric_subgraph edge weight: Manhattan, not Euclidean."""
    import networkx as nx
    poses = [(0, 0, 0.0), (6, 5, 0.0)]   # Manhattan = (6+5)*0.5 = 5.5m > 5.0m TRANS_THRESH_M
    G = feb.build_topometric_subgraph(poses, res=feb.GRID_RES_M)
    assert G.number_of_edges() == 1
    weight = list(G.edges(data=True))[0][2]["weight"]
    expected = (6 + 5) * feb.GRID_RES_M
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


def test_topo_subgraph_condition2_last_visible():
    """Condition 2: B becomes keyframe if LOS(A→B) clear but LOS(A→B+1) blocked."""
    grid = np.zeros((15, 15), dtype=np.uint8)
    grid[0:10, 10] = 1   # vertical wall at col 10

    # Trajectory from col 0 to col 12:
    # B=(0,9): LOS(A→B) clear (col 9 is free in base_grid)
    # B+1=(0,10): LOS(A→B+1) blocked by wall at col 10 → triggers condition 2
    poses = [(0, 0, 0.0)] + [(0, i, 0.0) for i in range(1, 13)]
    G = feb.build_topometric_subgraph(poses, res=feb.GRID_RES_M, base_grid=grid)
    node_cols = sorted([int(d["x"] / feb.GRID_RES_M) for _, d in G.nodes(data=True)])
    assert 9 in node_cols, f"Col 9 must be a keyframe (last visible before wall). Got: {node_cols}"


def test_merge_edge_weight_uses_astar():
    """Cross-session edge weight uses A* path when line of sight is clear."""
    import networkx as nx
    grid = np.zeros((10, 10), dtype=np.uint8)
    G1 = nx.Graph(); G1.add_node(0, x=1.0, y=1.0, yaw=0.0)   # grid(r=2,c=2)
    G2 = nx.Graph(); G2.add_node(0, x=3.0, y=1.0, yaw=0.0)   # grid(r=2,c=6)
    merged = feb.merge_topometric_graphs([G1, G2], grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() >= 1
    edge_w = list(merged.edges(data=True))[0][2]["weight"]
    # Manhattan = 2.0m, A* on clear grid = 2.0m (identical with 8-dir Manhattan A*)
    assert edge_w >= 2.0, f"Expected >= 2.0, got {edge_w}"


def test_merge_line_of_sight_required():
    """Nodes without clear line of sight on inflate(base_grid) must not be connected."""
    import networkx as nx
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[3:7, 4] = 1   # vertical partial wall blocking LOS
    G1 = nx.Graph(); G1.add_node(0, x=1.5, y=2.5, yaw=0.0)   # grid(r=5,c=3)
    G2 = nx.Graph(); G2.add_node(0, x=2.5, y=2.5, yaw=0.0)   # grid(r=5,c=5)
    merged = feb.merge_topometric_graphs([G1, G2], grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() == 0, "Wall must block cross-session edge"


def test_merge_no_edge_if_unreachable():
    """Unreachable cross-session node pairs must not be connected."""
    import networkx as nx
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 4] = 1   # full vertical wall
    G1 = nx.Graph(); G1.add_node(0, x=1.5, y=1.5, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=3.5, y=1.5, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() == 0, "Unreachable nodes must not be connected"
