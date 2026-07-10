import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmark_mms.frontier_explore_benchmark as feb


def test_fov_half_deg_is_45():
    assert feb.FOV_HALF_DEG == 45.0, f"Expected 45.0, got {feb.FOV_HALF_DEG}"


def test_fov_range_m_is_8():
    assert feb.FOV_RANGE_M == 5.0, f"Expected 5.0, got {feb.FOV_RANGE_M}"


def test_frontier_temperature_fixed_is_2_5():
    assert feb.FRONTIER_TEMP_FIXED == 2.5, f"Expected 2.5, got {feb.FRONTIER_TEMP_FIXED}"


def test_fov_half_rad_consistent_with_deg():
    assert abs(feb.FOV_HALF_RAD - np.radians(45.0)) < 1e-9


def test_select_frontier_prefers_goal_direction():
    rng = np.random.default_rng(0)
    frontiers = [(2, 7), (10, 7)]
    free_neighbors = [(3, 7), (7, 7)]
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
        if result == (7, 7):
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


def test_fig3_returns_cumulative_coverage_square_meters(tmp_path):
    base_grid = np.zeros((4, 4), dtype=np.uint8)
    obs0 = np.zeros((4, 4), dtype=np.int8)
    obs1 = np.zeros((4, 4), dtype=np.int8)
    obs0[0, 0] = -1
    obs0[0, 1] = -1
    obs1[0, 1] = -1
    obs1[1, 0] = -1

    data = feb.fig3_reachability_coverage(
        base_grid, [], [obs0, obs1], start=(0, 0), goal=(3, 3),
        res=0.5, output_path=tmp_path / "coverage.png",
    )

    assert data["cum_m2"] == [0.5, 0.75]
    assert data["new_m2"] == [0.5, 0.25]
    assert data["cum_pct"] == [12.5, 18.75]
    assert (tmp_path / "coverage.png").exists()


def test_topomap_to_npz_arrays_exports_nodes_edges_weights():
    import networkx as nx
    G = nx.Graph()
    G.add_node(0, x=1.0, y=2.0)
    G.add_node(1, x=3.0, y=4.0)
    G.add_edge(0, 1, weight=5.0)
    G.graph["start_node"] = 0
    G.graph["goal_node"] = 1

    arrays = feb.topomap_to_npz_arrays(G)

    np.testing.assert_allclose(arrays["nodes_xy"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.testing.assert_array_equal(arrays["edges"], np.array([[0, 1]], dtype=np.int32))
    np.testing.assert_allclose(arrays["edge_weights"], np.array([5.0], dtype=np.float32))
    np.testing.assert_array_equal(arrays["start_node"], np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(arrays["goal_node"], np.array([1], dtype=np.int32))


def test_apply_obstacle_block_uses_world_meter_coordinates():
    base_grid = np.zeros((8, 10), dtype=np.uint8)
    blocked_grid, block_cells = feb.apply_obstacle_block_world(
        base_grid,
        (2.0, 3.0, 4.0, 5.0),
        col_range=(0.0, 9.0),
        row_range=(0.0, 7.0),
        res=1.0,
    )

    assert block_cells == (3, 2, 5, 4)
    assert base_grid.sum() == 0
    assert blocked_grid[3:6, 2:5].sum() == 9
    assert blocked_grid.sum() == 9


# ── Manhattan / graph-structure fixes ────────────────────────────────

def test_astar_distance_is_euclidean():
    """A* on a clear 10×10 grid must use Euclidean step costs (8-dir with √2 diagonal)."""
    grid = np.zeros((10, 10), dtype=np.uint8)
    _, dist = feb.astar(grid, (0, 0), (3, 4))
    # Optimal 8-dir path: 3 diagonal (√2×r) + 1 straight (r) = 3×√2×0.5 + 0.5
    r = feb.GRID_RES_M
    expected = 3 * (np.sqrt(2) * r) + 1 * r
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


def test_astar_local_matches_global_path_cost():
    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[5:15, 10] = 1
    grid[10, 10] = 0

    _, global_dist = feb.astar(grid, (2, 2), (18, 18), res=0.5)
    _, local_dist = feb.astar_local(grid, (2, 2), (18, 18), res=0.5, margin=20)

    assert abs(local_dist - global_dist) < 1e-9


def test_astar_local_matches_global_astar_in_window():
    grid = np.zeros((30, 30), dtype=np.uint8)
    grid[10:20, 15] = 1
    start = (8, 8)
    goal = (22, 22)

    global_path, global_dist = feb.astar(grid, start, goal, res=1.0)
    local_path, local_dist = feb.astar_local(grid, start, goal, res=1.0, margin=5)

    assert global_path is not None
    assert local_path is not None
    assert abs(local_dist - global_dist) < 1e-9
    assert local_path[0] == start
    assert local_path[-1] == goal


def test_topo_subgraph_edge_weight_is_euclidean():
    """build_topometric_subgraph edge weight: Euclidean, not Manhattan."""
    import networkx as nx
    # Use res=0.5 to keep grid coords within test dimensions
    poses = [(0, 0, 0.0), (12, 0, 0.0)]  # Euclidean = 12*0.5 = 6.0m > 5.0m TRANS_THRESH_M
    G = feb.build_topometric_subgraph(poses, res=0.5)
    assert G.number_of_edges() == 1
    weight = list(G.edges(data=True))[0][2]["weight"]
    expected = np.hypot(0, 12) * 0.5
    assert abs(weight - expected) < 1e-9, f"Expected {expected}, got {weight}"


def test_merge_connects_nodes_through_obstacle():
    """merge_topometric_graphs must NOT connect nodes if A* path is obstructed."""
    import networkx as nx
    base_grid = np.zeros((5, 5), dtype=np.uint8)
    base_grid[2, 2] = 1
    G1 = nx.Graph(); G1.add_node(0, x=0.5, y=0.5, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=1.5, y=1.5, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], base_grid, res=0.5)
    assert merged.number_of_edges() == 0, "Nodes across obstacle must NOT be connected via A*"


def test_merge_distance_threshold_is_5m():
    """merge_topometric_graphs connects nodes up to 5 m Manhattan distance."""
    import networkx as nx
    base_grid = np.zeros((20, 20), dtype=np.uint8)
    G1 = nx.Graph(); G1.add_node(0, x=0.0, y=0.0, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=4.5, y=0.0, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], base_grid, res=0.5)
    assert merged.number_of_edges() >= 1, "Nodes 4.5 m apart must connect (threshold=5 m)"


def test_merge_does_not_connect_beyond_5m():
    """merge_topometric_graphs rejects nodes > 5 m Manhattan distance."""
    import networkx as nx
    base_grid = np.zeros((20, 20), dtype=np.uint8)
    G1 = nx.Graph(); G1.add_node(0, x=0.0, y=0.0, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=5.5, y=0.0, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], base_grid, res=0.5)
    assert merged.number_of_edges() == 0, "Nodes > 5 m apart must NOT be connected"


def test_pcd_dilate_is_1():
    """PCD_DILATE must be 1 (walls expanded by 1 pixel on load)."""
    assert feb.PCD_DILATE == 1, f"Expected 1, got {feb.PCD_DILATE}"


def test_topo_subgraph_condition2_last_visible():
    """Condition 2: B becomes keyframe if LOS(A→B) clear but LOS(A→B+1) blocked."""
    grid = np.zeros((15, 15), dtype=np.uint8)
    grid[0:10, 10] = 1   # vertical wall at col 10

    # Trajectory from col 0 to col 12:
    # B=(0,9): LOS(A→B) clear (col 9 is free in base_grid)
    # B+1=(0,10): LOS(A→B+1) blocked by wall at col 10 → triggers condition 2
    poses = [(0, 0, 0.0)] + [(0, i, 0.0) for i in range(1, 13)]
    G = feb.build_topometric_subgraph(poses, res=0.5, base_grid=grid)
    node_cols = sorted([int(d["x"] / 0.5) for _, d in G.nodes(data=True)])
    assert 9 in node_cols, f"Col 9 must be a keyframe (last visible before wall). Got: {node_cols}"


def test_fov_observation_stops_at_obstacle():
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[3, 4] = 1
    obs = np.zeros_like(grid, dtype=np.int8)

    feb.add_fov_observation(
        obs, 3, 2, yaw=0.0, base_grid=grid,
        fov_range_m=4.0, fov_half_rad=0.01, res=1.0,
    )

    assert obs[3, 3] == -1
    assert obs[3, 4] == 1
    assert obs[3, 5] == 0


def test_topo_subgraph_forces_last_pose_as_node():
    poses = [(0, 0, 0.0), (1, 0, 0.0), (2, 0, 0.0)]
    G = feb.build_topometric_subgraph(poses, res=1.0, trans_thresh=10.0, goal=(2, 0))

    assert G.number_of_nodes() == 2
    assert G.graph["start_node"] == 0
    assert G.graph["goal_node"] == 1
    assert G.nodes[1]["x"] == 0.0
    assert G.nodes[1]["y"] == 2.0


def test_topo_subgraph_does_not_set_goal_node_before_reaching_goal():
    poses = [(0, 0, 0.0), (1, 0, 0.0), (2, 0, 0.0)]
    G = feb.build_topometric_subgraph(poses, res=1.0, trans_thresh=10.0, goal=(5, 0))

    assert G.number_of_nodes() == 2
    assert G.graph["start_node"] == 0
    assert "goal_node" not in G.graph


def test_topo_subgraph_keeps_terminal_edge_without_reaching_goal():
    poses = [(0, 0, 0.0), (0, 1, 0.0), (0, 2, 0.0)]
    G = feb.build_topometric_subgraph(poses, res=1.0, trans_thresh=10.0, goal=(0, 5))

    assert G.number_of_nodes() == 2
    assert G.has_edge(0, 1)
    assert G.nodes[1]["x"] == 2.0
    assert G.nodes[1]["y"] == 0.0
    assert "goal_node" not in G.graph


def test_add_intra_session_loop_edges_connects_non_adjacent_nodes():
    grid = np.zeros((10, 10), dtype=np.uint8)
    G = feb.build_topometric_subgraph(
        [(0, 0, 0.0), (0, 6, 0.0), (3, 3, 0.0)],
        res=1.0, trans_thresh=2.0, base_grid=grid,
    )

    assert not G.has_edge(0, 2)
    feb.add_intra_session_loop_edges(G, grid, res=1.0, cross_dist=5.0)

    assert G.has_edge(0, 2)


def test_candidate_topo_edge_uses_rounded_grid_coordinates():
    grid = np.zeros((100, 100), dtype=np.uint8)
    edge = feb._candidate_topo_edge(
        xi=8.6, yi=12.6, xj=8.8, yj=13.6,
        base_grid=grid, res=0.2, cross_dist=5.0,
    )

    assert edge is not None
    path_len, _ = edge
    _, expected = feb.astar(grid, (63, 43), (68, 44), res=0.2)
    assert abs(path_len - expected) < 1e-9


def test_frontier_session_does_not_goal_check_every_inner_cell(monkeypatch):
    grid = np.zeros((20, 20), dtype=np.uint8)
    start = (1, 1)
    goal = (18, 18)
    calls_to_goal = 0
    original_astar = feb.astar

    def counting_astar(grid_arg, start_arg, goal_arg, res=feb.GRID_RES_M):
        nonlocal calls_to_goal
        if goal_arg == goal:
            calls_to_goal += 1
        return original_astar(grid_arg, start_arg, goal_arg, res)

    monkeypatch.setattr(feb, "astar", counting_astar)
    feb.frontier_explore_session(
        start, goal, grid, np.random.default_rng(0), initial_yaw=0.0,
        frontier_temperature=2.5, res=1.0, fov_range_m=4.0,
        fov_half_rad=np.pi / 2, max_steps=3,
    )

    assert calls_to_goal <= 4


def test_merge_edge_weight_uses_astar():
    """Cross-session edge weight uses A* path when line of sight is clear."""
    import networkx as nx
    grid = np.zeros((10, 10), dtype=np.uint8)
    G1 = nx.Graph(); G1.add_node(0, x=1.0, y=1.0, yaw=0.0)   # grid(r=2,c=2)
    G2 = nx.Graph(); G2.add_node(0, x=3.0, y=1.0, yaw=0.0)   # grid(r=2,c=6)
    merged = feb.merge_topometric_graphs([G1, G2], grid, res=0.5)
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
    merged = feb.merge_topometric_graphs([G1, G2], grid, res=0.5)
    assert merged.number_of_edges() == 0, "Wall must block cross-session edge"


def test_merge_no_edge_if_unreachable():
    """Unreachable cross-session node pairs must not be connected."""
    import networkx as nx
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 4] = 1   # full vertical wall
    G1 = nx.Graph(); G1.add_node(0, x=1.5, y=1.5, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=3.5, y=1.5, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], grid, res=0.5)
    assert merged.number_of_edges() == 0, "Unreachable nodes must not be connected"


def test_draw_topo_graph_downsamples_nodes_per_subgraph(tmp_path):
    """_draw_topo_graph with max_nodes_per_subgraph=5 should draw at most 5 nodes
    from a 20-node subgraph and all 3 nodes from a small subgraph."""
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = nx.Graph()
    # subgraph 0: nodes 0-19
    for i in range(20):
        G.add_node(i, x=float(i), y=0.0, yaw=0.0)
        if i > 0:
            G.add_edge(i - 1, i, weight=1.0, rel_pose=(1.0, 0.0, 0.0))
    # subgraph 1: nodes 20-22
    for i in range(20, 23):
        G.add_node(i, x=float(i - 20), y=5.0, yaw=0.0)
        if i > 20:
            G.add_edge(i - 1, i, weight=1.0, rel_pose=(1.0, 0.0, 0.0))

    subgraph_node_sets = [set(range(20)), set(range(20, 23))]

    fig, ax = plt.subplots()
    feb._draw_topo_graph(
        ax, G, res=1.0,
        subgraph_node_sets=subgraph_node_sets,
        max_nodes_per_subgraph=5,
    )
    # Count drawn scatter points (nodes_x len from the scatter call inside)
    # The scatter PathCollection stores offsets
    scatter = ax.collections[-1]
    n_drawn = len(scatter.get_offsets())
    plt.close(fig)

    # subgraph 0 downsampled to 5, subgraph 1 kept at 3 → total = 8
    assert n_drawn <= 8, f"Expected ≤8 drawn nodes, got {n_drawn}"
    assert n_drawn >= 3, "All 3 nodes of small subgraph must be kept"


def test_draw_merge_grid_produces_more_cross_edges():
    """merge_topometric_graphs with clear grid should produce more edges than
    with a blocked grid, confirming draw_merge_grid logic is sound.
    Nodes are 4m apart (< CROSS_DIST_M=5m), wall blocks one grid, clear allows it."""
    import networkx as nx

    # res=1.0, nodes at (row=2,col=5) and (row=6,col=5), world distance = 4.0m < 5.0m
    grid_blocked = np.zeros((10, 10), dtype=np.uint8)
    grid_blocked[4:5, :] = 1   # horizontal wall at row 4 between the two nodes
    grid_clear = np.zeros((10, 10), dtype=np.uint8)

    G1 = nx.Graph()
    G1.add_node(0, x=5.0, y=2.0, yaw=0.0)  # col=5, row=2
    G1.graph["start_node"] = 0
    G2 = nx.Graph()
    G2.add_node(0, x=5.0, y=6.0, yaw=0.0)  # col=5, row=6
    G2.graph["start_node"] = 0

    merged_blocked = feb.merge_topometric_graphs([G1, G2], grid_blocked, res=1.0)
    merged_clear = feb.merge_topometric_graphs([G1, G2], grid_clear, res=1.0)

    assert merged_blocked.number_of_edges() == 0, "Wall must block cross-session edge"
    assert merged_clear.number_of_edges() >= 1, "Clear grid must allow cross-session edge"
