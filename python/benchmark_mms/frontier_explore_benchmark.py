#!/usr/bin/env python3
"""Frontier-Based Goal-Directed Exploration Benchmark — Octa Maze Demo.

All K sessions share the same (start, goal) pair. Session diversity comes from
per-session exploration perturbation (initial yaw + softmax-temperature frontier
selection), not from different starting positions.

As sessions accumulate, the shortest path on the merged topometric map converges
toward the ground-truth optimal path.

Usage
-----
python frontier_explore_benchmark.py --start 7 5 --goal 65 64
python frontier_explore_benchmark.py --start 7 5 --goal 65 64 --k 5 --seed 42
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.ndimage import binary_dilation

# ============================================================================
# Constants (maze-specific, 71×71 grid @ 0.5 m/cell)
# ============================================================================
GRID_RES_M = 0.5
N_SESSIONS = 5
FOV_HALF_DEG = 30.0
FOV_HALF_RAD = np.radians(FOV_HALF_DEG)
FOV_RANGE_M = 5.0
TRANS_THRESH_M = 2.0
ROT_THRESH_RAD = np.radians(60)
CROSS_DIST_M = 3.0
INFLATE_RADIUS = 1
FRONTIER_DIST_MIN = 30
TOPO_SNAP_DIST_M = 3.0
MAX_STEPS_COVERAGE_BUDGET = 0.5
FRONTIER_TEMP_MIN = 0.5
FRONTIER_TEMP_MAX = 5.0
FRONTIER_TOP_N = 5
PCD_HEIGHT_SLICE = 2.0
PCD_HEIGHT_TOL = 0.3
PCD_DILATE = 0
MASTER_SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PCD_PATH = SCRIPT_DIR / "data" / "octa_maze.pcd"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "octa_maze"


# ============================================================================
# PCD Loading
# ============================================================================
def load_pcd_grid(
    pcd_path: str | Path,
    resolution: float = 0.5,
    height_slice: float = PCD_HEIGHT_SLICE,
    height_tolerance: float = PCD_HEIGHT_TOL,
    dilate: int = PCD_DILATE,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Load ASCII PCD, crop Y-height slice, rasterize to 2D occupancy grid.

    Returns:
        grid: uint8 (0=free, 1=obstacle)
        x_range: (x_min, x_max) in meters
        z_range: (z_min, z_max) in meters
    """
    pts = np.loadtxt(pcd_path, skiprows=10)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

    mask = np.abs(pts[:, 1] - height_slice) <= height_tolerance
    pts_slice = pts[mask]
    print(f"  PCD height slice (Y={height_slice} +/-{height_tolerance}): "
          f"{len(pts_slice)} / {len(pts)} points")

    nx_cells = int((x_max - x_min) / resolution) + 1
    nz_cells = int((z_max - z_min) / resolution) + 1
    grid = np.zeros((nz_cells, nx_cells), dtype=np.uint8)
    xi = np.clip(((pts_slice[:, 0] - x_min) / resolution).astype(int), 0, nx_cells - 1)
    zi = np.clip(((pts_slice[:, 2] - z_min) / resolution).astype(int), 0, nz_cells - 1)
    grid[zi, xi] = 1

    if dilate > 0:
        se = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=bool)
        grid = binary_dilation(grid.astype(bool), structure=se).astype(np.uint8)

    print(f"  Occupancy grid: {grid.shape}  "
          f"obstacle={grid.sum()}/{grid.size} ({100 * grid.sum() / grid.size:.1f}%)")
    return grid, (x_min, x_max), (z_min, z_max)


# ============================================================================
# Grid Utilities
# ============================================================================
def inflate_grid(grid: np.ndarray, radius: int = INFLATE_RADIUS) -> np.ndarray:
    struct = np.ones((2 * radius + 1, 2 * radius + 1), bool)
    return binary_dilation(grid.astype(bool), structure=struct).astype(np.uint8)


def astar(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int], res: float = GRID_RES_M
) -> tuple[list | None, float]:
    if grid[start] or grid[goal]:
        return None, float("inf")
    H, W = grid.shape
    diag = res * np.sqrt(2)
    dirs = [
        (-1, 0, res), (1, 0, res), (0, -1, res), (0, 1, res),
        (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag),
    ]

    g_score = {start: 0.0}
    parent: dict = {}
    pq = [(res * np.hypot(start[0] - goal[0], start[1] - goal[1]), start)]

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path, g_score[goal]
        for dr, dc, step in dirs:
            nb = (cur[0] + dr, cur[1] + dc)
            if 0 <= nb[0] < H and 0 <= nb[1] < W and not grid[nb]:
                ng = g_score[cur] + step
                if ng < g_score.get(nb, float("inf")):
                    g_score[nb] = ng
                    parent[nb] = cur
                    h = res * np.hypot(nb[0] - goal[0], nb[1] - goal[1])
                    heapq.heappush(pq, (ng + h, nb))
    return None, float("inf")


def world_to_grid(
    point_world: tuple[float, float, float],
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    res: float = GRID_RES_M,
) -> tuple[int, int]:
    x, _y, z = point_world
    c = int((x - x_range[0]) / res)
    r = int((z - z_range[0]) / res)
    return r, c


# ============================================================================
# FOV Observation
# ============================================================================
def add_fov_observation(
    obs: np.ndarray,
    r: int,
    c: int,
    yaw: float,
    base_grid: np.ndarray,
    fov_range_m: float = FOV_RANGE_M,
    fov_half_rad: float = FOV_HALF_RAD,
    res: float = GRID_RES_M,
) -> None:
    H, W = obs.shape
    R = int(np.ceil(fov_range_m / res))
    dr_arr = np.arange(-R, R + 1)
    dc_arr = np.arange(-R, R + 1)
    dR, dC = np.meshgrid(dr_arr, dc_arr, indexing="ij")
    dist2 = dR.astype(float) ** 2 + dC.astype(float) ** 2
    range_mask = dist2 <= R**2
    angle_to_cell = np.arctan2(dR.astype(float), dC.astype(float))
    angle_diff = np.abs(np.arctan2(np.sin(angle_to_cell - yaw), np.cos(angle_to_cell - yaw)))
    fov_mask = (angle_diff <= fov_half_rad) & range_mask
    rr = np.clip(r + dR[fov_mask], 0, H - 1)
    cc = np.clip(c + dC[fov_mask], 0, W - 1)
    obs[rr[base_grid[rr, cc] == 0], cc[base_grid[rr, cc] == 0]] = -1
    obs[rr[base_grid[rr, cc] == 1], cc[base_grid[rr, cc] == 1]] = 1


def obs_to_planning_grid(
    obs: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
    inflate_radius: int = INFLATE_RADIUS,
    for_goal_check: bool = True,
) -> np.ndarray:
    """Build planning grid from partial observation.

    for_goal_check=True (goal reachability): only KNOWN free cells are traversable.
    for_goal_check=False (frontier navigation): unknown cells are also traversable.
    """
    if for_goal_check:
        pg = 1 - (obs == -1).astype(np.uint8)   # known free=0, rest=1
    else:
        pg = (obs == 1).astype(np.uint8)          # known obstacle=1, rest=0

    for pt in (start, goal):
        r_lo = max(0, pt[0] - inflate_radius)
        r_hi = min(pg.shape[0], pt[0] + inflate_radius + 1)
        c_lo = max(0, pt[1] - inflate_radius)
        c_hi = min(pg.shape[1], pt[1] + inflate_radius + 1)
        pg[r_lo:r_hi, c_lo:c_hi] = 0
    return pg


# ============================================================================
# Frontier Functions
# ============================================================================
def find_frontiers(obs: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Find frontier cells (unknown adjacent to free) and their nearest free neighbors.

    Returns:
        frontiers: list of (r, c) unknown cells bordering known free space
        free_neighbors: list of (r, c) known free cells adjacent to each frontier
    """
    free_mask = obs == -1
    frontiers: set[tuple[int, int]] = set()
    free_neighbor_map: dict[tuple[int, int], tuple[int, int]] = {}
    H, W = obs.shape
    free_rows, free_cols = np.where(free_mask)
    for r, c in zip(free_rows, free_cols):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and obs[nr, nc] == 0:
                fcell = (nr, nc)
                frontiers.add(fcell)
                free_neighbor_map[fcell] = (r, c)
    return list(frontiers), [free_neighbor_map[f] for f in frontiers]


def _fallback_select(order, frontiers, free_neighbors, inf_pg, current, res):
    for idx in order:
        if free_neighbors is not None:
            tgt = free_neighbors[idx]
        else:
            tgt = frontiers[idx]
        if tgt == current:
            continue
        _, length = astar(inf_pg, current, (int(tgt[0]), int(tgt[1])), res)
        if length < float("inf"):
            return (int(tgt[0]), int(tgt[1]))
    return None


def select_frontier(
    frontiers: list[tuple[int, int]],
    current: tuple[int, int],
    obs: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
    top_n: int,
    inf_pg: np.ndarray,
    res: float = GRID_RES_M,
    frontier_free_neighbors: list[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    """Softmax-temperature frontier selection.

    Filters top-N frontiers by Euclidean distance, computes A* path lengths
    to the nearest known-free neighbor of each frontier, applies softmax over
    -dist / T, and randomly picks one. Returns the nearest FREE cell adjacent
    to the chosen frontier (not the unknown frontier cell itself).

    If frontier_free_neighbors is provided, it maps frontier[i] → free_neighbor[i].
    """
    if not frontiers:
        return None

    f_arr = np.array(frontiers)
    cr, cc = current
    eucl_dists = np.abs(f_arr[:, 0] - cr) + np.abs(f_arr[:, 1] - cc)
    order = np.argsort(eucl_dists)
    top_k = min(top_n, len(frontiers))

    targets = []
    for idx in order[:top_k]:
        fr, fc = frontiers[idx]
        if frontier_free_neighbors is not None:
            tgt = frontier_free_neighbors[idx]
        else:
            tgt = (fr, fc)
        if tgt == (cr, cc):
            continue
        targets.append(tgt)

    if not targets:
        return _fallback_select(order, frontiers, frontier_free_neighbors, inf_pg, current, res)

    tgt_arr = np.array(targets)
    eucl_to_targets = np.abs(tgt_arr[:, 0] - cr) + np.abs(tgt_arr[:, 1] - cc)
    logits = -eucl_to_targets / max(temperature, 1e-6)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    chosen_idx = rng.choice(len(targets), p=probs)

    # Validate: A* must be reachable. If not, fallback.
    _, length = astar(inf_pg, current, (int(targets[chosen_idx][0]), int(targets[chosen_idx][1])), res)
    if length >= float("inf"):
        return _fallback_select(order, frontiers, frontier_free_neighbors, inf_pg, current, res)

    return (int(targets[chosen_idx][0]), int(targets[chosen_idx][1]))


# ============================================================================
# Topometric Graph
# ============================================================================
def build_topometric_subgraph(
    poses: list[tuple[int, int, float]],
    res: float = GRID_RES_M,
    trans_thresh: float = TRANS_THRESH_M,
    rot_thresh: float = ROT_THRESH_RAD,
) -> nx.Graph:
    G = nx.Graph()
    if not poses:
        return G
    r0, c0, y0 = poses[0]
    G.add_node(0, x=c0 * res, y=r0 * res, yaw=y0)
    node_idx = 0
    prev = (r0, c0, y0)
    for r, c, yaw in poses[1:]:
        pr, pc, py = prev
        trans = res * np.hypot(r - pr, c - pc)
        rot = abs(np.arctan2(np.sin(yaw - py), np.cos(yaw - py)))
        if trans > trans_thresh or rot > rot_thresh:
            node_idx += 1
            G.add_node(node_idx, x=c * res, y=r * res, yaw=yaw)
            dist_m = res * np.hypot(r - pr, c - pc)
            G.add_edge(node_idx - 1, node_idx, weight=dist_m,
                       rel_pose=(c - pc, r - pr, yaw - py))
            prev = (r, c, yaw)
    return G


def _line_free(r0: int, c0: int, r1: int, c1: int, base_grid: np.ndarray) -> bool:
    H, W = base_grid.shape
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0
    if dr > dc:
        err = dr // 2
        while r != r1:
            if 0 <= r < H and 0 <= c < W and base_grid[r, c] != 0:
                return False
            err -= dc
            if err < 0:
                c += sc
                err += dr
            r += sr
    else:
        err = dc // 2
        while c != c1:
            if 0 <= r < H and 0 <= c < W and base_grid[r, c] != 0:
                return False
            err -= dr
            if err < 0:
                r += sr
                err += dc
            c += sc
    if 0 <= r1 < H and 0 <= c1 < W and base_grid[r1, c1] != 0:
        return False
    return True


def merge_topometric_graphs(
    subgraphs: list[nx.Graph],
    base_grid: np.ndarray,
    res: float = GRID_RES_M,
    cross_dist: float = CROSS_DIST_M,
) -> nx.Graph:
    merged = nx.Graph()
    offset = 0
    subgraph_node_sets: list[set[int]] = []
    for G in subgraphs:
        mapping = {n: n + offset for n in G.nodes}
        merged.update(nx.relabel_nodes(G, mapping))
        subgraph_node_sets.append({n + offset for n in G.nodes})
        offset += G.number_of_nodes() + 1

    all_nodes = [(n, d["x"], d["y"]) for n, d in merged.nodes(data=True)]
    for si in range(len(subgraph_node_sets)):
        for sj in range(si + 1, len(subgraph_node_sets)):
            ni_list = [(n, x, y) for n, x, y in all_nodes if n in subgraph_node_sets[si]]
            nj_list = [(n, x, y) for n, x, y in all_nodes if n in subgraph_node_sets[sj]]
            for ni, xi, yi in ni_list:
                for nj, xj, yj in nj_list:
                    d = np.hypot(xi - xj, yi - yj)
                    if d < cross_dist and not merged.has_edge(ni, nj):
                        ri, ci = int(yi / res), int(xi / res)
                        rj, cj = int(yj / res), int(xj / res)
                        if _line_free(ri, ci, rj, cj, base_grid):
                            merged.add_edge(ni, nj, weight=d,
                                            rel_pose=(xj - xi, yj - yi, 0.0))
    return merged


def topo_path_length(
    topo_graph: nx.Graph,
    start: tuple[int, int],
    goal: tuple[int, int],
    res: float = GRID_RES_M,
    snap_dist: float = TOPO_SNAP_DIST_M,
) -> tuple[float, int | None, int | None]:
    """Shortest path on topo graph for (start, goal).

    Returns: (length_m, start_node, goal_node) or (float('inf'), None, None)
    """
    sx, sy = start[1] * res, start[0] * res
    gx, gy = goal[1] * res, goal[0] * res
    best_sn, best_gn = None, None
    best_sd, best_gd = float("inf"), float("inf")
    for n, d in topo_graph.nodes(data=True):
        ds = np.hypot(d["x"] - sx, d["y"] - sy)
        dg = np.hypot(d["x"] - gx, d["y"] - gy)
        if ds < best_sd:
            best_sd, best_sn = ds, n
        if dg < best_gd:
            best_gd, best_gn = dg, n
    if best_sn is None or best_gn is None or best_sd > snap_dist or best_gd > snap_dist:
        return float("inf"), None, None
    try:
        length = nx.shortest_path_length(topo_graph, best_sn, best_gn, weight="weight")
        return length + best_sd + best_gd, best_sn, best_gn
    except nx.NetworkXNoPath:
        return float("inf"), None, None


# ============================================================================
# Frontier Exploration — Single Session
# ============================================================================
def frontier_explore_session(
    start: tuple[int, int],
    goal: tuple[int, int],
    base_grid: np.ndarray,
    rng: np.random.Generator,
    initial_yaw: float,
    frontier_temperature: float,
    res: float = GRID_RES_M,
    fov_range_m: float = FOV_RANGE_M,
    fov_half_rad: float = FOV_HALF_RAD,
    inflate_radius: int = INFLATE_RADIUS,
    max_steps: int | None = None,
    top_n: int = FRONTIER_TOP_N,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """Run one session of frontier-based goal-directed exploration.

    Returns:
        trajectory: list of (r, c, yaw) poses
        obs: int8 observation grid (-1=free, 1=obstacle, 0=unknown)
    """
    H, W = base_grid.shape
    if max_steps is None:
        max_steps = int(H * W * MAX_STEPS_COVERAGE_BUDGET)

    obs = np.zeros_like(base_grid, dtype=np.int8)
    traj: list[tuple[int, int, float]] = [(start[0], start[1], initial_yaw)]
    visited_for_fov: set[tuple[int, int]] = {(start[0], start[1])}

    def _scan_position(pr: int, pc: int, pyaw: float) -> None:
        add_fov_observation(obs, pr, pc, pyaw, base_grid, fov_range_m, fov_half_rad, res)

    def _try_goal(pr: int, pc: int) -> tuple[list[tuple[int, int]] | None, np.ndarray]:
        pg = obs_to_planning_grid(obs, start, goal, inflate_radius, for_goal_check=True)
        inf_pg = inflate_grid(pg, inflate_radius)
        return astar(inf_pg, (pr, pc), goal, res)

    _scan_position(traj[0][0], traj[0][1], traj[0][2])

    for _step in range(max_steps):
        r, c, yaw = traj[-1]

        path_to_goal, _ = _try_goal(r, c)
        if path_to_goal is not None and len(path_to_goal) > 1:
            for nr, nc in path_to_goal[1:]:
                ny = np.arctan2(nr - traj[-1][0], nc - traj[-1][1])
                traj.append((nr, nc, float(ny)))
                if (nr, nc) not in visited_for_fov:
                    visited_for_fov.add((nr, nc))
                    _scan_position(nr, nc, float(ny))
            break

        frontiers, free_neighbors = find_frontiers(obs)
        if not frontiers:
            break

        pg = obs_to_planning_grid(obs, start, goal, inflate_radius, for_goal_check=False)
        inf_pg = pg  # no inflation for frontier navigation — need to traverse into unknown

        next_f = select_frontier(frontiers, (r, c), obs, rng,
                                 frontier_temperature, top_n, inf_pg, res,
                                 frontier_free_neighbors=free_neighbors)
        if next_f is None:
            break

        path_to_f, _ = astar(inf_pg, (r, c), next_f, res)
        if path_to_f is None or len(path_to_f) < 2:
            break

        for nr, nc in path_to_f[1:]:
            ny = np.arctan2(nr - traj[-1][0], nc - traj[-1][1])
            traj.append((nr, nc, float(ny)))

            if (nr, nc) not in visited_for_fov:
                visited_for_fov.add((nr, nc))
                _scan_position(nr, nc, float(ny))

            goal_path, _ = _try_goal(nr, nc)
            if goal_path is not None and len(goal_path) > 1:
                for gr, gc in goal_path[1:]:
                    gy = np.arctan2(gr - traj[-1][0], gc - traj[-1][1])
                    traj.append((gr, gc, float(gy)))
                    if (gr, gc) not in visited_for_fov:
                        visited_for_fov.add((gr, gc))
                        _scan_position(gr, gc, float(gy))
                return traj, obs

    return traj, obs


# ============================================================================
# Visualization
# ============================================================================
STYLE_FREE = np.array([243, 244, 246]) / 255.0
STYLE_OBS = np.array([31, 41, 55]) / 255.0
BG_COLOR = "#111827"
COLOR_TOPO_NODE = "#3B82F6"
COLOR_TOPO_EDGE_INTRA = "#60A5FA"
COLOR_TOPO_EDGE_CROSS = "#FBBF24"
COLOR_TOPO_PATH = "#F97316"
COLOR_GT_PATH = "#F97316"
COLOR_START = "#10B981"
COLOR_GOAL = "#EF4444"
COLOR_COV_HIST = "#10B981"
COLOR_COV_NEW = "#06B6D4"
COLOR_TRAJ = "#FFFFFF"
# Pre-convert for RGBA overlay usage
_COV_HIST_RGB = tuple(int(COLOR_COV_HIST.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4))
_COV_NEW_RGB = tuple(int(COLOR_COV_NEW.lstrip("#")[i:i+2], 16) / 255.0 for i in (0, 2, 4))
ALPHA_COV_HIST = 0.35
ALPHA_COV_NEW = 0.50


def _draw_base_grid(ax, base_grid):
    H, W = base_grid.shape
    rgb = np.full((H, W, 3), STYLE_FREE, dtype=np.float64)
    rgb[base_grid == 1] = STYLE_OBS
    ax.imshow(rgb, origin="upper", interpolation="none")


def _draw_topo_graph(ax, G, node_color=COLOR_TOPO_NODE, edge_color=COLOR_TOPO_EDGE_INTRA,
                     edge_alpha=0.5, node_size=15, edge_lw=0.6, res=GRID_RES_M):
    for u, v in G.edges():
        xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
        xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
        ax.plot([xu / res, xv / res], [yu / res, yv / res],
                color=edge_color, alpha=edge_alpha, linewidth=edge_lw)
    xs = [d["x"] / res for _, d in G.nodes(data=True)]
    ys = [d["y"] / res for _, d in G.nodes(data=True)]
    ax.scatter(xs, ys, c=node_color, s=node_size, zorder=5)


def _draw_path(ax, path_nodes, G, color=COLOR_TOPO_PATH, lw=2.0, res=GRID_RES_M):
    if not path_nodes or len(path_nodes) < 2:
        return
    xs, ys = [], []
    for n in path_nodes:
        xs.append(G.nodes[n]["x"] / res)
        ys.append(G.nodes[n]["y"] / res)
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=6)


def _draw_gt_path(ax, gt_path, lw=1.5, res=GRID_RES_M):
    if gt_path and len(gt_path) > 1:
        pts = np.array(gt_path)
        ax.plot(pts[:, 1], pts[:, 0], color=COLOR_GT_PATH, linewidth=lw,
                linestyle="--", alpha=0.7, zorder=4)


def fig1_session_exploration(
    base_grid: np.ndarray,
    all_poses: list[list[tuple[int, int, float]]],
    all_obs: list[np.ndarray],
    subgraphs: list[nx.Graph],
    merged_topo: nx.Graph,
    start: tuple[int, int],
    goal: tuple[int, int],
    gt_path: list[tuple[int, int]] | None,
    gt_len: float,
    temperatures: list[float],
    res: float = GRID_RES_M,
    output_path: str | Path = "fig1_session_exploration.png",
) -> None:
    K = len(subgraphs)
    ncols = 3
    nrows = (K + 1 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows),
                             facecolor=BG_COLOR)
    axes = np.atleast_1d(axes).flatten()

    for k in range(K):
        ax = axes[k]
        ax.set_facecolor(BG_COLOR)
        _draw_base_grid(ax, base_grid)

        Gk = subgraphs[k]
        if Gk.number_of_nodes() > 0:
            _draw_topo_graph(ax, Gk, res=res)
            tlen, sn, gn = topo_path_length(Gk, start, goal, res)
            reachable = "YES" if sn is not None and gn is not None else "NO"
            if reachable == "YES":
                try:
                    sp_nodes = nx.shortest_path(Gk, sn, gn, weight="weight")
                    _draw_path(ax, sp_nodes, Gk, res=res)
                except nx.NetworkXNoPath:
                    reachable = "NO"
        else:
            tlen, reachable = float("inf"), "NO"

        ax.scatter(start[1], start[0], c=COLOR_START, s=80, marker="o",
                   edgecolors="white", linewidths=1.5, zorder=7)
        ax.scatter(goal[1], goal[0], c=COLOR_GOAL, s=80, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=7)
        title = (f"Session {k}  T={temperatures[k]:.1f}  "
                 f"nodes={Gk.number_of_nodes()}  "
                 f"reach={reachable}")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Merged panel
    ax = axes[K]
    ax.set_facecolor(BG_COLOR)
    _draw_base_grid(ax, base_grid)
    if merged_topo.number_of_nodes() > 0:
        _draw_topo_graph(ax, merged_topo,
                         edge_color=COLOR_TOPO_EDGE_INTRA, res=res)
        # highlight cross-session edges in yellow
        sub_node_sets = []
        off = 0
        for Gk in subgraphs:
            sub_node_sets.append(set(range(off, off + Gk.number_of_nodes())))
            off += Gk.number_of_nodes() + 1
        for si in range(len(sub_node_sets)):
            for sj in range(si + 1, len(sub_node_sets)):
                for u in sub_node_sets[si]:
                    for v in sub_node_sets[sj]:
                        if merged_topo.has_edge(u, v):
                            xu = merged_topo.nodes[u]["x"] / res
                            yu = merged_topo.nodes[u]["y"] / res
                            xv = merged_topo.nodes[v]["x"] / res
                            yv = merged_topo.nodes[v]["y"] / res
                            ax.plot([xu, xv], [yu, yv],
                                    color=COLOR_TOPO_EDGE_CROSS, alpha=0.5,
                                    linewidth=0.6, zorder=4)
        nodes_x = [d["x"] / res for _, d in merged_topo.nodes(data=True)]
        nodes_y = [d["y"] / res for _, d in merged_topo.nodes(data=True)]
        ax.scatter(nodes_x, nodes_y, c=COLOR_TOPO_NODE, s=15, zorder=5)
        tlen, sn, gn = topo_path_length(merged_topo, start, goal, res)
        if sn is not None and gn is not None:
            try:
                sp_nodes = nx.shortest_path(merged_topo, sn, gn, weight="weight")
                _draw_path(ax, sp_nodes, merged_topo, res=res)
            except nx.NetworkXNoPath:
                tlen = float("inf")

    ax.scatter(start[1], start[0], c=COLOR_START, s=80, marker="o",
               edgecolors="white", linewidths=1.5, zorder=7)
    ax.scatter(goal[1], goal[0], c=COLOR_GOAL, s=80, marker="X",
               edgecolors="white", linewidths=1.5, zorder=7)
    _draw_gt_path(ax, gt_path, res=res)
    ratio = tlen / gt_len if gt_len > 0 and tlen < float("inf") else float("inf")
    title = (f"Merged (k=0..{K - 1})  "
             f"topo={tlen:.1f}m  GT={gt_len:.1f}m  "
             f"ratio={ratio:.2f}" if ratio < float("inf") else
             f"Merged (k=0..{K - 1})  unreachable")
    ax.set_title(title, color="white", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    for ax_extra in axes[K + 1:]:
        ax_extra.set_visible(False)

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved {output_path}")


def fig2_optimality_curve(
    ratios: list[float],
    gt_len: float,
    res: float = GRID_RES_M,
    output_path: str | Path = "fig2_optimality_curve.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    k_vals = list(range(1, len(ratios) + 1))
    finite_mask = np.array([r < float("inf") for r in ratios])
    finite_ratios = np.array(ratios)[finite_mask]
    finite_k = np.array(k_vals)[finite_mask]

    if len(finite_k) > 0:
        ax.plot(finite_k, finite_ratios, color="#3B82F6", marker="o",
                markersize=8, linewidth=2, label="topo ratio")
        for k, r in zip(finite_k, finite_ratios):
            ax.annotate(f"r={r:.2f}", (k, r), textcoords="offset points",
                        xytext=(0, 10), ha="center", color="white", fontsize=9)

    inf_k = np.array(k_vals)[~finite_mask]
    if len(inf_k) > 0:
        ax.scatter(inf_k, [1.3] * len(inf_k), marker="x", color="#6B7280",
                   s=60, zorder=6)
        for k in inf_k:
            ax.text(k, 1.35, "unreachable", ha="center", color="#6B7280", fontsize=8)

    ax.axhline(y=1.0, color=COLOR_GOAL, linestyle="--", linewidth=1.5, label="GT optimal (1.0)")
    ax.set_xlabel("Number of Sessions (k)", color="white", fontsize=12)
    ax.set_ylabel("Optimality Ratio (topo_len / GT_len)", color="white", fontsize=12)
    ax.set_title("Path Optimality vs. Number of Sessions", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.set_xticks(range(1, len(ratios) + 1))
    ax.legend(loc="upper right", facecolor=BG_COLOR, edgecolor="white",
              labelcolor="white", fontsize=10)
    ax.set_xlim(0.5, len(ratios) + 0.5)
    y_max = max(1.1, np.max(finite_ratios) * 1.15 if len(finite_ratios) > 0 else 1.5)
    ax.set_ylim(0.9, y_max)
    for spine in ax.spines.values():
        spine.set_color("white")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved {output_path}")


def fig3_reachability_coverage(
    base_grid: np.ndarray,
    all_poses: list[list[tuple[int, int, float]]],
    all_obs: list[np.ndarray],
    start: tuple[int, int],
    goal: tuple[int, int],
    res: float = GRID_RES_M,
    output_path: str | Path = "fig3_reachability_coverage.png",
) -> None:
    K = len(all_obs)
    total_free = int((base_grid == 0).sum())
    ncols = 3
    nrows = (K + 1 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows),
                             facecolor=BG_COLOR)
    axes = np.atleast_1d(axes).flatten()

    cov_pcts = []
    cum_free_mask = np.zeros(base_grid.shape, dtype=bool)

    for k in range(K):
        ax = axes[k]
        ax.set_facecolor(BG_COLOR)
        _draw_base_grid(ax, base_grid)

        prev_mask = cum_free_mask.copy()
        new_free = (all_obs[k] == -1)
        cum_free_mask |= new_free

        cov_layer = np.zeros((*base_grid.shape, 4))
        cov_layer[prev_mask, :] = (*_COV_HIST_RGB, ALPHA_COV_HIST)
        cov_layer[new_free & ~prev_mask, :] = (*_COV_NEW_RGB, ALPHA_COV_NEW)
        ax.imshow(cov_layer, origin="upper", interpolation="none", zorder=2)

        traj = all_poses[k]
        if traj:
            tr = np.array([(p[0], p[1]) for p in traj])
            ax.plot(tr[:, 1], tr[:, 0], color=COLOR_TRAJ, linewidth=0.8, alpha=0.8, zorder=3)

        total_cov = int(cum_free_mask.sum())
        new_cov = int((new_free & ~prev_mask).sum())
        total_pct = 100.0 * total_cov / total_free if total_free > 0 else 0
        new_pct = 100.0 * new_cov / total_free if total_free > 0 else 0
        cov_pcts.append(total_pct)

        ax.scatter(start[1], start[0], c=COLOR_START, s=60, marker="o",
                   edgecolors="white", linewidths=1, zorder=7)
        ax.scatter(goal[1], goal[0], c=COLOR_GOAL, s=60, marker="X",
                   edgecolors="white", linewidths=1, zorder=7)
        cov_m2 = total_cov * res * res
        title = (f"k={k + 1}  new_cov={new_pct:.0f}%  "
                 f"total_cov={total_pct:.0f}%  area={cov_m2:.0f}m²")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Line plot panel
    ax = axes[K]
    ax.set_facecolor(BG_COLOR)
    ks = list(range(1, K + 1))
    ax.bar(ks, cov_pcts, color=COLOR_COV_NEW, alpha=0.7, label="cumulative coverage")
    ax.plot(ks, cov_pcts, color=COLOR_TOPO_NODE, marker="o", markersize=6, linewidth=2)
    for k, pct in zip(ks, cov_pcts):
        ax.annotate(f"{pct:.0f}%", (k, pct), textcoords="offset points",
                    xytext=(0, 8), ha="center", color="white", fontsize=9)
    ax.set_xlabel("Number of Sessions (k)", color="white", fontsize=10)
    ax.set_ylabel("Coverage (%)", color="white", fontsize=10)
    ax.set_title("Cumulative Coverage Growth", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(cov_pcts) * 1.2 if cov_pcts else 100)
    for spine in ax.spines.values():
        spine.set_color("white")

    for ax_extra in axes[K + 1:]:
        ax_extra.set_visible(False)

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved {output_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Frontier-Based Goal-Directed Exploration Benchmark — Octa Maze")
    parser.add_argument("--start", type=int, nargs=2, required=True,
                        metavar=("R", "C"), help="Fixed start cell (row, col)")
    parser.add_argument("--goal", type=int, nargs=2, required=True,
                        metavar=("R", "C"), help="Fixed goal cell (row, col)")
    parser.add_argument("--res_m", type=float, default=GRID_RES_M,
                        help=f"Grid resolution (m/cell, default={GRID_RES_M})")
    parser.add_argument("--k", type=int, default=N_SESSIONS,
                        help=f"Number of sessions (default={N_SESSIONS})")
    parser.add_argument("--seed", type=int, default=MASTER_SEED,
                        help=f"Master random seed (default={MASTER_SEED})")
    parser.add_argument("--pcd", type=str, default=str(DEFAULT_PCD_PATH),
                        help=f"Path to octa_maze.pcd (default={DEFAULT_PCD_PATH})")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default={DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    start = (args.start[0], args.start[1])
    goal = (args.goal[0], args.goal[1])
    K = args.k
    seed = args.seed
    pcd_path = Path(args.pcd)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    res = args.res_m

    # ------------------------------------------------------------------
    print(f"=== Frontier Exploration Benchmark: Octa Maze ===")
    print(f"  start={(start)}, goal={(goal)}, K={K}, seed={seed}")
    print(f"  PCD: {pcd_path}")
    print(f"  output: {output_dir}")

    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n--- Loading PCD ---")
    base_grid, x_range, z_range = load_pcd_grid(pcd_path, res)
    np.save(output_dir / "base_map.npy", base_grid)
    H, W = base_grid.shape

    # Validate start / goal
    if base_grid[start] == 1:
        print(f"ERROR: start {start} is on obstacle")
        return 1
    if base_grid[goal] == 1:
        print(f"ERROR: goal {goal} is on obstacle")
        return 1

    inf_full = inflate_grid(base_grid)
    gt_path, gt_len = astar(inf_full, start, goal, res)
    if gt_path is None:
        print(f"ERROR: start {start} → goal {goal} GT unreachable on base map")
        return 1
    print(f"  GT path length: {gt_len:.1f} m, cells={len(gt_path)}")

    # Validate distance constraint
    eucl_cells = np.hypot(start[0] - goal[0], start[1] - goal[1])
    if eucl_cells < FRONTIER_DIST_MIN:
        print(f"WARNING: Euclidean distance {eucl_cells:.0f} cells < "
              f"FRONTIER_DIST_MIN={FRONTIER_DIST_MIN}")

    # Save fixed_pair.json
    fixed_pair = {
        "start": list(start), "goal": list(goal),
        "world_start": [2.5, 2.0, 3.5], "world_goal": [32.0, 2.0, 32.5],
        "grid_shape": [H, W], "res_m": res,
        "gt_len_m": round(gt_len, 3), "eucl_cells": round(eucl_cells, 1),
    }
    with open(output_dir / "fixed_pair.json", "w") as f:
        json.dump(fixed_pair, f, indent=2)
    print(f"  fixed_pair.json saved")

    # ------------------------------------------------------------------
    print(f"\n--- Running {K} Frontier Exploration Sessions ---")

    all_poses: list[list[tuple[int, int, float]]] = []
    all_obs: list[np.ndarray] = []
    subgraphs: list[nx.Graph] = []
    merged_topo_list: list[nx.Graph] = []
    ratios: list[float] = []
    temperatures: list[float] = []

    for k in range(K):
        rng = np.random.default_rng(seed + k)
        initial_yaw = k * (2 * np.pi / K)
        temperature = FRONTIER_TEMP_MIN + k * (FRONTIER_TEMP_MAX - FRONTIER_TEMP_MIN) / max(K - 1, 1)
        temperatures.append(temperature)
        max_steps = int(H * W * MAX_STEPS_COVERAGE_BUDGET)

        t1 = time.time()
        print(f"  Session {k}: yaw={np.degrees(initial_yaw):.0f}°, "
              f"T={temperature:.2f}, max_steps={max_steps}")

        traj, obs = frontier_explore_session(
            start, goal, base_grid, rng, initial_yaw, temperature,
            res=res, max_steps=max_steps, verbose=False)
        dt = time.time() - t1

        subgraph = build_topometric_subgraph(traj, res=res)
        all_poses.append(traj)
        all_obs.append(obs)
        subgraphs.append(subgraph)  # solo subgraph for session k

        merged = merge_topometric_graphs(subgraphs, base_grid, res=res) if subgraphs else nx.Graph()
        merged_topo_list.append(merged)

        tlen, _, _ = topo_path_length(merged, start, goal, res=res)
        ratio = tlen / gt_len if tlen < float("inf") and gt_len > 0 else float("inf")
        ratios.append(ratio)

        ratio_str = f"{ratio:.3f}" if ratio < float("inf") else "inf"
        print(f"    traj={len(traj)} steps, topo_nodes={subgraph.number_of_nodes()}, "
              f"merged_nodes={merged.number_of_nodes()}, "
              f"topo_len={tlen:.1f}m, ratio={ratio_str}, "
              f"time={dt:.1f}s")

        np.save(output_dir / "data" / f"session_{k}_poses.npy",
                np.array(traj, dtype=np.float32))
        np.save(output_dir / "data" / f"session_{k}_obs.npy", obs)

        merged_obs = np.zeros_like(base_grid, dtype=np.int8)
        for prev_obs in all_obs:
            merged_obs[prev_obs == -1] = -1
            merged_obs[prev_obs == 1] = 1
        np.save(output_dir / "data" / f"merged_obs_k{k}.npy", merged_obs)

    # ------------------------------------------------------------------
    print(f"\n--- Summary ---")
    print(f"  {'k':>3}  {'T':>5}  {'yaw':>6}  {'nodes':>5}  "
          f"{'reach':>5}  {'topo_m':>8}  {'GT_m':>8}  {'ratio':>7}")
    for k in range(K):
        tlen, sn, gn = topo_path_length(merged_topo_list[k], start, goal, res=res)
        reachable = "YES" if sn is not None and gn is not None else "NO"
        ratio = tlen / gt_len if tlen < float("inf") and gt_len > 0 else float("inf")
        ratio_str = f"{ratio:.3f}" if ratio < float("inf") else "inf"
        print(f"  {k:3d}  {temperatures[k]:5.2f}  "
              f"{np.degrees(k * 2 * np.pi / K):5.0f}°  "
              f"{subgraphs[k].number_of_nodes():5d}  "
              f"{reachable:5s}  {tlen:8.1f}  {gt_len:8.1f}  {ratio_str:>7s}")

    # ------------------------------------------------------------------
    print(f"\n--- Saving metrics ---")
    metrics = {
        "K": K, "seed": seed, "res_m": res,
        "grid_shape": [H, W],
        "start": list(start), "goal": list(goal),
        "gt_len_m": round(gt_len, 3),
        "ratios": [round(r, 4) if r < float("inf") else "inf" for r in ratios],
        "temperatures": [round(t, 2) for t in temperatures],
        "node_counts": [G.number_of_nodes() for G in subgraphs],
        "merged_node_counts": [G.number_of_nodes() for G in merged_topo_list],
        "traj_lengths": [len(p) for p in all_poses],
    }
    with open(output_dir / "data" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    print(f"\n--- Generating Figures (300 dpi) ---")
    fig1_session_exploration(
        base_grid, all_poses, all_obs, subgraphs, merged_topo_list[-1],
        start, goal, gt_path, gt_len, temperatures, res=res,
        output_path=output_dir / "fig1_session_exploration.png")
    fig2_optimality_curve(ratios, gt_len, res=res,
                          output_path=output_dir / "fig2_optimality_curve.png")
    fig3_reachability_coverage(
        base_grid, all_poses, all_obs, start, goal, res=res,
        output_path=output_dir / "fig3_reachability_coverage.png")

    print(f"\nDone. Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
