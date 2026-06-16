#!/usr/bin/env python3
"""Frontier-Based Goal-Directed Exploration Benchmark — Octa Maze Demo.

All K sessions share the same (start, goal) pair. Session diversity comes from
per-session exploration perturbation (initial yaw + softmax-temperature frontier
selection), not from different starting positions.

As sessions accumulate, the shortest path on the merged topometric map converges
toward the ground-truth optimal path.

Usage
-----
python frontier_explore_benchmark.py --start 1.4 1.0 --goal 13.0 12.8
python frontier_explore_benchmark.py --start 1.4 1.0 --goal 13.0 12.8 --k 5 --seed 42
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import time
from pathlib import Path
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.ndimage import binary_dilation


def _acquire_color_palette() -> np.ndarray:
    t = np.linspace(-510, 510, 60)
    palette = np.round(np.clip(
        np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255
    )).astype("float32") / 255
    palette[0] = [0, 152 / 255, 83 / 255]
    palette[1] = [228 / 255, 53 / 255, 39 / 255]
    palette[2] = [140 / 255, 3 / 255, 120 / 255]
    palette[3] = [0, 95 / 255, 129 / 255]
    palette[4] = [0.9290, 0.6940, 0.1250]
    palette[5] = [0.6350, 0.0780, 0.1840]
    palette[6] = [0.494, 0.184, 0.556]
    palette[7] = [0.850, 0.3250, 0.0980]
    palette[8] = [0.466, 0.674, 0.188]
    palette[9] = [0.3010, 0.7450, 0.9330]
    return palette


def _acquire_markers() -> list[str]:
    return ["o", "s", "^", "D", "X", "*", "+"]


def _acquire_linestyles() -> list[str]:
    return ["-", "--", "-.", ":", "--", "-.", ":"]


def _setting_font(
    fontsize: int = 12,
    titlesize: int = 12,
    legend_fontsize: int = 10,
) -> None:
    try:
        from colorama import init as colorama_init
        colorama_init(autoreset=True)
    except ImportError:
        pass
    try:
        from matplotlib import rc, pylab
        rc("font", **{"family": "serif", "serif": ["Palatino"], "size": fontsize})
        rc("text", usetex=True)
        pylab.rcParams.update({
            "axes.titlesize": titlesize,
            "legend.fontsize": legend_fontsize,
            "legend.numpoints": 1,
        })
    except Exception:
        from matplotlib import rc, pylab
        rc("font", **{"family": "serif", "serif": ["DejaVu Serif"], "size": fontsize})
        rc("text", usetex=False)
        pylab.rcParams.update({
            "axes.titlesize": titlesize,
            "legend.fontsize": legend_fontsize,
            "legend.numpoints": 1,
        })


_PALETTE = _acquire_color_palette()
_MARKERS = _acquire_markers()
_LINESTYLES = _acquire_linestyles()

# ============================================================================
# Constants (maze-specific, 71×71 grid @ 0.5 m/cell)
# ============================================================================
GRID_RES_M = 0.2
N_SESSIONS = 5
FOV_HALF_DEG = 45.0
FOV_HALF_RAD = np.radians(FOV_HALF_DEG)
FOV_RANGE_M = 5.0
TRANS_THRESH_M = 5.0
ROT_THRESH_RAD = np.radians(60)
CROSS_DIST_M = 5.0
FRONTIER_DIST_MIN = 75
TOPO_SNAP_DIST_M = 3.0
MAX_STEPS_COVERAGE_BUDGET = 0.1
FRONTIER_TEMP_MIN = 0.5
FRONTIER_TEMP_MAX = 3.0
FRONTIER_TEMP_FIXED = 2.5
FRONTIER_TOP_N = 5
PCD_HEIGHT_SLICE = 1.5
PCD_HEIGHT_TOL = 0.1
PCD_DILATE = 1
MASTER_SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PCD_PATH = SCRIPT_DIR / "data" / "octa_maze.pcd"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "octa_maze"


# ============================================================================
# PCD Loading
# ============================================================================
def _parse_pcd_header(
    pcd_path: str | Path,
) -> tuple[int, int, bool]:
    """Parse PCD header, return (header_byte_count, num_fields, is_binary)."""
    with open(pcd_path, "rb") as f:
        raw = f.read()
    hdr_end = raw.find(b"DATA ")
    if hdr_end < 0:
        raise ValueError("PCD header not found")
    next_nl = raw.find(b"\n", hdr_end)
    header_bytes = next_nl + 1

    header_text = raw[:header_bytes].decode()
    is_binary = "binary" in header_text.split("DATA")[-1].lower() if "DATA" in header_text else False

    num_fields = 3
    for line in header_text.split("\n"):
        if line.startswith("FIELDS"):
            num_fields = len(line.split()) - 1
            break
    return header_bytes, num_fields, is_binary


def _read_pcd_points(
    pcd_path: str | Path, header_bytes: int, n_fields: int, is_binary: bool,
) -> np.ndarray:
    """Read point cloud data from PCD file."""
    if is_binary:
        data = np.fromfile(str(pcd_path), dtype=np.float32, offset=header_bytes)
        n_points, _, _, _ = _parse_pcd_metadata(pcd_path, header_bytes)
        return data[:n_points * n_fields].reshape(-1, n_fields)
    else:
        all_lines = Path(pcd_path).read_text().splitlines()
        n_header = 0
        for i, line in enumerate(all_lines):
            n_header = i + 1
            if line.strip().startswith("DATA"):
                break
        return np.loadtxt(str(pcd_path), skiprows=n_header)


def _parse_pcd_metadata(
    pcd_path: str | Path, header_bytes: int,
) -> tuple[int, int, int, int]:
    """Return (n_points, width, height, num_fields) from PCD header."""
    with open(pcd_path, "rb") as f:
        header = f.read(header_bytes).decode()
    n_points = width = height = num_fields = 3
    for line in header.split("\n"):
        if line.startswith("POINTS"):
            n_points = int(line.split()[1])
        if line.startswith("WIDTH"):
            width = int(line.split()[1])
        if line.startswith("HEIGHT"):
            height = int(line.split()[1])
        if line.startswith("FIELDS"):
            num_fields = len(line.split()) - 1
    if n_points != width * height:
        n_points = width * height
    return n_points, width, height, num_fields


def load_pcd_grid(
    pcd_path: str | Path,
    resolution: float = 0.5,
    height_slice: float = PCD_HEIGHT_SLICE,
    height_tolerance: float = PCD_HEIGHT_TOL,
    dilate: int = PCD_DILATE,
    col_axis: int = 0,
    row_axis: int = 1,
    height_axis: int = 2,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Load ASCII or binary PCD, crop a height band, rasterize to 2D occupancy grid.

    Args:
        col_axis:  PCD field index used for grid columns (default 0 = X).
        row_axis:  PCD field index used for grid rows    (default 1 = Y).
        height_axis: PCD field index used for the height-slice filter (default 2 = Z).

    Returns:
        grid: uint8 (0=free, 1=obstacle)
        col_range: (min, max) along col_axis
        row_range: (min, max) along row_axis
    """
    hdr_bytes, n_fields, is_binary = _parse_pcd_header(pcd_path)
    pts = _read_pcd_points(pcd_path, hdr_bytes, n_fields, is_binary)

    col_vals = pts[:, col_axis]
    row_vals = pts[:, row_axis]
    ht_vals = pts[:, height_axis]

    col_min, col_max = col_vals.min(), col_vals.max()
    row_min, row_max = row_vals.min(), row_vals.max()

    mask = np.abs(ht_vals - height_slice) <= height_tolerance
    print(f"  PCD height slice (axis={height_axis} @{height_slice} +/-{height_tolerance}): "
          f"{mask.sum()} / {len(pts)} points")
    print(f"  col_range=[{col_min:.1f}, {col_max:.1f}]  row_range=[{row_min:.1f}, {row_max:.1f}]")

    ncols = int((col_max - col_min) / resolution) + 1
    nrows = int((row_max - row_min) / resolution) + 1
    grid = np.zeros((nrows, ncols), dtype=np.uint8)

    ci = np.clip(((col_vals[mask] - col_min) / resolution).astype(int), 0, ncols - 1)
    ri = np.clip(((row_vals[mask] - row_min) / resolution).astype(int), 0, nrows - 1)
    grid[ri, ci] = 1

    if dilate > 0:
        se = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=bool)
        grid = binary_dilation(grid.astype(bool), structure=se).astype(np.uint8)

    print(f"  Occupancy grid: {grid.shape}  "
          f"obstacle={grid.sum()}/{grid.size} ({100 * grid.sum() / grid.size:.1f}%)")
    return grid, (col_min, col_max), (row_min, row_max)


# ============================================================================
# Grid Utilities
# ============================================================================
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
    closed: set[tuple[int, int]] = set()

    while pq:
        _, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        closed.add(cur)
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


def astar_local(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    res: float = GRID_RES_M,
    margin: int = 50,
) -> tuple[list | None, float]:
    r_min = max(0, min(start[0], goal[0]) - margin)
    r_max = min(grid.shape[0] - 1, max(start[0], goal[0]) + margin)
    c_min = max(0, min(start[1], goal[1]) - margin)
    c_max = min(grid.shape[1] - 1, max(start[1], goal[1]) + margin)

    sub_grid = grid[r_min:r_max + 1, c_min:c_max + 1]
    local_start = (start[0] - r_min, start[1] - c_min)
    local_goal = (goal[0] - r_min, goal[1] - c_min)
    local_path, path_len = astar(sub_grid, local_start, local_goal, res)
    if local_path is None:
        return None, float("inf")
    path = [(r + r_min, c + c_min) for r, c in local_path]
    return path, path_len


def world_to_grid(
    col_val: float,
    row_val: float,
    col_range: tuple[float, float],
    row_range: tuple[float, float],
    res: float = GRID_RES_M,
) -> tuple[int, int]:
    c = int((col_val - col_range[0]) / res)
    r = int((row_val - row_range[0]) / res)
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
    if R <= 0:
        return
    angle_step = max(np.arctan(1.0 / R), 1e-3)
    n_rays = max(2, int(np.ceil((2 * fov_half_rad) / angle_step)) + 1)
    for theta in np.linspace(yaw - fov_half_rad, yaw + fov_half_rad, n_rays):
        seen_cells: set[tuple[int, int]] = set()
        for step in range(1, R + 1):
            nr = int(round(r + step * np.sin(theta)))
            nc = int(round(c + step * np.cos(theta)))
            if (nr, nc) in seen_cells:
                continue
            seen_cells.add((nr, nc))
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if base_grid[nr, nc] == 1:
                obs[nr, nc] = 1
                break
            obs[nr, nc] = -1


def obs_to_planning_grid(
    obs: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
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
        r_lo = max(0, pt[0] - 1)
        r_hi = min(pg.shape[0], pt[0] + 2)
        c_lo = max(0, pt[1] - 1)
        c_hi = min(pg.shape[1], pt[1] + 2)
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


def _fallback_select(order, frontiers, free_neighbors, inf_pg, current, res, local_margin=None):
    for idx in order:
        if free_neighbors is not None:
            tgt = free_neighbors[idx]
        else:
            tgt = frontiers[idx]
        if tgt == current:
            continue
        planner = astar_local if local_margin is not None else astar
        if local_margin is not None:
            _, length = planner(inf_pg, current, (int(tgt[0]), int(tgt[1])), res, local_margin)
        else:
            _, length = planner(inf_pg, current, (int(tgt[0]), int(tgt[1])), res)
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
    goal: tuple[int, int] | None = None,
    goal_bias: float = 0.5,
    local_margin: int | None = None,
) -> tuple[int, int] | None:
    """Softmax-temperature frontier selection with optional goal-direction bias.

    Picks a target (the known-free cell adjacent to a frontier) via softmax
    over Euclidean distance, optionally boosted by cosine similarity toward goal.

    Args:
        goal:      Grid (r, c) of goal cell. If provided, adds goal_bias * cos_sim bonus.
        goal_bias: Additive weight for the direction bonus (0 = disabled).
    """
    if not frontiers:
        return None

    f_arr = np.array(frontiers)
    cr, cc = current
    eucl_dists = np.hypot(f_arr[:, 0] - cr, f_arr[:, 1] - cc)
    order = np.argsort(eucl_dists)
    top_k = min(top_n, len(frontiers))

    targets = []
    for idx in order[:top_k]:
        fr, fc = frontiers[idx]
        tgt = frontier_free_neighbors[idx] if frontier_free_neighbors is not None else (fr, fc)
        if tgt == (cr, cc):
            continue
        targets.append(tgt)

    if not targets:
        return _fallback_select(order, frontiers, frontier_free_neighbors, inf_pg,
                                current, res, local_margin)

    tgt_arr = np.array(targets)
    eucl_to_targets = np.hypot(tgt_arr[:, 0] - cr, tgt_arr[:, 1] - cc)
    logits = -eucl_to_targets / max(temperature, 1e-6)

    if goal is not None:
        goal_r, goal_c = goal
        dot = ((tgt_arr[:, 0] - cr) * (goal_r - cr) +
               (tgt_arr[:, 1] - cc) * (goal_c - cc))
        tgt_to_goal = np.maximum(np.hypot(goal_r - tgt_arr[:, 0], goal_c - tgt_arr[:, 1]), 1e-6)
        direction_score = dot / tgt_to_goal
        logits += goal_bias * direction_score

    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    chosen_idx = rng.choice(len(targets), p=probs)

    chosen_target = (int(targets[chosen_idx][0]), int(targets[chosen_idx][1]))
    if local_margin is not None:
        _, length = astar_local(inf_pg, current, chosen_target, res, local_margin)
    else:
        _, length = astar(inf_pg, current, chosen_target, res)
    if length >= float("inf"):
        return _fallback_select(order, frontiers, frontier_free_neighbors, inf_pg,
                                current, res, local_margin)

    return chosen_target


# ============================================================================
# Topometric Graph
# ============================================================================
def build_topometric_subgraph(
    poses: list[tuple[int, int, float]],
    res: float = GRID_RES_M,
    trans_thresh: float = TRANS_THRESH_M,
    rot_thresh: float = ROT_THRESH_RAD,
    base_grid: np.ndarray | None = None,
    force_end_node: bool = True,
    goal: tuple[int, int] | None = None,
) -> nx.Graph:
    G = nx.Graph()
    if not poses:
        return G
    r0, c0, y0 = poses[0]
    G.add_node(0, x=c0 * res, y=r0 * res, yaw=y0)
    G.graph["start_node"] = 0
    node_idx = 0
    prev = (r0, c0, y0)
    inf_grid = base_grid

    for i in range(1, len(poses)):
        r, c, yaw = poses[i]
        pr, pc, py = prev

        if inf_grid is not None:
            if not _line_free(pr, pc, r, c, base_grid):
                continue

            trans = res * np.hypot(r - pr, c - pc)
            trigger = trans > trans_thresh

            if not trigger and i + 1 < len(poses):
                nr, nc, _ = poses[i + 1]
                if not _line_free(pr, pc, nr, nc, base_grid):
                    trigger = True
        else:
            trans = res * np.hypot(r - pr, c - pc)
            rot = abs(np.arctan2(np.sin(yaw - py), np.cos(yaw - py)))
            trigger = trans > trans_thresh or rot > rot_thresh

        if trigger:
            node_idx += 1
            G.add_node(node_idx, x=c * res, y=r * res, yaw=yaw)
            if inf_grid is not None:
                _, dist_m = astar(inf_grid, (pr, pc), (r, c), res)
                if dist_m >= float("inf"):
                    dist_m = res * np.hypot(r - pr, c - pc)
            else:
                dist_m = res * np.hypot(r - pr, c - pc)
            G.add_edge(node_idx - 1, node_idx, weight=dist_m,
                       rel_pose=(c - pc, r - pr, yaw - py))
            prev = (r, c, yaw)
    last_r, last_c, last_yaw = poses[-1]
    prev_r, prev_c, prev_yaw = prev
    reached_goal = goal is None or (last_r, last_c) == goal
    if force_end_node and reached_goal and (last_r, last_c) != (prev_r, prev_c):
        node_idx += 1
        G.add_node(node_idx, x=last_c * res, y=last_r * res, yaw=last_yaw)
        if inf_grid is not None:
            _, dist_m = astar(inf_grid, (prev_r, prev_c), (last_r, last_c), res)
            if dist_m >= float("inf"):
                dist_m = res * np.hypot(last_r - prev_r, last_c - prev_c)
        else:
            dist_m = res * np.hypot(last_r - prev_r, last_c - prev_c)
        G.add_edge(node_idx - 1, node_idx, weight=dist_m,
                   rel_pose=(last_c - prev_c, last_r - prev_r, last_yaw - prev_yaw))
    if reached_goal:
        G.graph["goal_node"] = node_idx
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


def _candidate_topo_edge(
    xi: float,
    yi: float,
    xj: float,
    yj: float,
    base_grid: np.ndarray,
    res: float,
    cross_dist: float,
) -> tuple[float, tuple[float, float, float]] | None:
    if np.hypot(xi - xj, yi - yj) >= cross_dist:
        return None
    ri, ci = round(yi / res), round(xi / res)
    rj, cj = round(yj / res), round(xj / res)
    if not _line_free(ri, ci, rj, cj, base_grid):
        return None
    _, path_len = astar(base_grid, (ri, ci), (rj, cj), res)
    if path_len >= float("inf"):
        return None
    return path_len, (xj - xi, yj - yi, 0.0)


def add_intra_session_loop_edges(
    G: nx.Graph,
    base_grid: np.ndarray,
    res: float = GRID_RES_M,
    cross_dist: float = CROSS_DIST_M,
) -> None:
    nodes = [(n, d["x"], d["y"]) for n, d in G.nodes(data=True)]
    for i, (ni, xi, yi) in enumerate(nodes):
        for nj, xj, yj in nodes[i + 1:]:
            if G.has_edge(ni, nj) or abs(ni - nj) == 1:
                continue
            edge = _candidate_topo_edge(xi, yi, xj, yj, base_grid, res, cross_dist)
            if edge is None:
                continue
            path_len, rel_pose = edge
            G.add_edge(ni, nj, weight=path_len, rel_pose=rel_pose)


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
        if "start_node" in G.graph:
            merged.graph.setdefault("start_nodes", []).append(G.graph["start_node"] + offset)
        if "goal_node" in G.graph:
            merged.graph.setdefault("goal_nodes", []).append(G.graph["goal_node"] + offset)
        subgraph_node_sets.append({n + offset for n in G.nodes})
        offset += G.number_of_nodes() + 1

    all_nodes = [(n, d["x"], d["y"]) for n, d in merged.nodes(data=True)]
    for si in range(len(subgraph_node_sets)):
        for sj in range(si + 1, len(subgraph_node_sets)):
            ni_list = [(n, x, y) for n, x, y in all_nodes if n in subgraph_node_sets[si]]
            nj_list = [(n, x, y) for n, x, y in all_nodes if n in subgraph_node_sets[sj]]
            for ni, xi, yi in ni_list:
                for nj, xj, yj in nj_list:
                    if merged.has_edge(ni, nj):
                        continue
                    edge = _candidate_topo_edge(xi, yi, xj, yj, base_grid, res, cross_dist)
                    if edge is None:
                        continue
                    path_len, rel_pose = edge
                    merged.add_edge(ni, nj, weight=path_len, rel_pose=rel_pose)
    return merged


def topo_path_length(
    topo_graph: nx.Graph,
    start: tuple[int, int],
    goal: tuple[int, int],
    res: float = GRID_RES_M,
) -> tuple[float, int | None, int | None]:
    """Shortest path between explicit start/goal topo nodes."""
    start_nodes = topo_graph.graph.get("start_nodes")
    goal_nodes = topo_graph.graph.get("goal_nodes")
    if start_nodes is None:
        start_node = topo_graph.graph.get("start_node")
        start_nodes = [] if start_node is None else [start_node]
    if goal_nodes is None:
        goal_node = topo_graph.graph.get("goal_node")
        goal_nodes = [] if goal_node is None else [goal_node]

    best_len = float("inf")
    best_sn, best_gn = None, None
    for sn in start_nodes:
        for gn in goal_nodes:
            if sn not in topo_graph or gn not in topo_graph:
                continue
            try:
                length = nx.shortest_path_length(topo_graph, sn, gn, weight="weight")
            except nx.NetworkXNoPath:
                continue
            if length < best_len:
                best_len = length
                best_sn, best_gn = sn, gn
    if best_sn is None or best_gn is None:
        return float("inf"), None, None
    return best_len, best_sn, best_gn


def topomap_to_npz_arrays(G: nx.Graph) -> dict[str, np.ndarray]:
    nodes_xy = np.array([[d["x"], d["y"]] for _, d in G.nodes(data=True)], dtype=np.float32)
    edges = np.array([[u, v] for u, v in G.edges()], dtype=np.int32)
    if edges.size == 0:
        edges = edges.reshape(0, 2)
    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=np.float32)

    arrays = {
        "nodes_xy": nodes_xy,
        "edges": edges,
        "edge_weights": edge_weights,
    }
    if "start_node" in G.graph:
        arrays["start_node"] = np.array([G.graph["start_node"]], dtype=np.int32)
    if "goal_node" in G.graph:
        arrays["goal_node"] = np.array([G.graph["goal_node"]], dtype=np.int32)
    if "start_nodes" in G.graph:
        arrays["start_nodes"] = np.array(G.graph["start_nodes"], dtype=np.int32)
    if "goal_nodes" in G.graph:
        arrays["goal_nodes"] = np.array(G.graph["goal_nodes"], dtype=np.int32)
    return arrays


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
    max_steps: int | None = None,
    top_n: int = FRONTIER_TOP_N,
    verbose: bool = False,
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
    recently_visited: deque[tuple[int, int]] = deque(maxlen=40)  # tabu list
    local_margin = max(int(np.ceil(fov_range_m / res)) + 20, 40)

    def _scan_position(pr: int, pc: int, pyaw: float) -> None:
        add_fov_observation(obs, pr, pc, pyaw, base_grid, fov_range_m, fov_half_rad, res)

    def _try_goal(pr: int, pc: int) -> tuple[list[tuple[int, int]] | None, np.ndarray]:
        pg = obs_to_planning_grid(obs, start, goal, for_goal_check=True)
        return astar(pg, (pr, pc), goal, res)

    _scan_position(traj[0][0], traj[0][1], traj[0][2])

    for _step in range(max_steps):
        r, c, yaw = traj[-1]

        if _step % 10 == 0:
            path_to_goal, _ = _try_goal(r, c)
            if path_to_goal is not None and len(path_to_goal) > 1:
                for nr, nc in path_to_goal[1:]:
                    ny = np.arctan2(nr - traj[-1][0], nc - traj[-1][1])
                    traj.append((nr, nc, float(ny)))
                    recently_visited.append((nr, nc))
                    if (nr, nc) not in visited_for_fov:
                        visited_for_fov.add((nr, nc))
                        _scan_position(nr, nc, float(ny))
                break

        frontiers, free_neighbors = find_frontiers(obs)
        if not frontiers:
            break

        # Filter out recently-visited targets to break oscillation loops
        filtered = [(f, fn) for f, fn in zip(frontiers, free_neighbors)
                    if fn not in recently_visited]
        if len(filtered) >= 3:
            frontiers, free_neighbors = zip(*filtered)
            frontiers = list(frontiers)
            free_neighbors = list(free_neighbors)

        pg = obs_to_planning_grid(obs, start, goal, for_goal_check=False)
        inf_pg = pg  # no inflation for frontier navigation — need to traverse into unknown

        next_f = select_frontier(frontiers, (r, c), obs, rng,
                                 frontier_temperature, top_n, inf_pg, res,
                                 frontier_free_neighbors=free_neighbors,
                                 goal=goal, goal_bias=0.5,
                                 local_margin=local_margin)
        if next_f is None:
            break

        path_to_f, _ = astar_local(inf_pg, (r, c), next_f, res, local_margin)
        if path_to_f is None or len(path_to_f) < 2:
            break

        blocked = False
        for nr, nc in path_to_f[1:]:
            if base_grid[nr, nc] == 1:  # discovered obstacle — record and re-plan
                obs[nr, nc] = 1
                blocked = True
                break
            ny = np.arctan2(nr - traj[-1][0], nc - traj[-1][1])
            traj.append((nr, nc, float(ny)))
            recently_visited.append((nr, nc))

            if (nr, nc) not in visited_for_fov:
                visited_for_fov.add((nr, nc))
                _scan_position(nr, nc, float(ny))

        if blocked:
            continue

        r_end, c_end, _ = traj[-1]
        goal_path, _ = _try_goal(r_end, c_end)
        if goal_path is not None and len(goal_path) > 1:
            for gr, gc in goal_path[1:]:
                gy = np.arctan2(gr - traj[-1][0], gc - traj[-1][1])
                traj.append((gr, gc, float(gy)))
                recently_visited.append((gr, gc))
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
COLOR_TOPO_NODE = _PALETTE[3]
COLOR_TOPO_EDGE_INTRA = _PALETTE[9]
COLOR_TOPO_PATH = _PALETTE[1]
COLOR_GT_PATH = _PALETTE[7]
COLOR_START = _PALETTE[0]
COLOR_GOAL = _PALETTE[1]
COLOR_COV_HIST = _PALETTE[0]
COLOR_COV_NEW = _PALETTE[9]
COLOR_TRAJ = "#FFFFFF"
STYLE_UNKNOWN = np.array([107, 114, 128]) / 255.0
# Pre-convert for RGBA overlay usage
_COV_HIST_RGB = tuple(float(v) for v in COLOR_COV_HIST)
_COV_NEW_RGB = tuple(float(v) for v in COLOR_COV_NEW)
ALPHA_COV_HIST = 0.35
ALPHA_COV_NEW = 0.50


def obs_to_rgb(obs: np.ndarray) -> np.ndarray:
    """Convert int8 obs grid to float32 RGB image for matplotlib.

    obs == -1  (free)     -> STYLE_FREE    (near-white)
    obs ==  0  (unknown)  -> STYLE_UNKNOWN (grey)
    obs ==  1  (obstacle) -> STYLE_OBS     (near-black)
    """
    H, W = obs.shape
    rgb = np.empty((H, W, 3), dtype=np.float32)
    rgb[:] = STYLE_UNKNOWN
    rgb[obs == -1] = STYLE_FREE
    rgb[obs == 1] = STYLE_OBS
    return rgb


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
    ax.scatter(xs, ys, color=node_color, s=node_size, zorder=5)


def _draw_path(ax, path_nodes, G, color=COLOR_TOPO_PATH, lw=3.0, res=GRID_RES_M):
    if not path_nodes or len(path_nodes) < 2:
        return
    xs, ys = [], []
    for n in path_nodes:
        xs.append(G.nodes[n]["x"] / res)
        ys.append(G.nodes[n]["y"] / res)
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=8)
    ax.scatter(xs, ys, color=color, s=60, zorder=9)


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
    _setting_font(fontsize=12, titlesize=12, legend_fontsize=10)
    K = len(subgraphs)
    ncols = 3
    nrows = (K + 1 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows),
                             facecolor=BG_COLOR)
    axes = np.atleast_1d(axes).flatten()

    for k in range(K):
        ax = axes[k]
        ax.set_facecolor(BG_COLOR)
        ax.imshow(obs_to_rgb(all_obs[k]), origin="upper", interpolation="none")

        traj = all_poses[k]
        if traj:
            tr = np.array([(p[0], p[1]) for p in traj])
            ax.plot(tr[:, 1], tr[:, 0], color=COLOR_TRAJ, linewidth=0.8,
                    alpha=0.6, zorder=3)

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

        ax.scatter(start[1], start[0], color=COLOR_START, s=80, marker="o",
                   edgecolors="white", linewidths=1.5, zorder=7)
        ax.scatter(goal[1], goal[0], color=COLOR_GOAL, s=80, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=7)
        title = (f"Session {k}  T={temperatures[k]:.1f}  "
                 f"nodes={Gk.number_of_nodes()}  "
                 f"reach={reachable}")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Merged panel
    merged_obs = np.zeros(base_grid.shape, dtype=np.int8)
    for obs_k in all_obs:
        merged_obs[obs_k == -1] = -1
        merged_obs[obs_k ==  1] =  1
    ax = axes[K]
    ax.set_facecolor(BG_COLOR)
    ax.imshow(obs_to_rgb(merged_obs), origin="upper", interpolation="none")
    if merged_topo.number_of_nodes() > 0:
        _draw_topo_graph(ax, merged_topo,
                         edge_color=COLOR_TOPO_EDGE_INTRA, res=res)
        nodes_x = [d["x"] / res for _, d in merged_topo.nodes(data=True)]
        nodes_y = [d["y"] / res for _, d in merged_topo.nodes(data=True)]
        ax.scatter(nodes_x, nodes_y, color=COLOR_TOPO_NODE, s=15, zorder=5)
        tlen, sn, gn = topo_path_length(merged_topo, start, goal, res)
        if sn is not None and gn is not None:
            try:
                sp_nodes = nx.shortest_path(merged_topo, sn, gn, weight="weight")
                _draw_path(ax, sp_nodes, merged_topo, res=res)
            except nx.NetworkXNoPath:
                tlen = float("inf")

    ax.scatter(start[1], start[0], color=COLOR_START, s=80, marker="o",
               edgecolors="white", linewidths=1.5, zorder=7)
    ax.scatter(goal[1], goal[0], color=COLOR_GOAL, s=80, marker="X",
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
    _setting_font(fontsize=12, titlesize=14, legend_fontsize=10)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    k_vals = list(range(1, len(ratios) + 1))
    finite_mask = np.array([r < float("inf") for r in ratios])
    finite_ratios = np.array(ratios)[finite_mask]
    finite_k = np.array(k_vals)[finite_mask]

    if len(finite_k) > 0:
        ax.plot(finite_k, finite_ratios, color=COLOR_TOPO_NODE,
                marker=_MARKERS[0], linestyle=_LINESTYLES[0],
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

    ax.axhline(y=1.0, color=COLOR_GOAL, linestyle=_LINESTYLES[1],
               linewidth=1.5, label="GT optimal (1.0)")
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
) -> dict[str, list[float] | float]:
    _setting_font(fontsize=12, titlesize=12, legend_fontsize=10)
    K = len(all_obs)
    total_free = int((base_grid == 0).sum())
    ncols = 3
    nrows = (K + 1 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows),
                             facecolor=BG_COLOR)
    axes = np.atleast_1d(axes).flatten()

    cov_pcts = []
    cov_m2_list = []
    new_m2_list = []
    new_pct_list = []
    cum_free_mask = np.zeros(base_grid.shape, dtype=bool)

    for k in range(K):
        ax = axes[k]
        ax.set_facecolor(BG_COLOR)
        ax.imshow(obs_to_rgb(all_obs[k]), origin="upper", interpolation="none")

        prev_mask = cum_free_mask.copy()
        new_free = (all_obs[k] == -1)
        cum_free_mask |= new_free

        new_this_session = new_free & ~prev_mask
        if new_this_session.any():
            cov_layer = np.zeros((*base_grid.shape, 4))
            cov_layer[new_this_session, :] = (*_COV_NEW_RGB, ALPHA_COV_NEW)
            ax.imshow(cov_layer, origin="upper", interpolation="none", zorder=2)

        traj = all_poses[k] if k < len(all_poses) else []
        if traj:
            tr = np.array([(p[0], p[1]) for p in traj])
            ax.plot(tr[:, 1], tr[:, 0], color=COLOR_TRAJ, linewidth=0.8, alpha=0.8, zorder=3)

        total_cov = int(cum_free_mask.sum())
        new_cov = int((new_free & ~prev_mask).sum())
        total_pct = 100.0 * total_cov / total_free if total_free > 0 else 0
        new_pct = 100.0 * new_cov / total_free if total_free > 0 else 0
        cov_m2 = total_cov * res * res
        new_m2 = new_cov * res * res
        cov_pcts.append(total_pct)
        cov_m2_list.append(cov_m2)
        new_m2_list.append(new_m2)
        new_pct_list.append(new_pct)

        ax.scatter(start[1], start[0], color=COLOR_START, s=60, marker="o",
                   edgecolors="white", linewidths=1, zorder=7)
        ax.scatter(goal[1], goal[0], color=COLOR_GOAL, s=60, marker="X",
                   edgecolors="white", linewidths=1, zorder=7)
        title = (f"k={k + 1}  new={new_m2:.0f} m²  "
                 f"total={cov_m2:.0f} m²  ({total_pct:.0f}%)")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Line plot panel
    ax = axes[K]
    ax.set_facecolor(BG_COLOR)
    ks = list(range(1, K + 1))
    ax.bar(ks, cov_m2_list, color=COLOR_COV_NEW, alpha=0.7, label="cumulative coverage")
    ax.plot(ks, cov_m2_list, color=COLOR_TOPO_NODE, marker=_MARKERS[0],
            linestyle=_LINESTYLES[0], markersize=6, linewidth=2)
    percent_symbol = r"\%" if plt.rcParams.get("text.usetex") else "%"
    for k, area_m2, pct in zip(ks, cov_m2_list, cov_pcts):
        ax.annotate(f"{area_m2:.0f} m²\n({pct:.0f}{percent_symbol})", (k, area_m2), textcoords="offset points",
                    xytext=(0, 8), ha="center", color="white", fontsize=9)
    ax.set_xlabel("Number of Sessions (k)", color="white", fontsize=10)
    ax.set_ylabel("Cumulative Coverage [m²]", color="white", fontsize=10)
    ax.set_title("Cumulative Coverage Growth", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(cov_m2_list) * 1.2 if cov_m2_list else 1)
    for spine in ax.spines.values():
        spine.set_color("white")

    for ax_extra in axes[K + 1:]:
        ax_extra.set_visible(False)

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved {output_path}")
    return {
        "res_m": res,
        "cum_m2": [round(float(v), 3) for v in cov_m2_list],
        "new_m2": [round(float(v), 3) for v in new_m2_list],
        "cum_pct": [round(float(v), 3) for v in cov_pcts],
        "new_pct": [round(float(v), 3) for v in new_pct_list],
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Frontier-Based Goal-Directed Exploration Benchmark")
    parser.add_argument("--start", type=float, nargs=2, required=True,
                        metavar=("COL_M", "ROW_M"),
                        help="Start position in world coordinates (col_axis value, row_axis value) [m]")
    parser.add_argument("--goal", type=float, nargs=2, required=True,
                        metavar=("COL_M", "ROW_M"),
                        help="Goal position in world coordinates (col_axis value, row_axis value) [m]")
    parser.add_argument("--res_m", type=float, default=GRID_RES_M,
                        help=f"Grid resolution (m/cell, default={GRID_RES_M})")
    parser.add_argument("--k", type=int, default=N_SESSIONS,
                        help=f"Number of sessions (default={N_SESSIONS})")
    parser.add_argument("--seed", type=int, default=MASTER_SEED,
                        help=f"Master random seed (default={MASTER_SEED})")
    parser.add_argument("--pcd", type=str, default=str(DEFAULT_PCD_PATH),
                        help=f"Path to PCD file (default={DEFAULT_PCD_PATH})")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default={DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--col_axis", type=int, default=0,
                        help="PCD field index for grid columns (default=0=X)")
    parser.add_argument("--row_axis", type=int, default=1,
                        help="PCD field index for grid rows (default=1=Y)")
    parser.add_argument("--height_axis", type=int, default=2,
                        help="PCD field index for height-slice filter (default=2=Z)")
    parser.add_argument("--dilate", type=int, default=None,
                        help=f"Obstacle dilation radius in pixels (default=PCD_DILATE={PCD_DILATE})")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max exploration steps per session (default=H*W*budget)")
    parser.add_argument("--temperature", type=float, default=FRONTIER_TEMP_FIXED,
                        help=f"Frontier softmax temperature for all sessions (default={FRONTIER_TEMP_FIXED})")
    args = parser.parse_args()

    if args.start is not None and args.goal is not None:
        pass  # both required, nothing to check

    K = args.k
    seed = args.seed
    pcd_path = Path(args.pcd)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    res = args.res_m
    col_axis = args.col_axis
    row_axis = args.row_axis
    height_axis = args.height_axis

    # ------------------------------------------------------------------
    print(f"=== Frontier Exploration Benchmark ===")
    print(f"  K={K}, seed={seed}, res={res}m")
    print(f"  PCD: {pcd_path}")
    print(f"  axes: col={col_axis} row={row_axis} height={height_axis}")
    print(f"  output: {output_dir}")

    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"\n--- Loading PCD ---")
    base_grid, col_range, row_range = load_pcd_grid(
        pcd_path, res, col_axis=col_axis, row_axis=row_axis, height_axis=height_axis,
        dilate=args.dilate if args.dilate is not None else PCD_DILATE)
    np.save(output_dir / "base_map.npy", base_grid)
    np.save(output_dir / "data" / "base_map.npy", base_grid)
    H, W = base_grid.shape

    # Resolve start / goal from world coordinates
    start = world_to_grid(args.start[0], args.start[1], col_range, row_range, res)
    goal  = world_to_grid(args.goal[0],  args.goal[1],  col_range, row_range, res)

    print(f"  start={(start)}, goal={(goal)}, K={K}, seed={seed}")

    # Validate start / goal
    if not (0 <= start[0] < H and 0 <= start[1] < W):
        print(f"ERROR: start {start} out of bounds (grid {H}x{W})")
        return 1
    if not (0 <= goal[0] < H and 0 <= goal[1] < W):
        print(f"ERROR: goal {goal} out of bounds (grid {H}x{W})")
        return 1
    if base_grid[start] == 1:
        print(f"ERROR: start {start} is on obstacle")
        return 1
    if base_grid[goal] == 1:
        print(f"ERROR: goal {goal} is on obstacle")
        return 1

    gt_path, gt_len = astar(base_grid, start, goal, res)
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
        "world_start": [round(float(start[1] * res + col_range[0]), 1),
                        round(float(start[0] * res + row_range[0]), 1)],
        "world_goal":  [round(float(goal[1] * res + col_range[0]), 1),
                        round(float(goal[0] * res + row_range[0]), 1)],
        "grid_shape": [H, W], "res_m": res,
        "col_range": [round(float(col_range[0]), 2), round(float(col_range[1]), 2)],
        "row_range": [round(float(row_range[0]), 2), round(float(row_range[1]), 2)],
        "col_axis": col_axis, "row_axis": row_axis, "height_axis": height_axis,
        "gt_len_m": round(float(gt_len), 3), "eucl_cells": round(float(eucl_cells), 1),
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
        initial_yaw = k * (2 * np.pi / K) + rng.uniform(-0.15, 0.15) * np.pi
        temperature = args.temperature
        temperatures.append(temperature)
        max_steps = args.max_steps if args.max_steps is not None else int(H * W * MAX_STEPS_COVERAGE_BUDGET)
        t1 = time.time()
        print(f"  Session {k}: yaw={np.degrees(initial_yaw):.0f}°, "
              f"T={temperature:.2f}, max_steps={max_steps}")

        traj, obs = frontier_explore_session(
            start, goal, base_grid, rng, initial_yaw, temperature,
            res=res, max_steps=max_steps, verbose=False)
        dt = time.time() - t1

        subgraph = build_topometric_subgraph(traj, res=res, base_grid=base_grid, goal=goal)
        add_intra_session_loop_edges(subgraph, base_grid, res=res)
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
        np.savez(output_dir / "data" / f"topomap_k{k}.npz",
                 **topomap_to_npz_arrays(subgraph))
        np.savez(output_dir / "data" / f"topomap_merged_k{k}.npz",
                 **topomap_to_npz_arrays(merged))

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
    with open(output_dir / "data" / "ratios.json", "w") as f:
        json.dump({"gt_len_m": round(gt_len, 3), "ratios": metrics["ratios"]}, f, indent=2)

    # ------------------------------------------------------------------
    print(f"\n--- Generating Figures (300 dpi) ---")
    fig1_session_exploration(
        base_grid, all_poses, all_obs, subgraphs, merged_topo_list[-1],
        start, goal, gt_path, gt_len, temperatures, res=res,
        output_path=output_dir / "fig1_session_exploration.png")
    fig2_optimality_curve(ratios, gt_len, res=res,
                          output_path=output_dir / "fig2_optimality_curve.png")
    coverage_data = fig3_reachability_coverage(
        base_grid, all_poses, all_obs, start, goal, res=res,
        output_path=output_dir / "fig3_reachability_coverage.png")
    with open(output_dir / "data" / "coverage.json", "w") as f:
        json.dump(coverage_data, f, indent=2)

    print(f"\nDone. Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
