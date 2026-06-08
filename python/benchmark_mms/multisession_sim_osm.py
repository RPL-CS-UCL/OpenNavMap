#!/usr/bin/env python
"""
Multi-Session Mapping Simulation on Real OSM Occupancy Map.

Demonstrates navigation benefit of multi-session mapping compared with
single-session mapping, using OpenStreetMap building footprints rasterized
into a 1000x1000 occupancy grid.

Experiments:
  Exp1 (A1+A2) — Spatial Coverage + Path Optimality, k=1..10
  Exp2 (B4)   — Temporal Adaptability under dynamic obstacles
"""

import os
import sys
import heapq
import time
from pathlib import Path
from collections import deque

import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, Wedge
from scipy.ndimage import binary_dilation, distance_transform_edt

# --- project font / color utilities ---
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.utils.utils_setting_color_font import (
    acquire_color_palette,
    acquire_marker,
    acquire_linestyle,
    setting_font,
)

# ============================================================================
# Constants
# ============================================================================
GRID_W, GRID_H = 1000, 1000
RES = 0.5  # m/cell → real world 500 × 500 m
N_SESSIONS = 10
N_GOAL_PAIRS = 20

FOV_HALF_DEG = 30.0
FOV_HALF_RAD = np.radians(FOV_HALF_DEG)
FOV_RANGE_M = 7.0
FOV_RANGE_CELLS = int(np.ceil(FOV_RANGE_M / RES))

TRANS_THRESH_M = 7.0
ROT_THRESH_RAD = np.radians(60)
CROSS_DIST_M = 10.0

INFLATE_RADIUS = 3

# OSM download
CENTER_LAT, CENTER_LON = 22.5076, 113.9437
AREA_M = 1200

FALLBACKS = [
    (22.3364, 114.2637, "HKUST Campus, Hong Kong"),
    (42.3601, -71.0942, "MIT Campus, Cambridge MA"),
]

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Step 0: plot style
# ============================================================================
PALETTE = acquire_color_palette()
MARKERS = acquire_marker()
LINESTYLES = acquire_linestyle()

STYLE_FREE = np.array([243, 244, 246]) / 255.0
STYLE_OBS = np.array([31, 41, 55]) / 255.0
STYLE_UNK = np.array([107, 114, 128]) / 255.0
BG_COLOR = "#111827"


def _init_style():
    setting_font(fontsize=16, titlesize=18, legend_fontsize=14)


# ============================================================================
# Step 1: OSM download & rasterization
# ============================================================================
def download_osm_map(lat, lon, dist):
    import osmnx as ox

    print(f"[1/5] Downloading OSM from ({lat:.4f}, {lon:.4f}), dist={dist} m ...")
    for flat, flon, fname in [(lat, lon, f"({lat:.4f}, {lon:.4f})")] + [
        (f[0], f[1], f[2]) for f in FALLBACKS
    ]:
        try:
            gdf = ox.features_from_point(
                (flat, flon), tags={"building": True}, dist=dist
            )
            if len(gdf) < 5:
                raise ValueError("Too few buildings")
            print(f"  OK — {len(gdf)} buildings, location: {fname}")
            return gdf, fname
        except Exception as e:
            print(f"  {fname} failed ({e}), trying next ...")
    raise RuntimeError("All OSM locations failed. Check internet.")


def rasterize_buildings(gdf_buildings):
    gdf_proj = gdf_buildings.to_crs(gdf_buildings.estimate_utm_crs())
    cx, cy = gdf_proj.geometry.union_all().centroid.x, gdf_proj.geometry.union_all().centroid.y
    x_min = cx - (GRID_W * RES) / 2
    y_max = cy + (GRID_H * RES) / 2
    print(f"[2/5] Rasterizing {GRID_W}×{GRID_H} grid, res={RES} m/cell ...")

    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)

    cells_x = x_min + (np.arange(GRID_W) + 0.5) * RES
    cells_y = y_max - (np.arange(GRID_H) + 0.5) * RES
    xx, yy = np.meshgrid(cells_x, cells_y)  # shape (H, W)

    from shapely.vectorized import contains

    for idx, geom in enumerate(gdf_proj.geometry):
        if geom is None or geom.is_empty:
            continue
        if idx % 100 == 0:
            print(f"  building {idx}/{len(gdf_proj)} ...", end="\r")
        mask = contains(geom, xx, yy)
        grid[mask] = 1

    grid[0:2, :] = 1
    grid[-2:, :] = 1
    grid[:, 0:2] = 1
    grid[:, -2:] = 1
    print(f"\n  Done. obstacle ratio={grid.mean():.3f}")
    return grid


def validate_grid(grid):
    obs_ratio = grid.mean()
    assert 0.03 < obs_ratio < 0.60, (
        f"Grid obstacle ratio {obs_ratio:.2f} out of range. Try different location."
    )
    return obs_ratio


# ============================================================================
# Step 2: Zone identification (10 zones)
# ============================================================================
def find_zones(grid, n_zones=N_SESSIONS):
    dist = distance_transform_edt(grid == 0)
    candidates = np.argwhere(dist > 5)
    seeds = []
    for _ in range(n_zones):
        if len(candidates) == 0:
            break
        scores = dist[candidates[:, 0], candidates[:, 1]]
        best = candidates[np.argmax(scores)]
        seeds.append(tuple(best))
        dists_to_best = np.abs(candidates[:, 0] - best[0]) + np.abs(
            candidates[:, 1] - best[1]
        )
        candidates = candidates[dists_to_best > 90]

    if len(seeds) < n_zones:
        print(f"  Only {len(seeds)} seeds found; fallback to grid subdivision ...")
        g_rows, g_cols = 4, 3
        for r_idx in range(g_rows):
            for c_idx in range(g_cols):
                if len(seeds) >= n_zones:
                    break
                sub = grid[
                    r_idx * GRID_H // g_rows : (r_idx + 1) * GRID_H // g_rows,
                    c_idx * GRID_W // g_cols : (c_idx + 1) * GRID_W // g_cols,
                ]
                if (sub == 0).sum() == 0:
                    continue
                sd = distance_transform_edt(sub == 0)
                best_r, best_c = np.unravel_index(sd.argmax(), sd.shape)
                best_r += r_idx * GRID_H // g_rows
                best_c += c_idx * GRID_W // g_cols
                candidate = (best_r, best_c)
                if grid[candidate] == 0 and candidate not in seeds:
                    seeds.append(candidate)

    print(f"  Found {len(seeds)} zone seeds.")
    return seeds[:n_zones]


# ============================================================================
# Step 3: A* planner
# ============================================================================
def inflate(grid, radius=INFLATE_RADIUS):
    struct = np.ones((2 * radius + 1, 2 * radius + 1), bool)
    return binary_dilation(grid.astype(bool), structure=struct).astype(np.uint8)


def astar(grid, start, goal, res=RES):
    if grid[start] or grid[goal]:
        return None, float("inf")
    H, W = grid.shape
    diag = res * np.sqrt(2)
    dirs = [
        (-1, 0, res),
        (1, 0, res),
        (0, -1, res),
        (0, 1, res),
        (-1, -1, diag),
        (-1, 1, diag),
        (1, -1, diag),
        (1, 1, diag),
    ]

    def h(a):
        return res * np.hypot(a[0] - goal[0], a[1] - goal[1])

    g_score = {start: 0.0}
    parent = {}
    pq = [(h(start), start)]

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
                    heapq.heappush(pq, (ng + h(nb), nb))
    return None, float("inf")


# ============================================================================
# Step 4: Session route generation (random, variable length)
# ============================================================================
def make_session_route(seed, grid, sess_id):
    rng = np.random.default_rng(sess_id)
    inf_grid = inflate(grid)
    free_cells = np.argwhere(inf_grid == 0)
    n_points = int(rng.integers(4, 9))
    radius = int(rng.integers(80, 201))

    wps = [tuple(seed)]
    for k in range(n_points):
        angle = 2 * np.pi * k / n_points + rng.uniform(-0.4, 0.4)
        rad_k = radius * rng.uniform(0.6, 1.2)
        tr = int(np.clip(seed[0] + rad_k * np.sin(angle), 5, GRID_H - 5))
        tc = int(np.clip(seed[1] + rad_k * np.cos(angle), 5, GRID_W - 5))
        d = np.abs(free_cells[:, 0] - tr) + np.abs(free_cells[:, 1] - tc)
        wps.append(tuple(free_cells[np.argmin(d)]))
    wps.append(tuple(seed))

    route = []
    for a, b in zip(wps[:-1], wps[1:]):
        seg, _ = astar(inf_grid, a, b)
        if seg is None:
            continue
        route.extend(seg if not route else seg[1:])

    poses = []
    for j, (r, c) in enumerate(route):
        nxt = route[min(j + 1, len(route) - 1)]
        yaw = np.arctan2(nxt[0] - r, nxt[1] - c)
        poses.append((r, c, float(yaw)))
    return poses


# ============================================================================
# Step 5: Camera FOV coverage
# ============================================================================
def observe_batch_fov(poses, grid, res=RES):
    obs = np.zeros_like(grid, dtype=np.int8)
    mask = np.zeros_like(grid, dtype=bool)
    H, W = grid.shape
    R = FOV_RANGE_CELLS
    dr_arr = np.arange(-R, R + 1)
    dc_arr = np.arange(-R, R + 1)
    dR, dC = np.meshgrid(dr_arr, dc_arr, indexing="ij")
    dist2 = dR.astype(float) ** 2 + dC.astype(float) ** 2
    range_mask = dist2 <= R**2

    for r, c, yaw in poses:
        angle_to_cell = np.arctan2(dR.astype(float), dC.astype(float))
        angle_diff = np.abs(
            np.arctan2(np.sin(angle_to_cell - yaw), np.cos(angle_to_cell - yaw))
        )
        fov_mask = (angle_diff <= FOV_HALF_RAD) & range_mask
        rr = np.clip(r + dR[fov_mask], 0, H - 1)
        cc = np.clip(c + dC[fov_mask], 0, W - 1)
        mask[rr, cc] = True

    obs[mask & (grid == 0)] = -1
    obs[mask & (grid == 1)] = 1
    return obs


# ============================================================================
# Step 6: Topometric subgraph
# ============================================================================
def build_topometric_subgraph(poses, res=RES):
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
        if trans > TRANS_THRESH_M or rot > ROT_THRESH_RAD:
            node_idx += 1
            G.add_node(node_idx, x=c * res, y=r * res, yaw=yaw)
            dist_m = res * np.hypot(r - pr, c - pc)
            G.add_edge(
                node_idx - 1,
                node_idx,
                weight=dist_m,
                rel_pose=(c - pc, r - pr, yaw - py),
            )
            prev = (r, c, yaw)
    return G


def _line_free(r0, c0, r1, c1, base_grid):
    pts = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    r, c = r0, c0
    if dr > dc:
        err = dr // 2
        while r != r1:
            pts.append((r, c))
            err -= dc
            if err < 0:
                c += sc
                err += dr
            r += sr
    else:
        err = dc // 2
        while c != c1:
            pts.append((r, c))
            err -= dr
            if err < 0:
                r += sr
                err += dc
            c += sc
    pts.append((r1, c1))
    H, W = base_grid.shape
    return all(
        base_grid[p] == 0
        for p in pts
        if 0 <= p[0] < H and 0 <= p[1] < W
    )


def merge_topometric_graphs(subgraphs, base_grid, res=RES):
    merged = nx.Graph()
    offset = 0
    subgraph_node_sets = []
    for G in subgraphs:
        mapping = {n: n + offset for n in G.nodes}
        merged.update(nx.relabel_nodes(G, mapping))
        subgraph_node_sets.append({n + offset for n in G.nodes})
        offset += G.number_of_nodes() + 1

    all_nodes = [(n, d["x"], d["y"]) for n, d in merged.nodes(data=True)]
    for si in range(len(subgraph_node_sets)):
        for sj in range(si + 1, len(subgraph_node_sets)):
            ni_list = [
                (n, x, y)
                for n, x, y in all_nodes
                if n in subgraph_node_sets[si]
            ]
            nj_list = [
                (n, x, y)
                for n, x, y in all_nodes
                if n in subgraph_node_sets[sj]
            ]
            for ni, xi, yi in ni_list:
                for nj, xj, yj in nj_list:
                    d = np.hypot(xi - xj, yi - yj)
                    if d < CROSS_DIST_M and not merged.has_edge(ni, nj):
                        ri, ci = int(yi / res), int(xi / res)
                        rj, cj = int(yj / res), int(xj / res)
                        if _line_free(ri, ci, rj, cj, base_grid):
                            merged.add_edge(
                                ni,
                                nj,
                                weight=d,
                                rel_pose=(xj - xi, yj - yi, 0.0),
                            )
    return merged


# ============================================================================
# Step 7: Dynamic obstacles (for B4 / Exp2)
# ============================================================================
def place_dynamic_obstacle(route_s1, base_grid, block_h=24, block_w=48, offset=60):
    mid_idx = len(route_s1) // 2 + offset
    mid_r, mid_c = route_s1[min(mid_idx, len(route_s1) - 1)]
    r0 = max(2, mid_r - block_h // 2)
    c0 = max(2, mid_c - block_w // 2)
    r1 = min(base_grid.shape[0] - 2, r0 + block_h)
    c1 = min(base_grid.shape[1] - 2, c0 + block_w)
    return (r0, c0, r1, c1)


def _check_connectivity(grid_with_obs, seeds):
    inf = inflate(grid_with_obs)
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            path, _ = astar(inf, seeds[i], seeds[j])
            if path is None:
                return False
    return True


# ============================================================================
# Step 8: Goal pair sampling
# ============================================================================
def sample_goals(base_grid, seeds, n_local=4, n_adj=8, n_far=8):
    inf_grid = inflate(base_grid)
    free = np.argwhere(inf_grid == 0)
    if len(free) == 0:
        return []

    goals = []

    def _valid_pair(s, g):
        p, _ = astar(inf_grid, s, g)
        return p is not None

    def _try_sample(candidates_s, candidates_g, n):
        pairs = []
        rng = np.random.default_rng(42 + len(goals))
        tries = 0
        while len(pairs) < n and tries < 2000:
            s = (int(candidates_s[rng.integers(0, len(candidates_s))][0]),
                 int(candidates_s[rng.integers(0, len(candidates_s))][1]))
            g = (int(candidates_g[rng.integers(0, len(candidates_g))][0]),
                 int(candidates_g[rng.integers(0, len(candidates_g))][1]))
            if s != g and _valid_pair(s, g) and (s, g) not in pairs and (g, s) not in pairs:
                pairs.append((s, g))
            tries += 1
        return pairs

    for _ in range(n_local // 2):
        zone_idx = np.random.default_rng(99 + len(goals)).integers(0, len(seeds))
        region = free[
            np.abs(free[:, 0] - seeds[zone_idx][0]) < 50
        ]
        region = region[np.abs(region[:, 1] - seeds[zone_idx][1]) < 50]
        if len(region) < 10:
            region = free
        goals.extend(_try_sample(region, region, 2))

    adj_pairs = [(i, (i + 1) % len(seeds)) for i in range(min(4, len(seeds)))]
    for zi, zj in adj_pairs:
        goals.extend(_try_sample(
            np.array([seeds[zi]]),
            np.array([seeds[zj]]),
            2,
        ))

    far_pairs = [(i, (i + 5) % len(seeds)) for i in range(min(4, len(seeds)))]
    for zi, zj in far_pairs:
        goals.extend(_try_sample(
            np.array([seeds[zi]]),
            np.array([seeds[zj]]),
            2,
        ))

    goals = goals[:N_GOAL_PAIRS]
    print(f"  Sampled {len(goals)} goal pairs (local/adj/far).")
    return goals


# ============================================================================
# Step 9: Experiment evaluation
# ============================================================================
def run_experiments(base_grid, sessions_poses, sessions_obs, subgraphs, seeds,
                    dyn_obs_blocks, goals):
    merged_obs_all = [sessions_obs[0].copy() for _ in range(N_SESSIONS)]
    for k in range(1, N_SESSIONS):
        merged_obs_all[k] = np.where(
            sessions_obs[k] != 0, sessions_obs[k], merged_obs_all[k - 1]
        )

    merged_topos_all = [None] * N_SESSIONS
    for k in range(N_SESSIONS):
        merged_topos_all[k] = merge_topometric_graphs(
            subgraphs[: k + 1], base_grid
        )

    # A1 reachable_ratio + A2 metric
    inflate_base = inflate(base_grid)
    gt_cache = {}
    for s, g in goals:
        _, gt_len = astar(inflate_base, s, g)
        gt_cache[(s, g)] = gt_len

    reachable_ratios = []
    metric_ratios = []
    topo_ratios = []
    node_counts = []
    cov_area_m2 = []
    cov_free_m2 = []
    goal_success_mat = np.zeros((N_SESSIONS, len(goals)), dtype=bool)

    for k in range(N_SESSIONS):
        pg = (merged_obs_all[k] == -1).astype(np.uint8)  # 1=free, 0=unknown/obstacle
        pg = 1 - pg  # 0=free, 1=obstacle/unknown
        inf_pg = inflate(pg)
        n_reached = 0
        for gi, (s, g) in enumerate(goals):
            path, _ = astar(inf_pg, s, g)
            if path is not None:
                goal_success_mat[k, gi] = True
                n_reached += 1
        reachable_ratios.append(n_reached / len(goals))

        # coverage
        known = (merged_obs_all[k] != 0)
        cov_area_m2.append(known.sum() * RES * RES)
        cov_free_m2.append(((merged_obs_all[k] == -1) & known).sum() * RES * RES)

        # A2 on fully-reachable subset (k=9)
        full_reachable_ids = np.where(goal_success_mat[-1])[0]
        if len(full_reachable_ids) >= 3:
            mr_vals = []
            tr_vals = []
            for gi in full_reachable_ids:
                s, g = goals[gi]
                _, est_len = astar(inf_pg, s, g)
                gt_len = gt_cache[(s, g)]
                if gt_len > 0 and est_len < float("inf"):
                    mr_vals.append(est_len / gt_len)
            metric_ratios.append(np.mean(mr_vals) if mr_vals else float("nan"))

            topo = merged_topos_all[k]
            for gi in full_reachable_ids:
                s, g = goals[gi]
                s_cell = (int(s[0] * RES), int(s[1] * RES))
                g_cell = (int(g[0] * RES), int(g[1] * RES))
                s_nodes = [
                    n for n, d in topo.nodes(data=True)
                    if np.hypot(d["x"] - s_cell[1], d["y"] - s_cell[0]) < 5.0
                ]
                g_nodes = [
                    n for n, d in topo.nodes(data=True)
                    if np.hypot(d["x"] - g_cell[1], d["y"] - g_cell[0]) < 5.0
                ]
                if s_nodes and g_nodes:
                    try:
                        topo_len = nx.shortest_path_length(
                            topo, s_nodes[0], g_nodes[0], weight="weight"
                        )
                        tr_vals.append(topo_len / gt_cache[(s, g)])
                    except (nx.NetworkXNoPath, KeyError):
                        pass
            topo_ratios.append(np.mean(tr_vals) if tr_vals else float("nan"))
        else:
            metric_ratios.append(float("nan"))
            topo_ratios.append(float("nan"))

        node_counts.append(merged_topos_all[k].number_of_nodes())

    # B4 temporal adaptability
    stale_merged = sessions_obs[0]
    pg_stale = (stale_merged == -1).astype(np.uint8)
    pg_stale = 1 - pg_stale
    inf_stale = inflate(pg_stale)

    start_b4 = seeds[0]
    goal_b4 = seeds[1]

    _, stale_len = astar(inf_stale, start_b4, goal_b4)

    true_world = base_grid.copy()
    if dyn_obs_blocks:
        for r0, c0, r1, c1 in dyn_obs_blocks:
            true_world[r0:r1, c0:c1] = 1

    stale_collides = False
    stale_path, _ = astar(inf_stale, start_b4, goal_b4)
    if stale_path:
        for (rr, cc) in stale_path:
            if true_world[rr, cc] == 1:
                stale_collides = True
                stale_len = float(
                    RES
                    * np.sum(
                        np.hypot(
                            np.diff([p[0] for p in stale_path]),
                            np.diff([p[1] for p in stale_path]),
                        )
                    )
                )
                break

    pg_updated = (merged_obs_all[-1] == -1).astype(np.uint8)
    pg_updated = 1 - pg_updated
    inf_updated = inflate(pg_updated)
    _, updated_len = astar(inf_updated, start_b4, goal_b4)

    updated_collides = False
    updated_path, _ = astar(inf_updated, start_b4, goal_b4)
    if updated_path:
        for (rr, cc) in updated_path:
            if true_world[rr, cc] == 1:
                updated_collides = True
                break

    if stale_len == float("inf"):
        stale_len = 0.0

    new_reachable = int(reachable_ratios[-1] * len(goals)) - int(reachable_ratios[0] * len(goals))
    new_reachable = max(0, new_reachable)

    return {
        "reachable_ratios": reachable_ratios,
        "metric_ratios": metric_ratios,
        "topo_ratios": topo_ratios,
        "node_counts": node_counts,
        "cov_area_m2": cov_area_m2,
        "cov_free_m2": cov_free_m2,
        "goal_success_mat": goal_success_mat,
        "goals": goals,
        "stale_len": stale_len,
        "updated_len": updated_len,
        "stale_collides": stale_collides,
        "updated_collides": updated_collides,
        "new_reachable": new_reachable,
        "dyn_obs_blocks": dyn_obs_blocks,
        "inflate_base": inflate_base,
    }


# ============================================================================
# Step 10: Output figures
# ============================================================================
def fig0_base_map(grid, location_name, obs_ratio):
    _init_style()
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    rgb = np.zeros((GRID_H, GRID_W, 3))
    rgb[grid == 0] = STYLE_FREE
    rgb[grid == 1] = STYLE_OBS
    ax.imshow(rgb, origin="upper")
    free_ratio = 1 - obs_ratio
    title = f"Real-World OSM Occupancy Map - {location_name}"
    ax.set_title(title, color="white")
    ax.text(
        0.02, 0.98,
        f"Total: {GRID_W*GRID_H*RES*RES:.0f} m$^2$  |  Obstacle: {obs_ratio*100:.1f}%  |  Free: {free_ratio*100:.1f}%",
        transform=ax.transAxes, color="white", va="top", fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig0_osm_base_map.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)


def fig1_session_routes(grid, sessions_poses, seeds):
    _init_style()
    n_cols, n_rows = 5, 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), facecolor=BG_COLOR
    )
    for k in range(N_SESSIONS):
        ax = axes[k // n_cols][k % n_cols]
        ax.set_facecolor(BG_COLOR)
        rgb = np.zeros((GRID_H, GRID_W, 3))
        rgb[grid == 0] = STYLE_FREE * 0.5
        rgb[grid == 1] = STYLE_OBS * 0.5
        ax.imshow(rgb, origin="upper")
        r_arr = [p[0] for p in sessions_poses[k]]
        c_arr = [p[1] for p in sessions_poses[k]]
        color = PALETTE[k % len(PALETTE)]
        ax.plot(c_arr, r_arr, "-", color=color, linewidth=1.5, alpha=0.9)
        ax.scatter(
            seeds[k][1], seeds[k][0],
            marker="*", color="yellow", s=80, edgecolors="black", linewidths=0.5,
        )
        ax.set_title(
            f"Session {k+1} — Zone {k+1} (len={len(sessions_poses[k])} cells)",
            color="white", fontsize=11,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_session_routes.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)


def fig2_cumulative_maps(grid, merged_all, seeds, results):
    _init_style()
    ks = [0, 2, 4, 6, 9]  # k=1,3,5,7,10
    titles_k = [f"k={k+1}" for k in ks]
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), facecolor=BG_COLOR)
    for idx, k in enumerate(ks):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_facecolor(BG_COLOR)
        merged = merged_all[k]
        rgb = np.zeros((GRID_H, GRID_W, 3))
        rgb[(merged == -1)] = STYLE_FREE
        rgb[(merged == 1)] = STYLE_OBS
        rgb[(merged == 0)] = STYLE_UNK
        ax.imshow(rgb, origin="upper")
        rr = results.get("reachable_ratios", [0]*N_SESSIONS)[k] * 100
        cov = results.get("cov_free_m2", [0]*N_SESSIONS)[k]
        total_cov = results.get("cov_area_m2", [0]*N_SESSIONS)[k]
        ax.set_title(f"After {titles_k[idx]}  |  Reach: {rr:.0f}%", color="white", fontsize=11)
        ax.text(0.02, 0.98, f"Free: {cov:.0f} m$^2$  |  Cov: {total_cov:.0f} m$^2$",
                transform=ax.transAxes, color="white", va="top", fontsize=9)
        ax.axis("off")

    gt_ax = axes[-1][-1]
    gt_ax.set_facecolor(BG_COLOR)
    rgb_gt = np.zeros((GRID_H, GRID_W, 3))
    rgb_gt[grid == 0] = STYLE_FREE
    rgb_gt[grid == 1] = STYLE_OBS
    gt_ax.imshow(rgb_gt, origin="upper")
    gt_ax.set_title("Ground Truth", color="white", fontsize=11)
    gt_ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_cumulative_maps.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)


def fig3_nav_success(results):
    _init_style()
    mat = results["goal_success_mat"]
    ratios = results["reachable_ratios"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor=BG_COLOR)

    ax1.set_facecolor(BG_COLOR)
    cmap = matplotlib.colors.ListedColormap(["#991b1b", "#065f46"])
    ax1.imshow(mat.astype(float), cmap=cmap, aspect="auto", origin="lower")
    ax1.set_yticks(range(N_SESSIONS))
    ax1.set_yticklabels([f"k={i+1}" for i in range(N_SESSIONS)], color="white")
    ax1.set_xticks(range(len(results["goals"])))
    ax1.set_xticklabels(
        [f"G{i+1}" for i in range(len(results["goals"]))],
        rotation=90, color="white", fontsize=8,
    )
    ax1.set_title("Goal Reachability Matrix (green=reachable)", color="white")
    for xi in range(mat.shape[1]):
        for yi in range(mat.shape[0]):
            symbol = "✓" if mat[yi, xi] else "✗"
            color = "#34D399" if mat[yi, xi] else "#F87171"
            ax1.text(xi, yi, symbol, ha="center", va="center", color=color, fontsize=8)

    ax2.set_facecolor(BG_COLOR)
    ks = list(range(1, N_SESSIONS + 1))
    ax2.plot(ks, [r * 100 for r in ratios], "-o", color=PALETTE[0], linewidth=2.5, markersize=8, label="Multi-session")
    ax2.axhline(y=ratios[0] * 100, color=PALETTE[1], linestyle="--", label=f"Single-session baseline ({ratios[0]*100:.0f}%)")
    ax2.set_xlabel("Cumulative sessions k", color="white")
    ax2.set_ylabel("Reachable Ratio (%)", color="white")
    ax2.set_title("Reachability Growth", color="white")
    ax2.legend(facecolor=BG_COLOR, edgecolor="white", labelcolor="white")
    ax2.tick_params(colors="white")
    ax2.set_xlim(0.5, N_SESSIONS + 0.5)
    improvement = (ratios[-1] - ratios[0]) * 100
    ax2.text(N_SESSIONS - 1, ratios[-1] * 100 + 2, f"+{improvement:.0f}%", color=PALETTE[0], fontsize=13, ha="right")
    ax2.grid(ls="--", alpha=0.3, color="white")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_nav_success.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)


def fig4_temporal(grid, sessions_poses, sessions_obs, results, seeds, dyn_obs_blocks):
    _init_style()
    true_world = grid.copy()
    for r0, c0, r1, c1 in dyn_obs_blocks:
        true_world[r0:r1, c0:c1] = 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG_COLOR)

    for ax, title, merged, collides, path_len in [
        (ax1, "Stale Single-Session Map", sessions_obs[0], results["stale_collides"], results["stale_len"]),
        (ax2, "Updated Multi-Session Map", sessions_obs[-1], results["updated_collides"], results["updated_len"]),
    ]:
        ax.set_facecolor(BG_COLOR)
        rgb = np.zeros((GRID_H, GRID_W, 3))
        rgb[(merged == -1)] = STYLE_FREE * 0.5
        rgb[(merged == 1)] = STYLE_OBS * 0.5
        rgb[(merged == 0)] = STYLE_UNK * 0.5
        ax.imshow(rgb, origin="upper")

        for r0, c0, r1, c1 in dyn_obs_blocks:
            rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0, facecolor="red", alpha=0.35)
            ax.add_patch(rect)

        start, goal = seeds[0], seeds[1]
        ax.scatter(start[1], start[0], marker="^", color="#34D399", s=120, zorder=5, label="Start")
        ax.scatter(goal[1], goal[0], marker="*", color="yellow", s=120, zorder=5, label="Goal")

        inf_pg = inflate((merged == -1).astype(np.uint8))
        inf_pg_blocked = inf_pg.copy()
        path, _ = astar(inf_pg_blocked, start, goal)
        if path:
            rp = [p[0] for p in path]
            cp = [p[1] for p in path]
            ax.plot(cp, rp, "-", color=PALETTE[0], linewidth=2.5, label="Planned Path")

        status = "COLLISION" if collides else "SAFE"
        color = "#F87171" if collides else "#34D399"
        ax.set_title(f"{title}  —  {status}", color="white", fontsize=14)
        ax.text(0.02, 0.98, f"Path length: {path_len:.1f} m", transform=ax.transAxes,
                color=color, va="top", fontsize=12, fontweight="bold")
        ax.legend(facecolor=BG_COLOR, edgecolor="white", labelcolor="white", loc="lower right")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_temporal_adaptability.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)


def fig5_growth_charts(results):
    _init_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG_COLOR)

    ks = np.arange(1, N_SESSIONS + 1)

    ax1.set_facecolor(BG_COLOR)
    ax1.plot(ks, np.array(results["cov_free_m2"]), "-o", color=PALETTE[0], linewidth=2.5, markersize=7)
    ax1.set_xlabel("Cumulative sessions k", color="white")
    ax1.set_ylabel("Known Free Area (m$^2$)", color="white")
    ax1.set_title("Spatial Coverage Growth", color="white")
    ax1.tick_params(colors="white")
    ax1.grid(ls="--", alpha=0.3, color="white")
    delta = results["cov_free_m2"][-1] - results["cov_free_m2"][0]
    ax1.annotate(f"+{delta:.0f} m$^2$", xy=(N_SESSIONS, results["cov_free_m2"][-1]),
                 xytext=(N_SESSIONS - 3, results["cov_free_m2"][-1] + delta * 0.1),
                 color=PALETTE[0], fontsize=13, arrowprops=dict(arrowstyle="->", color=PALETTE[0]))

    ax2.set_facecolor(BG_COLOR)
    ax2.plot(ks, [r * 100 for r in results["reachable_ratios"]], "-s", color=PALETTE[2], linewidth=2.5, markersize=7, label="Multi-session")
    ax2.axhline(y=results["reachable_ratios"][0] * 100, color=PALETTE[1], linestyle="--", label="Single-session baseline")
    ax2.set_xlabel("Cumulative sessions k", color="white")
    ax2.set_ylabel("Reachable Ratio (%)", color="white")
    ax2.set_title("Reachability Growth", color="white")
    ax2.legend(facecolor=BG_COLOR, edgecolor="white", labelcolor="white")
    ax2.tick_params(colors="white")
    ax2.grid(ls="--", alpha=0.3, color="white")

    ax3.set_facecolor(BG_COLOR)
    mr = results["metric_ratios"]
    tr = results["topo_ratios"]
    valid = ~np.isnan(mr)
    ax3.plot(ks[valid], np.array(mr)[valid], "-o", color=PALETTE[0], linewidth=2.5, markersize=7, label="metric_ratio (grid A*)")
    valid_t = ~np.isnan(tr)
    ax3.plot(ks[valid_t], np.array(tr)[valid_t], "-^", color=PALETTE[3], linewidth=2.5, markersize=7, label="topological_ratio")
    ax3.axhline(y=1.0, color="white", linestyle="--", alpha=0.5, label="GT optimal (=1.0)")
    ax3.set_xlabel("Cumulative sessions k", color="white")
    ax3.set_ylabel("Ratio (lower = better)", color="white")
    ax3.set_title("Path Optimality", color="white")
    ax3.legend(facecolor=BG_COLOR, edgecolor="white", labelcolor="white")
    ax3.tick_params(colors="white")
    ax3.grid(ls="--", alpha=0.3, color="white")
    if valid.any():
        ax3.annotate(f"{mr[0]:.2f}→{mr[-1]:.2f}",
                     xy=(N_SESSIONS, mr[-1]),
                     xytext=(N_SESSIONS - 3, mr[-1] + 0.1),
                     color=PALETTE[0], fontsize=12,
                     arrowprops=dict(arrowstyle="->", color=PALETTE[0]))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_growth_charts.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)


def fig6_summary(results, location_name, obs_ratio):
    _init_style()
    fig, (ax_top, ax_table) = plt.subplots(2, 1, figsize=(18, 14),
                                           gridspec_kw={"height_ratios": [1, 3]},
                                           facecolor=BG_COLOR)

    ax_top.set_facecolor(BG_COLOR)
    ax_top.axis("off")
    cards = [
        ("Reachability", f"{results['reachable_ratios'][0]*100:.0f}% → {results['reachable_ratios'][-1]*100:.0f}%", PALETTE[0]),
        ("Coverage (Free)", f"{results['cov_free_m2'][0]:.0f} → {results['cov_free_m2'][-1]:.0f} m$^2$", PALETTE[2]),
        ("Optimality (metric)", f"{results['metric_ratios'][0]:.2f} → {results['metric_ratios'][-1]:.2f}", PALETTE[3]),
        ("Temporal", f"{'COLLISION' if results['stale_collides'] else 'SAFE'} → {'SAFE' if not results['updated_collides'] else 'COLLISION'}", PALETTE[1]),
    ]
    for i, (label, value, color) in enumerate(cards):
        x = 0.05 + i * 0.24
        rect = FancyBboxPatch((x, 0.2), 0.20, 0.65,
                              boxstyle="round,pad=0.08", facecolor="#1f2937", edgecolor=color, linewidth=2)
        ax_top.add_patch(rect)
        ax_top.text(x + 0.10, 0.70, label,
                    ha="center", va="center", color="white", fontsize=13,
                    transform=ax_top.transAxes, fontweight="bold")
        ax_top.text(x + 0.10, 0.45, value,
                    ha="center", va="center", color=color, fontsize=16,
                    transform=ax_top.transAxes, fontweight="bold")

    ax_table.set_facecolor(BG_COLOR)
    ax_table.axis("off")
    header = ["k", "Config", "Cov(m²)", "Free(m²)", "Free%", "Reach%", "MetricR", "TopoR", "Nodes"]
    data = []
    for k in range(N_SESSIONS):
        data.append([
            str(k + 1),
            f"+S{k+1}" if k > 0 else "S1 [single]",
            f"{results['cov_area_m2'][k]:.0f}",
            f"{results['cov_free_m2'][k]:.0f}",
            f"{results['cov_free_m2'][k]/max(results['cov_area_m2'][k],1)*100:.1f}",
            f"{results['reachable_ratios'][k]*100:.0f}",
            f"{results['metric_ratios'][k]:.2f}" if not np.isnan(results['metric_ratios'][k]) else "N/A",
            f"{results['topo_ratios'][k]:.2f}" if not np.isnan(results['topo_ratios'][k]) else "N/A",
            str(results['node_counts'][k]),
        ])

    table = ax_table.table(cellText=data, colLabels=header,
                           cellLoc="center", loc="center",
                           colColours=["#1f2937"] * len(header))
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for key, cell in table.get_celld().items():
        cell.set_facecolor("#1f2937")
        cell.set_edgecolor("#374151")
        cell.set_text_props(color="white")
        if key[0] == 0:
            cell.set_text_props(color="white", fontweight="bold")
        if key[0] == 10:
            cell.set_text_props(color=PALETTE[0], fontweight="bold")

    b4_text = (f"B4 Temporal: path_before={results['stale_len']:.1f} m, path_after={results['updated_len']:.1f} m, "
               f"new_reachable={results['new_reachable']}, "
               f"stale={'COLLISION' if results['stale_collides'] else 'SAFE'}, "
               f"updated={'COLLISION' if results['updated_collides'] else 'SAFE'}")
    ax_table.text(0.5, -0.02, b4_text, transform=ax_table.transAxes, ha="center",
                  color="white", fontsize=11)

    fig.suptitle(f"Multi-Session Mapping Summary — {location_name}",
                 color="white", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "fig6_summary_table.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)


def create_gif(merged_all, sessions_poses, results):
    import imageio

    frames = []
    for k in range(N_SESSIONS):
        _init_style()
        fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        merged = merged_all[k]
        rgb = np.zeros((GRID_H, GRID_W, 3))
        rgb[(merged == -1)] = STYLE_FREE
        rgb[(merged == 1)] = STYLE_OBS
        rgb[(merged == 0)] = STYLE_UNK
        ax.imshow(rgb, origin="upper")

        if k < len(sessions_poses):
            r_arr = [p[0] for p in sessions_poses[k]]
            c_arr = [p[1] for p in sessions_poses[k]]
            ax.plot(c_arr, r_arr, "-", color=PALETTE[k % len(PALETTE)], linewidth=1.2, alpha=0.8)

        reach = results["reachable_ratios"][k] * 100
        mr = results["metric_ratios"][k]
        mr_str = f"{mr:.2f}" if not np.isnan(mr) else "N/A"
        cov = results["cov_free_m2"][k]
        ax.set_title(f"After Session {k+1} | Free: {cov:.0f} m$^2$ | Reach: {reach:.0f}% | MetricR: {mr_str}",
                     color="white", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(OUTPUT_DIR / "map_growth.gif", frames, fps=3, loop=0)


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("Multi-Session Mapping Simulation — OSM + 10 Sessions + A* + FOV")
    print("=" * 70)

    # --- Step 1: OSM download & rasterize ---
    gdf, loc_name = download_osm_map(CENTER_LAT, CENTER_LON, AREA_M // 2)
    t_dl = time.time()
    print(f"  Download done in {t_dl - t0:.1f}s")

    base_grid = rasterize_buildings(gdf)
    obs_ratio = validate_grid(base_grid)
    np.save(OUTPUT_DIR / "base_map.npy", base_grid)
    print(f"  Rasterization done. obstacle_ratio={obs_ratio:.3f}")

    # --- Step 2: 10 zones ---
    seeds = find_zones(base_grid, N_SESSIONS)
    print(f"  Zones: {len(seeds)} seeds found.")

    # --- Step 3: Session routes (random, variable length) ---
    sessions_poses = []
    for i in range(N_SESSIONS):
        poses = make_session_route(seeds[i], base_grid, sess_id=1000 + i)
        sessions_poses.append(poses)
        print(f"  Session {i+1}: {len(poses)} poses, seed=({seeds[i][0]},{seeds[i][1]})")

    # --- Step 4: FOV coverage ---
    sessions_obs = []
    for i, poses in enumerate(sessions_poses):
        obs = observe_batch_fov(poses, base_grid)
        sessions_obs.append(obs)

    merged_all = [sessions_obs[0].copy() for _ in range(N_SESSIONS)]
    for k in range(1, N_SESSIONS):
        merged_all[k] = np.where(
            sessions_obs[k] != 0, sessions_obs[k], merged_all[k - 1]
        )

    # --- Step 5: Topometric subgraphs ---
    subgraphs = [build_topometric_subgraph(poses) for poses in sessions_poses]
    for i, g in enumerate(subgraphs):
        print(f"  Topometric subgraph {i+1}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # --- Step 6: Dynamic obstacles ---
    dyn_obs_blocks = []
    route_s1_cells = [(p[0], p[1]) for p in sessions_poses[0]]
    obs1 = place_dynamic_obstacle(route_s1_cells, base_grid, block_h=24, block_w=48, offset=60)
    obs2 = place_dynamic_obstacle(route_s1_cells, base_grid, block_h=24, block_w=48, offset=-40)
    for obs in [obs1, obs2]:
        test_grid = base_grid.copy()
        test_grid[obs[0]:obs[2], obs[1]:obs[3]] = 1
        if _check_connectivity(test_grid, seeds):
            dyn_obs_blocks.append(obs)
    print(f"  Dynamic obstacles: {len(dyn_obs_blocks)} placed (non-disconnecting).")

    # --- Step 7: Goal pairs ---
    goals = sample_goals(base_grid, seeds)
    print(f"  Goal pairs: {len(goals)}")

    # --- Step 8: Run experiments ---
    print("\n[3/5] Running experiments A1+A2+B4 ...")
    results = run_experiments(
        base_grid, sessions_poses, sessions_obs, subgraphs, seeds,
        dyn_obs_blocks, goals,
    )
    results["merged_all"] = merged_all
    results["inflate_base"] = inflate(base_grid)
    print(f"  A1 reachable_ratios: {[f'{r*100:.0f}%' for r in results['reachable_ratios']]}")

    # --- Step 9: Output figures ---
    print("\n[4/5] Generating output figures ...")
    fig0_base_map(base_grid, loc_name, obs_ratio)
    print("  fig0_osm_base_map.png")
    fig1_session_routes(base_grid, sessions_poses, seeds)
    print("  fig1_session_routes.png")
    fig2_cumulative_maps(base_grid, results["merged_all"], seeds, results)
    print("  fig2_cumulative_maps.png")
    fig3_nav_success(results)
    print("  fig3_nav_success.png")
    fig4_temporal(base_grid, sessions_poses, sessions_obs, results, seeds, dyn_obs_blocks)
    print("  fig4_temporal_adaptability.png")
    fig5_growth_charts(results)
    print("  fig5_growth_charts.png")
    fig6_summary(results, loc_name, obs_ratio)
    print("  fig6_summary_table.png")
    print("  Generating map_growth.gif ...")
    create_gif(results["merged_all"], sessions_poses, results)
    print("  map_growth.gif")

    # --- Step 10: Print summary ---
    print("\n[5/5] Quantitative Summary")
    print("=" * 80)
    print(f"MAP SOURCE: OSM — {loc_name}")
    print(f"GRID: {GRID_W}×{GRID_H} cells, {RES} m/cell → {GRID_W*RES}×{GRID_H*RES} m real world")
    print(f"SESSIONS: {N_SESSIONS} | GOALS: {len(goals)} pairs | PLANNER: A* | FOV: {FOV_HALF_DEG*2}°/{FOV_RANGE_M}m")
    print(f"GT: geodesic A* on full base_map")
    print("=" * 80)
    print(f"{'k':<4} {'Config':<15} {'Cov(m²)':<10} {'Free(m²)':<10} {'Free%':<8} {'Reach%':<8} {'MetricR':<9} {'TopoR':<9} {'Nodes':<7}")
    print("-" * 80)
    for k in range(N_SESSIONS):
        cfg = f"+S{k+1}" if k > 0 else "S1 [single]"
        mr = results["metric_ratios"][k]
        tr = results["topo_ratios"][k]
        print(
            f"{k+1:<4} {cfg:<15} "
            f"{results['cov_area_m2'][k]:<10.0f} "
            f"{results['cov_free_m2'][k]:<10.0f} "
            f"{results['cov_free_m2'][k]/max(results['cov_area_m2'][k],1)*100:<8.1f} "
            f"{results['reachable_ratios'][k]*100:<8.0f} "
            f"{f'{mr:.2f}' if not np.isnan(mr) else 'N/A':<9} "
            f"{f'{tr:.2f}' if not np.isnan(tr) else 'N/A':<9} "
            f"{results['node_counts'][k]:<7}"
        )
    print("-" * 80)
    b4_status = f"stale={'COLLISION' if results['stale_collides'] else 'SAFE'}, updated={'COLLISION' if results['updated_collides'] else 'SAFE'}"
    print(f"B4 (temporal): path_before={results['stale_len']:.1f} m, path_after={results['updated_len']:.1f} m, "
          f"new_reachable={results['new_reachable']}, {b4_status}")
    print("=" * 80)

    total_time = time.time() - t0
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
