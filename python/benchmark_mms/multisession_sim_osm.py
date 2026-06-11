#!/usr/bin/env python
"""
Multi-Session Mapping Simulation on Real OSM Occupancy Map.

Demonstrates navigation benefit of multi-session mapping compared with
single-session mapping, using OpenStreetMap building footprints rasterized
into a 1000x1000 occupancy grid.

Experiments:
  Exp1 (A1+A2) — Spatial Coverage + Path Optimality, k=1..10
  Exp2 (B4)   — Temporal Adaptability under dynamic obstacles

Usage
-----
# Benchmark:
python multisession_sim_osm.py [--map {synthetic,osm,both}]

# Map-only:
python multisession_sim_osm.py --mode map_only \\
    --lat LAT --lon LON --width_m W --length_m L

Outputs (map_only):
  output/map_only/lat_<lat>_lon_<lon>_<W>x<L>m/
    base_map.npy       occupancy grid (uint8, 0=free, 1=obstacle)
    buildings.geojson  building footprints (WGS84)
    map_viz.png        visualization with metadata panel
    metadata.json      bbox / grid / stats
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

FOV_HALF_DEG = 45.0
FOV_HALF_RAD = np.radians(FOV_HALF_DEG)
FOV_RANGE_M = 15.0
FOV_RANGE_CELLS = int(np.ceil(FOV_RANGE_M / RES))

TRANS_THRESH_M = 7.0
ROT_THRESH_RAD = np.radians(60)
CROSS_DIST_M = 10.0

INFLATE_RADIUS = 3

# Funnel map constants
CHANNEL_WAYPOINTS = {
    0: [(40, 0), (-30, 0), (40, 0), (-30, 0), (40, 0), (-30, 0), (40, 0), (-25, 0), (40, 0), (-25, 0), (40, 0)],
    1: [(50, 0), (-35, 0), (50, 0), (-35, 0), (50, 0), (-30, 0), (50, 0)],
    2: [(60, 0), (-40, 0), (60, 0), (-35, 0), (60, 0)],
    3: [(80, 0), (-40, 0), (80, 0)],
    4: [(100, 0), (-30, 0), (100, 0)],
    5: [(120, 0), (-20, 0), (80, 0)],
    6: [(80, 0), (-10, 0), (120, 0)],
    7: [],
}
CHANNEL_CENTERS = [310, 338, 366, 394, 422, 450, 478, 506]
CHANNEL_BOUNDS = [
    (300, 320), (328, 348), (356, 376), (384, 404),
    (412, 432), (440, 460), (468, 488), (496, 516),
]
N_CHANNELS = 8
DILATION_RADIUS = 9
S_PLAZA = (200, 400, 300, 700)
G_PLAZA = (600, 800, 300, 700)

# OSM download
CENTER_LAT, CENTER_LON = 22.5076, 113.9437
AREA_M = 1200

FALLBACKS = [
    (22.3364, 114.2637, "HKUST Campus, Hong Kong"),
    (42.3601, -71.0942, "MIT Campus, Cambridge MA"),
]

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _set_outdir(path):
    global OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    (path / "data").mkdir(exist_ok=True)
    OUTPUT_DIR = path

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
    from matplotlib import rc, pylab
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans'], 'size': 14})
    rc('text', usetex=False)
    pylab.rcParams.update({'axes.titlesize': 16, 'legend.fontsize': 12})


# ============================================================================
# Synthetic funnel map generator
# ============================================================================
def _bresenham_line(r0, c0, r1, c1):
    pts = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0
    if dr > dc:
        err = dr // 2
        for _ in range(dr + 1):
            pts.append((r, c))
            err -= dc
            if err < 0:
                c += sc
                err += dr
            r += sr
    else:
        err = dc // 2
        for _ in range(dc + 1):
            pts.append((r, c))
            err -= dr
            if err < 0:
                r += sr
                err += dc
            c += sc
    return pts


def _carve_channel(grid, start_row, center_col, waypoints, end_row):
    r, c = start_row, center_col
    path_cells = [(r, c)]
    for dr, dc in waypoints:
        r_end, c_end = r + dr, c + dc
        path_cells.extend(_bresenham_line(r, c, r_end, c_end))
        r, c = r_end, c_end
    path_cells.extend(_bresenham_line(r, c, end_row, c))
    for pr, pc in path_cells:
        if 0 <= pr < GRID_H and 0 <= pc < GRID_W:
            grid[pr, pc] = 0
    obst = (grid == 0) | (grid == 1)
    inflated = binary_dilation(
        obst, np.ones((DILATION_RADIUS * 2 + 1, DILATION_RADIUS * 2 + 1), bool)
    ).astype(np.uint8)
    dilated_obs = inflated & ~obst
    grid[dilated_obs > 0] = 1
    return grid


def generate_synthetic_map():
    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    sr0, sr1, sc0, sc1 = S_PLAZA
    gr0, gr1, gc0, gc1 = G_PLAZA
    grid[sr0:sr1, sc0:sc1] = 0
    grid[gr0:gr1, gc0:gc1] = 0
    for ch_idx in range(N_CHANNELS):
        wp = CHANNEL_WAYPOINTS[ch_idx]
        center = CHANNEL_CENTERS[ch_idx]
        grid = _carve_channel(grid, sr1, center, wp, gr0)
    grid[sr1:gr0, CHANNEL_BOUNDS[-1][1]:gc1] = 1
    for ch_idx in range(N_CHANNELS - 1):
        wall_c0 = CHANNEL_BOUNDS[ch_idx][1]
        wall_c1 = wall_c0 + 8
        grid[sr1:gr0, wall_c0:wall_c1] = 1
    grid[0:3, :] = 1
    grid[-3:, :] = 1
    grid[:, 0:3] = 1
    grid[:, -3:] = 1
    return grid


def inject_funnel_walls(grid):
    sr0, sr1, sc0, sc1 = S_PLAZA
    gr0, gr1, gc0, gc1 = G_PLAZA
    for ch_idx in range(N_CHANNELS):
        wp = CHANNEL_WAYPOINTS[ch_idx]
        center = CHANNEL_CENTERS[ch_idx]
        grid = _carve_channel(grid, sr1, center, wp, gr0)
    grid[sr1:gr0, CHANNEL_BOUNDS[-1][1]:gc1] = 1
    for ch_idx in range(N_CHANNELS - 1):
        wall_c0 = CHANNEL_BOUNDS[ch_idx][1]
        wall_c1 = wall_c0 + 8
        grid[sr1:gr0, wall_c0:wall_c1] = 1
    return grid


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


def rasterize_buildings(gdf_buildings, res_m: float = RES,
                        grid_w: int = GRID_W, grid_h: int = GRID_H):
    gdf_proj = gdf_buildings.to_crs(gdf_buildings.estimate_utm_crs())
    cx = gdf_proj.geometry.unary_union.centroid.x
    cy = gdf_proj.geometry.unary_union.centroid.y
    x_min = cx - (grid_w * res_m) / 2
    y_max = cy + (grid_h * res_m) / 2
    print(f"[2/5] Rasterizing {grid_w}×{grid_h} grid, res={res_m} m/cell ...")

    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    cells_x = x_min + (np.arange(grid_w) + 0.5) * res_m
    cells_y = y_max - (np.arange(grid_h) + 0.5) * res_m
    xx, yy = np.meshgrid(cells_x, cells_y)

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

    utm_bbox = {
        "xmin": float(x_min),
        "ymin": float(y_max - grid_h * res_m),
        "xmax": float(x_min + grid_w * res_m),
        "ymax": float(y_max),
    }
    print(f"\n  Done. obstacle ratio={grid.mean():.3f}")
    return grid, utm_bbox


def validate_grid(grid):
    obs_ratio = grid.mean()
    assert 0.03 < obs_ratio < 0.60, (
        f"Grid obstacle ratio {obs_ratio:.2f} out of range. Try different location."
    )
    return obs_ratio


# ============================================================================
# Map-only mode: OSM download + rasterize with user-defined bbox
# ============================================================================
def _latlon_delta(length_m, width_m, lat_rad):
    dlat = length_m / 2.0 / 111320.0
    dlon = width_m / 2.0 / (111320.0 * np.cos(lat_rad))
    return dlat, dlon


ROAD_BUFFER_M = 3.0


def download_osm_map_rect(lat, lon, width_m, length_m):
    import osmnx as ox

    def _try_download(flat, flon):
        dlat, dlon = _latlon_delta(length_m, width_m, np.radians(flat))
        gdf = ox.features_from_bbox(
            north=flat + dlat, south=flat - dlat,
            east=flon + dlon, west=flon - dlon,
            tags={"highway": True},
        )
        gdf = gdf[gdf.geometry.type.isin({"LineString", "MultiLineString"})]
        return gdf

    for flat, flon, fname in (
        [(lat, lon, f"({lat:.4f}, {lon:.4f})")] +
        [f for f in FALLBACKS]
    ):
        try:
            gdf = _try_download(flat, flon)
            if len(gdf) < 3:
                raise ValueError("Too few roads")
            print(f"[1/4] Downloaded OSM — {len(gdf)} roads, location: {fname}")
            return gdf, fname
        except Exception as e:
            print(f"  {fname} failed ({e}), trying next ...")
    raise RuntimeError("All OSM locations failed. Check internet.")


def rasterize_roads_rect(gdf_roads, width_m, length_m, res_m=RES, expand_cells=0):
    grid_w = int(round(width_m / res_m))
    grid_h = int(round(length_m / res_m))
    gdf_proj = gdf_roads.to_crs(gdf_roads.estimate_utm_crs())
    cx = gdf_proj.geometry.unary_union.centroid.x
    cy = gdf_proj.geometry.unary_union.centroid.y
    x_min = cx - (grid_w * res_m) / 2
    y_max = cy + (grid_h * res_m) / 2
    print(f"[2/4] Rasterizing {grid_w}x{grid_h} grid, res={res_m} m/cell ...")

    grid = np.ones((grid_h, grid_w), dtype=np.uint8)
    cells_x = x_min + (np.arange(grid_w) + 0.5) * res_m
    cells_y = y_max - (np.arange(grid_h) + 0.5) * res_m
    xx, yy = np.meshgrid(cells_x, cells_y)

    from shapely.vectorized import contains

    buffered = gdf_proj.geometry.buffer(ROAD_BUFFER_M)
    for idx, geom in enumerate(buffered):
        if geom is None or geom.is_empty:
            continue
        if idx % 100 == 0:
            print(f"  road {idx}/{len(buffered)} ...", end="\r")
        mask = contains(geom, xx, yy)
        grid[mask] = 0

    if expand_cells > 0:
        free_mask = (grid == 0)
        dilated = binary_dilation(free_mask, np.ones((2 * expand_cells + 1, 2 * expand_cells + 1), bool))
        grid[dilated] = 0
        print(f"  Expanded free space by {expand_cells} cells")

    grid[0:2, :] = 1
    grid[-2:, :] = 1
    grid[:, 0:2] = 1
    grid[:, -2:] = 1

    utm_bbox = {
        "xmin": float(x_min),
        "ymin": float(y_max - grid_h * res_m),
        "xmax": float(x_min + grid_w * res_m),
        "ymax": float(y_max),
    }
    free_pct = (1.0 - grid.mean()) * 100
    print(f"\n  Done. free ratio={free_pct:.1f}%, UTM bbox={utm_bbox}")
    return grid, utm_bbox


def save_osm_assets(gdf, outdir):
    gdf_wgs = gdf.to_crs("EPSG:4326")
    gdf_wgs.to_file(str(outdir / "roads.geojson"), driver="GeoJSON")
    print(f"  Saved roads.geojson ({len(gdf_wgs)} roads)")


def _validate_map_only(grid):
    obs_ratio = grid.mean()
    ok = 0.30 <= obs_ratio <= 0.99
    if not ok:
        print(f"  WARNING: obstacle ratio {obs_ratio:.3f} out of [0.30, 0.99], continuing ...")
    return {"obstacle_ratio": obs_ratio, "passed": ok}


def build_metadata(lat, lon, width_m, length_m, grid, utm_bbox, n_roads, loc_name, obs_ratio):
    return {
        "center_lat": lat,
        "center_lon": lon,
        "width_m": width_m,
        "length_m": length_m,
        "res_m_per_cell": RES,
        "grid_shape": list(grid.shape),
        "utm_bbox": utm_bbox,
        "obstacle_ratio": float(obs_ratio),
        "free_ratio": float(1.0 - obs_ratio),
        "n_roads": n_roads,
        "osm_location": loc_name,
    }


def save_map_metadata(metadata, outdir):
    import json
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Saved metadata.json")


def fig_map_only(grid, metadata, outdir):
    _init_style()
    fig, (ax_map, ax_info) = plt.subplots(1, 2, figsize=(14, 7),
                                          gridspec_kw={"width_ratios": [2.5, 1]},
                                          facecolor=BG_COLOR)
    ax_map.set_facecolor(BG_COLOR)
    rgb = np.zeros((*grid.shape, 3))
    rgb[grid == 0] = STYLE_FREE
    rgb[grid == 1] = STYLE_OBS
    ax_map.imshow(rgb, origin="upper")
    ax_map.set_title(
        f"OSM Map — ({metadata['center_lat']:.4f} N, {metadata['center_lon']:.4f} E) — "
        f"{int(metadata['width_m'])}x{int(metadata['length_m'])} m",
        color="white"
    )
    grid_h, grid_w = metadata["grid_shape"]
    w_m = metadata["width_m"]
    l_m = metadata["length_m"]
    ax_map.set_xlabel(f"Width (m) | {grid_w} cells", color="white")
    ax_map.set_ylabel(f"Length (m) | {grid_h} cells", color="white")
    ax_map.tick_params(colors="white")
    xticks = np.linspace(0, grid_w - 1, 5)
    ax_map.set_xticks(xticks)
    ax_map.set_xticklabels([f"{v:.0f}" for v in np.linspace(0, w_m, 5)])
    yticks = np.linspace(0, grid_h - 1, 5)
    ax_map.set_yticks(yticks)
    ax_map.set_yticklabels([f"{v:.0f}" for v in np.linspace(l_m, 0, 5)])

    ax_info.set_facecolor(BG_COLOR)
    ax_info.axis("off")
    info_lines = [
        f"Center GPS : {metadata['center_lat']:.4f} N, {metadata['center_lon']:.4f} E",
        f"Width      : {int(metadata['width_m'])} m",
        f"Length     : {int(metadata['length_m'])} m",
        f"Resolution : {metadata['res_m_per_cell']} m/cell",
        f"Grid shape : {grid_h} x {grid_w} cells",
        f"Free %     : {(1 - metadata['obstacle_ratio'])*100:.1f} %",
        f"Obstacle % : {metadata['obstacle_ratio']*100:.1f} %",
        f"Roads      : {metadata['n_roads']}",
        "",
        f"Location   : {metadata['osm_location']}",
    ]
    if metadata.get("utm_bbox"):
        ub = metadata["utm_bbox"]
        info_lines += [
            "",
            "UTM bbox:",
            f"  x: [{ub['xmin']:.0f}, {ub['xmax']:.0f}]",
            f"  y: [{ub['ymin']:.0f}, {ub['ymax']:.0f}]",
        ]
    for i, line in enumerate(info_lines):
        ax_info.text(0.05, 0.95 - i * 0.045, line, transform=ax_info.transAxes,
                     color="white", fontsize=11, fontfamily="monospace",
                     va="top", ha="left")
    fig.tight_layout(pad=2)
    fig.savefig(outdir / "map_viz.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print("  Saved map_viz.png")


def run_map_only(lat, lon, width_m, length_m, res_m=RES, expand_cells=0):
    t0 = time.time()
    outdir = Path(__file__).resolve().parent / "output" / "map_only" / \
             f"lat_{lat}_lon_{lon}_{int(width_m)}x{int(length_m)}m"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"OSM Map-Only (roads as free) — ({lat:.4f}, {lon:.4f}) — {int(width_m)}x{int(length_m)} m, res={res_m}m/cell, expand={expand_cells}cell")
    print("=" * 70)

    gdf, loc_name = download_osm_map_rect(lat, lon, width_m, length_m)
    grid, utm_bbox = rasterize_roads_rect(gdf, width_m, length_m, res_m, expand_cells)
    v = _validate_map_only(grid)

    np.save(outdir / "base_map.npy", grid)
    print(f"  Saved base_map.npy ({grid.shape})")

    save_osm_assets(gdf, outdir)
    metadata = build_metadata(lat, lon, width_m, length_m, grid, utm_bbox,
                              len(gdf), loc_name, v["obstacle_ratio"])
    metadata["res_m_per_cell"] = res_m
    save_map_metadata(metadata, outdir)
    fig_map_only(grid, metadata, outdir)

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Done — {total:.1f}s")
    print(f"Output dir: {outdir}")
    print(f"  base_map.npy   : {grid.shape} ({(1 - v['obstacle_ratio'])*100:.1f}% free, {v['obstacle_ratio']*100:.1f}% obstacle)")
    print(f"  roads.geojson  : {len(gdf)} roads")
    print(f"  map_viz.png    : {grid.shape[1]}x{grid.shape[0]} road occupancy")
    print(f"  metadata.json  : GPS/UTM bbox / stats")
    print("=" * 70)


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
        candidates = candidates[dists_to_best > 40]

    if len(seeds) < n_zones:
        print(f"  Only {len(seeds)} seeds found; fallback to grid subdivision ...")
        gh, gw = grid.shape
        g_rows, g_cols = 4, 3
        for r_idx in range(g_rows):
            for c_idx in range(g_cols):
                if len(seeds) >= n_zones:
                    break
                sub = grid[
                    r_idx * gh // g_rows : (r_idx + 1) * gh // g_rows,
                    c_idx * gw // g_cols : (c_idx + 1) * gw // g_cols,
                ]
                if (sub == 0).sum() == 0:
                    continue
                sd = distance_transform_edt(sub == 0)
                best_r, best_c = np.unravel_index(sd.argmax(), sd.shape)
                best_r += r_idx * gh // g_rows
                best_c += c_idx * gw // g_cols
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
        gh, gw = grid.shape
        tr = int(np.clip(seed[0] + rad_k * np.sin(angle), 5, gh - 5))
        tc = int(np.clip(seed[1] + rad_k * np.cos(angle), 5, gw - 5))
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


def make_channel_route(seed, grid, sess_id, bounds):
    rng = np.random.default_rng(sess_id)
    inf_grid = inflate(grid)
    free_cells = np.argwhere(inf_grid == 0)
    ch_free = free_cells[
        (free_cells[:, 0] >= bounds[0])
        & (free_cells[:, 1] >= bounds[2])
        & (free_cells[:, 1] <= bounds[3])
    ]
    if len(ch_free) < 10:
        ch_free = free_cells
    n_points = int(rng.integers(4, 9))
    radius = int(rng.integers(40, 120))
    wps = [tuple(seed)]
    for k in range(n_points):
        angle = 2 * np.pi * k / n_points + rng.uniform(-0.3, 0.3)
        rad_k = radius * rng.uniform(0.5, 1.0)
        tr = int(np.clip(seed[0] + rad_k * np.sin(angle), bounds[0] + 5, min(bounds[1] - 5, GRID_H - 5)))
        tc = int(np.clip(seed[1] + rad_k * np.cos(angle), bounds[2] + 3, bounds[3] - 3))
        d = np.abs(ch_free[:, 0] - tr) + np.abs(ch_free[:, 1] - tc)
        wps.append(tuple(ch_free[np.argmin(d)]))
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


SYNTHETIC_SEEDS = [
    (410, 310), (410, 338), (410, 366), (300, 500),
    (300, 400), (700, 500), (700, 350), (300, 600),
    (500, 500), (700, 650),
]


# ============================================================================
# Step 5: Camera FOV coverage
# ============================================================================
def observe_batch_fov(poses, grid, res=RES):
    obs = np.zeros_like(grid, dtype=np.int8)
    mask = np.zeros_like(grid, dtype=bool)
    H, W = grid.shape
    R = int(np.ceil(FOV_RANGE_M / res))
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
def sample_goals(base_grid, n_pairs: int = N_SESSIONS, seed: int = 42):
    """Sample n_pairs independent (start, goal) pairs from base_grid free space.

    Safety: start and goal must lie on both base_grid==0 AND inflate(base_grid)==0.
    If a sampled goal fails these checks or is unreachable, re-sample goal.
    If too many goal attempts fail, re-sample start.
    """
    inf_grid = inflate(base_grid)
    free_cells = np.argwhere(inf_grid == 0)
    if len(free_cells) < 2:
        return []

    rng = np.random.default_rng(seed)
    goals: list = []
    used_pairs = set()

    def _is_safe_cell(rc):
        r, c = rc
        return base_grid[r, c] == 0 and inf_grid[r, c] == 0

    for _ in range(max(500, 200 * n_pairs)):
        if len(goals) >= n_pairs:
            break
        si = rng.integers(0, len(free_cells))
        start = (int(free_cells[si][0]), int(free_cells[si][1]))
        if not _is_safe_cell(start):
            continue

        for _ in range(200):
            gi = rng.integers(0, len(free_cells))
            goal = (int(free_cells[gi][0]), int(free_cells[gi][1]))
            if not _is_safe_cell(goal):
                continue
            if start == goal:
                continue
            if np.hypot(start[0] - goal[0], start[1] - goal[1]) < 30:
                continue
            p, _ = astar(inf_grid, start, goal)
            if p is None:
                continue
            pair_key = (start, goal) if start < goal else (goal, start)
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)
            goals.append((start, goal))
            break

    print(f"  Sampled {len(goals)} goal pairs from inflate(base_grid) free space.")
    return goals[:n_pairs]


def _plan_on_merged_obs(merged_obs, start, goal, res_m):
    """Plan A* path on a merged-observation grid.

    Only observed-free cells (-1) are traversable.
    Unknown (0) and observed-obstacle (1) are treated as obstacles.
    start and goal are forced free before inflation.
    """
    pg = 1 - (merged_obs == -1).astype(np.uint8)
    pg[start], pg[goal] = 0, 0
    return astar(inflate(pg), start, goal, res=res_m)


def pick_best_fixed_pair(base_grid, sessions_obs, goals, res_m: float = RES, n_candidates: int = 50):
    """Find the best (start, goal) pair for fixed-pair optimality evaluation.

    Samples candidates from session 1's observed-free region (so k=1 is reachable),
    selects the pair with the largest optimality improvement:
      score = (k1_ratio - 1.0) + (k1_ratio - k10_ratio)
    Falls back to goals[0] if no good candidate found.
    """
    if len(goals) < 1:
        return goals[0] if goals else ((0, 0), (0, 0))

    rng = np.random.default_rng(42)
    inf_base = inflate(base_grid)
    session1_obs = sessions_obs[0]
    observed_free = np.argwhere(session1_obs == -1)
    if len(observed_free) < 2:
        return goals[0]

    merged_all = [session1_obs.copy() for _ in range(N_SESSIONS)]
    for k in range(1, N_SESSIONS):
        merged_all[k] = np.where(
            sessions_obs[k] != 0, sessions_obs[k], merged_all[k - 1],
        )

    best_pair = goals[0]
    best_score = -999.0

    for _ in range(n_candidates):
        si = rng.integers(0, len(observed_free))
        gi = rng.integers(0, len(observed_free))
        s = (int(observed_free[si][0]), int(observed_free[si][1]))
        g = (int(observed_free[gi][0]), int(observed_free[gi][1]))
        if s == g or np.hypot(s[0] - g[0], s[1] - g[1]) < 30:
            continue
        if base_grid[s] != 0 or base_grid[g] != 0:
            continue

        gt_path, gt_len = astar(inf_base, s, g, res=res_m)
        if gt_path is None or gt_len <= 0:
            continue

        path1, len1 = _plan_on_merged_obs(merged_all[0], s, g, res_m)
        if path1 is None:
            continue

        path10, len10 = _plan_on_merged_obs(merged_all[-1], s, g, res_m)
        k1_ratio = len1 / gt_len
        k10_ratio = len10 / gt_len if path10 is not None else float("inf")
        score = (k1_ratio - 1.0) + (k1_ratio - min(k10_ratio, k1_ratio))
        if k1_ratio > 1.0 and score > best_score:
            best_score = score
            best_pair = (s, g)

    return best_pair


# ============================================================================
# Step 9: Experiment evaluation
# ============================================================================
def run_experiments(base_grid, sessions_poses, sessions_obs, subgraphs, seeds,
                    dyn_obs_blocks, goals, res_m: float = RES):
    merged_obs_all = [sessions_obs[0].copy() for _ in range(N_SESSIONS)]
    for k in range(1, N_SESSIONS):
        merged_obs_all[k] = np.where(
            sessions_obs[k] != 0, sessions_obs[k], merged_obs_all[k - 1]
        )

    merged_topos_all = [None] * N_SESSIONS
    prev_merged = None
    for k in range(N_SESSIONS):
        if prev_merged is None:
            merged_topos_all[k] = merge_topometric_graphs(
                subgraphs[: k + 1], base_grid
            )
        else:
            merged_topos_all[k] = _add_subgraph_to_merged(
                prev_merged, subgraphs[k], k, base_grid
            )
        prev_merged = merged_topos_all[k]

    inflate_base = inflate(base_grid)
    gt_paths = []
    gt_lens = []
    n_gt_reachable = 0
    for s, g in goals:
        path, gt_len = astar(inflate_base, s, g, res=res_m)
        gt_paths.append(path)
        gt_lens.append(gt_len if path is not None else None)
        if path is not None:
            n_gt_reachable += 1
    gt_reachable_ratio = n_gt_reachable / len(goals) if len(goals) > 0 else 0.0

    goal_success_mat = np.zeros(N_SESSIONS, dtype=bool)
    est_paths = [None] * N_SESSIONS
    est_lens = [float("nan")] * N_SESSIONS
    reachable_ratios = []
    metric_ratios = []
    topo_ratios = []
    node_counts = []
    cov_area_m2 = []
    cov_free_m2 = []

    print(f"  Evaluating k=1..{N_SESSIONS} ...")
    for k in range(N_SESSIONS):
        pg = 1 - (merged_obs_all[k] == -1).astype(np.uint8)
        s, g = goals[k]
        pg_mod = pg.copy()
        pg_mod[s], pg_mod[g] = 0, 0
        inf_mod = inflate(pg_mod)
        path, est_len = astar(inf_mod, s, g, res=res_m)
        reachable = path is not None
        goal_success_mat[k] = reachable
        est_paths[k] = path
        if reachable:
            est_lens[k] = est_len

        # Cumulative reachable ratio: fraction of pairs 0..k that succeeded
        reachable_ratios.append(goal_success_mat[: k + 1].mean())

        known = (merged_obs_all[k] != 0)
        cov_area_m2.append(known.sum() * res_m * res_m)
        cov_free_m2.append(((merged_obs_all[k] == -1) & known).sum() * res_m * res_m)

        # Metric ratio: est_len / gt_len for this session's pair
        gt_len = gt_lens[k]
        if reachable and gt_len is not None and gt_len > 0 and est_len < float("inf"):
            metric_ratios.append(est_len / gt_len)
        else:
            metric_ratios.append(float("nan"))

        # Topological ratio: shortest path in merged topo graph / GT len
        topo = merged_topos_all[k]
        sx, sy = s[1] * res_m, s[0] * res_m
        gx, gy = g[1] * res_m, g[0] * res_m
        s_nodes = [n for n, d in topo.nodes(data=True)
                   if np.hypot(d["x"] - sx, d["y"] - sy) < 5.0]
        g_nodes = [n for n, d in topo.nodes(data=True)
                   if np.hypot(d["x"] - gx, d["y"] - gy) < 5.0]
        if s_nodes and g_nodes and gt_len is not None and gt_len > 0:
            try:
                topo_len = nx.shortest_path_length(topo, s_nodes[0], g_nodes[0], weight="weight")
                topo_ratios.append(topo_len / gt_len)
            except (nx.NetworkXNoPath, KeyError):
                topo_ratios.append(float("nan"))
        else:
            topo_ratios.append(float("nan"))

        node_counts.append(topo.number_of_nodes())

    # Fixed pair: read from file if saved, else auto-pick
    import json as _json
    fp_file = OUTPUT_DIR / "fixed_pair.json"
    if fp_file.exists():
        fp_data = _json.loads(fp_file.read_text())
        fixed_pair = (
            (int(fp_data["start"][0]), int(fp_data["start"][1])),
            (int(fp_data["goal"][0]), int(fp_data["goal"][1])),
        )
        print(f"  Fixed pair loaded from {fp_file}")
    else:
        fixed_pair = pick_best_fixed_pair(base_grid, sessions_obs, goals, res_m=res_m)
    fixed_start, fixed_goal = fixed_pair
    fixed_pair_paths = [None] * N_SESSIONS
    fixed_pair_lens = [float("inf")] * N_SESSIONS
    fixed_pair_success = [False] * N_SESSIONS
    fixed_pair_ratios = [float("nan")] * N_SESSIONS
    _, fixed_gt_len = astar(inflate_base, fixed_start, fixed_goal, res=res_m)
    fixed_gt_len = fixed_gt_len if fixed_gt_len is not None and fixed_gt_len < float("inf") else None
    for k in range(N_SESSIONS):
        path, length = _plan_on_merged_obs(merged_obs_all[k], fixed_start, fixed_goal, res_m)
        if path is not None:
            fixed_pair_paths[k] = path
            fixed_pair_lens[k] = length
            fixed_pair_success[k] = True
            if fixed_gt_len is not None and fixed_gt_len > 0:
                fixed_pair_ratios[k] = length / fixed_gt_len

    stale_len, updated_len, stale_collides, updated_collides = _eval_b4(
        base_grid, sessions_obs, sessions_poses, dyn_obs_blocks
    )
    new_reachable = int(goal_success_mat.sum())

    return {
        "reachable_ratios": reachable_ratios,
        "metric_ratios": metric_ratios,
        "topo_ratios": topo_ratios,
        "node_counts": node_counts,
        "cov_area_m2": cov_area_m2,
        "cov_free_m2": cov_free_m2,
        "goal_success_mat": goal_success_mat,
        "goals": goals,
        "est_paths": est_paths,
        "gt_paths": gt_paths,
        "est_lens": est_lens,
        "gt_lens": gt_lens,
        "stale_len": stale_len,
        "updated_len": updated_len,
        "stale_collides": stale_collides,
        "updated_collides": updated_collides,
        "new_reachable": new_reachable,
        "gt_reachable_ratio": gt_reachable_ratio,
        "dyn_obs_blocks": dyn_obs_blocks,
        "merged_topos": merged_topos_all,
        "fixed_pair": fixed_pair,
        "fixed_pair_paths": fixed_pair_paths,
        "fixed_pair_lens": fixed_pair_lens,
        "fixed_pair_success": fixed_pair_success,
        "fixed_pair_ratios": fixed_pair_ratios,
    }


def _add_subgraph_to_merged(prev_merged, new_subgraph, new_idx, base_grid, res=RES):
    offset = max(prev_merged.nodes()) + 1 if prev_merged.nodes() else 0
    mapping = {n: n + offset for n in new_subgraph.nodes}
    merged = nx.compose(prev_merged, nx.relabel_nodes(new_subgraph, mapping))
    new_nodes = [(n + offset, d["x"], d["y"]) for n, d in new_subgraph.nodes(data=True)]
    old_nodes_list = [
        (n, d["x"], d["y"])
        for n, d in prev_merged.nodes(data=True)
    ]
    for ni, xi, yi in new_nodes:
        for nj, xj, yj in old_nodes_list:
            d = np.hypot(xi - xj, yi - yj)
            if d < CROSS_DIST_M and not merged.has_edge(ni, nj):
                ri, ci = int(yi / res), int(xi / res)
                rj, cj = int(yj / res), int(xj / res)
                if _line_free(ri, ci, rj, cj, base_grid):
                    merged.add_edge(ni, nj, weight=d, rel_pose=(xj - xi, yj - yi, 0.0))
    return merged


def _eval_b4(base_grid, sessions_obs, sessions_poses, dyn_obs_blocks):
    obs_stale = sessions_obs[0]
    obs_updated = sessions_obs[-1]
    true_world = base_grid.copy()
    for r0, c0, r1, c1 in dyn_obs_blocks:
        true_world[r0:r1, c0:c1] = 1

    route = sessions_poses[0]
    mid = len(route) // 2
    start_b4 = (int(route[mid // 2][0]), int(route[mid // 2][1]))
    goal_b4 = (int(route[min(mid + mid // 2, len(route) - 1)][0]),
               int(route[min(mid + mid // 2, len(route) - 1)][1]))

    pg_stale = 1 - (obs_stale == -1).astype(np.uint8)
    inf_stale = inflate(pg_stale)
    stale_path, stale_len = astar(inf_stale, start_b4, goal_b4)

    stale_collides = False
    if stale_path:
        for rr, cc in stale_path:
            if true_world[rr, cc] == 1:
                stale_collides = True
                break

    pg_updated = 1 - (obs_updated == -1).astype(np.uint8)
    inf_updated = inflate(pg_updated)
    updated_path, updated_len = astar(inf_updated, start_b4, goal_b4)

    updated_collides = False
    if updated_path:
        for rr, cc in updated_path:
            if true_world[rr, cc] == 1:
                updated_collides = True
                break

    if stale_len == float("inf"):
        stale_len = 0.0
    if updated_len == float("inf"):
        updated_len = 0.0

    return stale_len, updated_len, stale_collides, updated_collides


# ============================================================================
# Step 10: Output figures
# ============================================================================
def fig0_base_map(grid, location_name, obs_ratio):
    _init_style()
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    gh, gw = grid.shape
    rgb = np.zeros((gh, gw, 3))
    rgb[grid == 0] = STYLE_FREE
    rgb[grid == 1] = STYLE_OBS
    ax.imshow(rgb, origin="upper")
    free_ratio = 1 - obs_ratio
    title = f"Real-World OSM Occupancy Map - {location_name}"
    ax.set_title(title, color="white")
    ax.text(
        0.02, 0.98,
        f"Total: {gw*gh*RES*RES:.0f} m$^2$  |  Obstacle: {obs_ratio*100:.1f}%  |  Free: {free_ratio*100:.1f}%",
        transform=ax.transAxes, color="white", va="top", fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig0_osm_base_map.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)


def fig1_session_routes(grid, sessions_poses, seeds):
    _init_style()
    gh, gw = grid.shape
    n_cols, n_rows = 5, 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), facecolor=BG_COLOR
    )
    for k in range(N_SESSIONS):
        ax = axes[k // n_cols][k % n_cols]
        ax.set_facecolor(BG_COLOR)
        rgb = np.zeros((gh, gw, 3))
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
            label="seed",
        )
        ax.scatter(c_arr[0], r_arr[0], marker="o", color="#34D399", s=60,
                   edgecolors="black", linewidths=0.5, label="start", zorder=5)
        ax.scatter(c_arr[-1], r_arr[-1], marker="X", color="#F87171", s=60,
                   edgecolors="black", linewidths=0.5, label="goal", zorder=5)
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
    ks = list(range(N_SESSIONS))  # k=0..9 = sessions 1..10
    n_cols, n_rows = 4, 3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), facecolor=BG_COLOR,
    )
    gh, gw = grid.shape
    for idx, k in enumerate(ks):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_facecolor(BG_COLOR)
        merged = merged_all[k]
        rgb = np.zeros((gh, gw, 3))
        rgb[(merged == -1)] = STYLE_FREE
        rgb[(merged == 1)] = STYLE_OBS
        rgb[(merged == 0)] = STYLE_UNK
        ax.imshow(rgb, origin="upper")
        rr = results.get("reachable_ratios", [0]*N_SESSIONS)[k] * 100
        cov = results.get("cov_free_m2", [0]*N_SESSIONS)[k]
        ax.set_title(f"After k={k+1}  |  Reach: {rr:.0f}%  Free: {cov:.0f} m$^2$",
                     color="white", fontsize=10)
        ax.axis("off")

    gt_ax = axes[-1][-1]
    gt_ax.set_facecolor(BG_COLOR)
    rgb_gt = np.zeros((gh, gw, 3))
    rgb_gt[grid == 0] = STYLE_FREE
    rgb_gt[grid == 1] = STYLE_OBS
    gt_ax.imshow(rgb_gt, origin="upper")
    gt_ax.set_title("Ground Truth", color="white", fontsize=11)
    gt_ax.axis("off")
    for idx in range(len(ks), n_rows * n_cols - 1):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_axis_off()
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
            symbol = "+" if mat[yi, xi] else "-"
            color = "#34D399" if mat[yi, xi] else "#F87171"
            ax1.text(xi, yi, symbol, ha="center", va="center", color=color, fontsize=8)

    ax2.set_facecolor(BG_COLOR)
    ks = list(range(1, N_SESSIONS + 1))
    ax2.plot(ks, [r * 100 for r in ratios], "-o", color=PALETTE[0], linewidth=2.5, markersize=8, label="Multi-session")
    gt_ratio = results.get("gt_reachable_ratio", 0.0)
    ax2.axhline(y=gt_ratio * 100, color=PALETTE[2], linestyle=":", linewidth=2,
                label=f"GT ceiling ({gt_ratio*100:.0f}%)")
    ax2.axhline(y=ratios[0] * 100, color=PALETTE[1], linestyle="--",
                label=f"Single-session ({ratios[0]*100:.0f}%)")
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


def fig4_temporal(grid, sessions_poses, sessions_obs, results, dyn_obs_blocks):
    _init_style()
    true_world = grid.copy()
    for r0, c0, r1, c1 in dyn_obs_blocks:
        true_world[r0:r1, c0:c1] = 1

    route = sessions_poses[0]
    mid = len(route) // 2
    start = (int(route[mid // 2][0]), int(route[mid // 2][1]))
    goal = (int(route[min(mid + mid // 2, len(route) - 1)][0]),
            int(route[min(mid + mid // 2, len(route) - 1)][1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG_COLOR)

    for ax, title, merged, collides, path_len in [
        (ax1, "Stale Single-Session Map", sessions_obs[0], results["stale_collides"], results["stale_len"]),
        (ax2, "Updated Multi-Session Map", sessions_obs[-1], results["updated_collides"], results["updated_len"]),
    ]:
        ax.set_facecolor(BG_COLOR)
        gh, gw = grid.shape
        rgb = np.zeros((gh, gw, 3))
        rgb[(merged == -1)] = STYLE_FREE * 0.5
        rgb[(merged == 1)] = STYLE_OBS * 0.5
        rgb[(merged == 0)] = STYLE_UNK * 0.5
        ax.imshow(rgb, origin="upper")

        for r0, c0, r1, c1 in dyn_obs_blocks:
            rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0, facecolor="red", alpha=0.35)
            ax.add_patch(rect)

        ax.scatter(start[1], start[0], marker="^", color="#34D399", s=120, zorder=5, label="Start")
        ax.scatter(goal[1], goal[0], marker="*", color="yellow", s=120, zorder=5, label="Goal")

        inf_pg = inflate((merged == -1).astype(np.uint8))
        inf_pg_mod = inf_pg.copy()
        inf_pg_mod[start], inf_pg_mod[goal] = 0, 0
        path, _ = astar(inf_pg_mod, start, goal)
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
    gt_r = results.get("gt_reachable_ratio", 0.0) * 100
    ax2.axhline(y=gt_r, color=PALETTE[4], linestyle=":", linewidth=2, label=f"GT ceiling ({gt_r:.0f}%)")
    ax2.axhline(y=results["reachable_ratios"][0] * 100, color=PALETTE[1], linestyle="--", label="Single-session")
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
        ("Reachability", f"{results['goal_success_mat'].sum()}/{N_SESSIONS} sessions  |  {results['reachable_ratios'][-1]*100:.0f}% cumul  (GT max: {results.get('gt_reachable_ratio', 0)*100:.0f}%)", PALETTE[0]),
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

    gh, gw = merged_all[0].shape
    frames = []
    for k in range(N_SESSIONS):
        _init_style()
        fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        merged = merged_all[k]
        rgb = np.zeros((gh, gw, 3))
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


def save_snapshots(data_dir, sessions_poses, sessions_obs, merged_obs, results):
    import json
    src_base = OUTPUT_DIR / "base_map.npy"
    if src_base.exists():
        np.save(data_dir / "base_map.npy", np.load(src_base))
    for k in range(N_SESSIONS):
        np.save(data_dir / f"session_{k+1:02d}_poses.npy",
                np.array(sessions_poses[k], dtype=np.float32))
        np.save(data_dir / f"session_{k+1:02d}_obs.npy",
                sessions_obs[k].astype(np.int8))
        np.save(data_dir / f"merged_obs_k{k+1:02d}.npy",
                merged_obs[k].astype(np.int8))
    merged_topos = results.get("merged_topos")
    if merged_topos is not None:
        for k in range(N_SESSIONS):
            G = merged_topos[k]
            nodes = [[int(n), {"x": float(d.get("x", 0)), "y": float(d.get("y", 0)),
                     "yaw": float(d.get("yaw", 0))}] for n, d in G.nodes(data=True)]
            edges = [[int(u), int(v), {"weight": float(G.edges[u, v].get("weight", 0))}]
                     for u, v in G.edges]
            with open(data_dir / f"topo_graph_k{k+1:02d}.json", "w") as f:
                json.dump({"nodes": nodes, "edges": edges}, f)
    metrics = {
        "reachable_ratios": results["reachable_ratios"],
        "metric_ratios": [float(r) if not np.isnan(r) else 0.0 for r in results["metric_ratios"]],
        "topo_ratios": [float(r) if not np.isnan(r) else 0.0 for r in results["topo_ratios"]],
        "node_counts": [int(n) for n in results["node_counts"]],
        "cov_area_m2": [float(v) for v in results["cov_area_m2"]],
        "cov_free_m2": [float(v) for v in results["cov_free_m2"]],
        "stale_len": float(results["stale_len"]),
        "updated_len": float(results["updated_len"]),
        "stale_collides": bool(results["stale_collides"]),
        "updated_collides": bool(results["updated_collides"]),
        "gt_reachable_ratio": float(results.get("gt_reachable_ratio", 0.0)),
        "goal_success_per_session": [bool(v) for v in results["goal_success_mat"]],
        "est_lens": [float(v) if not np.isnan(v) else 0.0 for v in results["est_lens"]],
        "gt_lens": [float(v) if v is not None else 0.0 for v in results["gt_lens"]],
        "fixed_pair": [[int(results["fixed_pair"][0][0]), int(results["fixed_pair"][0][1]),
                        int(results["fixed_pair"][1][0]), int(results["fixed_pair"][1][1])]]
                       if results.get("fixed_pair") else [],
        "fixed_pair_success": [bool(v) for v in results.get("fixed_pair_success", [])],
        "fixed_pair_lens": [float(v) for v in results.get("fixed_pair_lens", [])],
        "fixed_pair_ratios": [float(v) if not np.isnan(v) else 0.0
                              for v in results.get("fixed_pair_ratios", [])],
    }
    with open(data_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    goals_data = [[int(s[0]), int(s[1]), int(g[0]), int(g[1])] for s, g in results["goals"]]
    with open(data_dir / "goals.json", "w") as f:
        json.dump(goals_data, f)


# ============================================================================
# Figure 1 — All 10 per-session path searches (grid 5×2)
# ============================================================================
def fig1_per_session_paths(grid, results):
    """Per-session path search results, 10 panels (5x2).

    Each panel shows the merged observation map at session k, the session's
    dedicated (start, goal) pair, topological graph nodes/edges, and the A*
    shortest path found (green line).
    """
    _init_style()
    gh, gw = grid.shape
    goals = results["goals"]
    est_paths = results["est_paths"]
    gt_paths = results["gt_paths"]
    merged_topos = results["merged_topos"]
    metric_ratios = results["metric_ratios"]
    est_lens = results["est_lens"]
    gt_lens = results["gt_lens"]
    merged_all = results["merged_all"]

    n_cols, n_rows = 5, 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6), facecolor=BG_COLOR,
    )
    for k in range(N_SESSIONS):
        ax = axes[k // n_cols][k % n_cols]
        ax.set_facecolor(BG_COLOR)
        merged = merged_all[k]
        rgb = np.zeros((gh, gw, 3))
        rgb[(merged == -1)] = STYLE_FREE * 0.6
        rgb[(merged == 1)] = STYLE_OBS * 0.4
        rgb[(merged == 0)] = STYLE_UNK * 0.3
        ax.imshow(rgb, origin="upper")

        topo = merged_topos[k]
        if topo is not None:
            for u, v in topo.edges:
                xu, yu = topo.nodes[u]["x"], topo.nodes[u]["y"]
                xv, yv = topo.nodes[v]["x"], topo.nodes[v]["y"]
                rc_u = (int(yu / RES), int(xu / RES))
                rc_v = (int(yv / RES), int(xv / RES))
                ax.plot([rc_u[1], rc_v[1]], [rc_u[0], rc_v[0]],
                        "-", color="white", linewidth=0.4, alpha=0.22, zorder=2)
            node_rcs = [(int(d["y"] / RES), int(d["x"] / RES))
                        for _, d in topo.nodes(data=True)]
            if node_rcs:
                nrs, ncs = zip(*node_rcs)
                ax.scatter(ncs, nrs, marker="o", color=PALETTE[3],
                           s=50, edgecolors="white", linewidths=0.3, zorder=3)

        s, g = goals[k]
        ax.scatter(s[1], s[0], marker="o", color="#34D399", s=100,
                   edgecolors="white", linewidths=1, zorder=5)
        ax.scatter(g[1], g[0], marker="X", color="#F87171", s=100,
                   edgecolors="white", linewidths=1, zorder=5)

        path = est_paths[k]
        if path is not None:
            r_arr = [p[0] for p in path]
            c_arr = [p[1] for p in path]
            ax.plot(c_arr, r_arr, "-", color="#34D399", linewidth=2.5, zorder=4)

        reachable = "YES" if path is not None else "NO"
        mr_str = f"{metric_ratios[k]:.2f}" if not np.isnan(metric_ratios[k]) else "N/A"
        est_str = f"{est_lens[k]:.1f}m" if path is not None else "N/A"
        gt_str = f"{gt_lens[k]:.1f}m" if gt_lens[k] is not None else "N/A"
        ax.set_title(
            f"k={k+1}  reach={reachable}  est={est_str}  GT={gt_str}  ratio={mr_str}",
            color="white", fontsize=10,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_per_session_paths.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)


# ============================================================================
# Figure 2 — Fixed pair traversal across incremental merge (planning grid bg)
# ============================================================================
def fig2_fixed_pair_traversal(grid, results):
    """Fixed pair across incremental merged maps (5x2), using the planning
    grid as background to show which regions are actually traversable.

    Background: 0 = traversable (light), 1 = obstacle (dark).
    Start/goal cells are forced traversable even in unknown territory.
    """
    _init_style()
    gh, gw = grid.shape
    fixed_start, fixed_goal = results["fixed_pair"]
    merged_all = results["merged_all"]
    merged_topos = results["merged_topos"]
    fixed_pair_paths = results["fixed_pair_paths"]
    fixed_pair_lens = results["fixed_pair_lens"]
    fixed_pair_success = results["fixed_pair_success"]
    fixed_pair_ratios = results["fixed_pair_ratios"]
    gt_lens = results["gt_lens"]
    gt_paths = results["gt_paths"]
    gt_path_ref = gt_paths[0]

    n_cols, n_rows = 5, 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6), facecolor=BG_COLOR,
    )
    for k in range(N_SESSIONS):
        ax = axes[k // n_cols][k % n_cols]
        ax.set_facecolor(BG_COLOR)

        # Planning grid: 0=traversable, 1=obstacle (like the planner sees it)
        merged = merged_all[k]
        pg = 1 - (merged == -1).astype(np.uint8)
        pg[fixed_start], pg[fixed_goal] = 0, 0
        rgb = np.zeros((gh, gw, 3))
        rgb[(pg == 0)] = STYLE_FREE * 0.5
        rgb[(pg == 1)] = STYLE_OBS * 0.6
        ax.imshow(rgb, origin="upper")

        topo = merged_topos[k]
        if topo is not None:
            for u, v in topo.edges:
                xu, yu = topo.nodes[u]["x"], topo.nodes[u]["y"]
                xv, yv = topo.nodes[v]["x"], topo.nodes[v]["y"]
                rc_u = (int(yu / RES), int(xu / RES))
                rc_v = (int(yv / RES), int(xv / RES))
                ax.plot([rc_u[1], rc_v[1]], [rc_u[0], rc_v[0]],
                        "-", color="white", linewidth=0.4, alpha=0.18, zorder=2)
            node_rcs = [(int(d["y"] / RES), int(d["x"] / RES))
                        for _, d in topo.nodes(data=True)]
            if node_rcs:
                nrs, ncs = zip(*node_rcs)
                ax.scatter(ncs, nrs, marker="o", color=PALETTE[3],
                           s=45, edgecolors="white", linewidths=0.3, zorder=3)

        ax.scatter(fixed_start[1], fixed_start[0], marker="o", color="#34D399",
                   s=100, edgecolors="white", linewidths=1, zorder=5)
        ax.scatter(fixed_goal[1], fixed_goal[0], marker="X", color="#F87171",
                   s=100, edgecolors="white", linewidths=1, zorder=5)

        path = fixed_pair_paths[k]
        if path is not None:
            r_arr = [p[0] for p in path]
            c_arr = [p[1] for p in path]
            ax.plot(c_arr, r_arr, "-", color="#34D399", linewidth=2.5, zorder=4)

        if gt_path_ref is not None:
            r_gt = [p[0] for p in gt_path_ref]
            c_gt = [p[1] for p in gt_path_ref]
            ax.plot(c_gt, r_gt, "--", color="#F59E0B", linewidth=1.0, alpha=0.5, zorder=2)

        reachable = "YES" if fixed_pair_success[k] else "NO"
        mr_str = f"{fixed_pair_ratios[k]:.2f}" if not np.isnan(fixed_pair_ratios[k]) else "N/A"
        len_str = f"{fixed_pair_lens[k]:.1f}m" if fixed_pair_success[k] else "N/A"
        gt_str = f"{gt_lens[0]:.1f}m" if gt_lens[0] is not None else "N/A"
        ax.set_title(
            f"k={k+1}  reach={reachable}  est={len_str}  GT={gt_str}  ratio={mr_str}",
            color="white", fontsize=10,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_fixed_pair_traversal.png", dpi=150, facecolor=BG_COLOR)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main(map_mode: str = "osm", lat: float = CENTER_LAT, lon: float = CENTER_LON,
         area_m: int = AREA_M, res_m: float = RES):
    t0 = time.time()
    modes = [map_mode] if map_mode != "both" else ["synthetic", "osm"]

    for mode in modes:
        _dir = "real" if mode == "osm" else mode
        _set_outdir(Path(__file__).resolve().parent / "output" / _dir)
        print("=" * 70)
        print(f"Multi-Session Simulation — {mode.upper()} MAP — 10 Sessions + A* + FOV")
        print("=" * 70)

        if mode == "synthetic":
            base_grid = generate_synthetic_map()
            loc_name = "Synthetic Funnel Map (8 channels)"
            obs_ratio = base_grid.mean()
            seeds = list(SYNTHETIC_SEEDS)
            np.save(OUTPUT_DIR / "base_map.npy", base_grid)
            print("  Synthetic funnel map generated.")
        else:
            grid_side = int(area_m / res_m)
            gdf, loc_name = download_osm_map(lat, lon, area_m // 2)
            base_grid, _ = rasterize_buildings(gdf, res_m=res_m,
                                               grid_w=grid_side, grid_h=grid_side)
            obs_ratio = validate_grid(base_grid)
            np.save(OUTPUT_DIR / "base_map.npy", base_grid)
            print(f"  OSM rasterized. obstacle_ratio={obs_ratio:.3f}, shape={base_grid.shape}")
            seeds = find_zones(base_grid, N_SESSIONS)
            print(f"  Zones: {len(seeds)} seeds found.")

        sessions_poses = []
        for i in range(N_SESSIONS):
            if mode == "synthetic" and i < 3:
                ch = CHANNEL_BOUNDS[i]
                bounds = (S_PLAZA[0], G_PLAZA[1], ch[0], ch[1])
                poses = make_channel_route(seeds[i], base_grid, sess_id=1000 + i, bounds=bounds)
            else:
                poses = make_session_route(seeds[i], base_grid, sess_id=1000 + i)
            sessions_poses.append(poses)
            rt = "channel" if (mode == "synthetic" and i < 3) else "random"
            print(f"  Session {i+1}: {len(poses)} poses ({rt}), seed=({seeds[i][0]},{seeds[i][1]})")

        sessions_obs = [observe_batch_fov(poses, base_grid, res=res_m) for poses in sessions_poses]
        merged_obs = [sessions_obs[0].copy() for _ in range(N_SESSIONS)]
        for k in range(1, N_SESSIONS):
            merged_obs[k] = np.where(sessions_obs[k] != 0, sessions_obs[k], merged_obs[k - 1])

        subgraphs = [build_topometric_subgraph(poses) for poses in sessions_poses]
        for i, g in enumerate(subgraphs):
            print(f"  Topometric subgraph {i+1}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

        dyn_obs_blocks = _place_obstacles(base_grid, sessions_poses, seeds)
        print(f"  Dynamic obstacles: {len(dyn_obs_blocks)} placed.")

        goals = sample_goals(base_grid, n_pairs=N_SESSIONS, seed=42)
        print(f"  Goal pairs: {len(goals)}")

        print("\n[3/5] Running experiments A1+A2+B4 ...")
        results = run_experiments(
            base_grid, sessions_poses, sessions_obs, subgraphs, seeds,
            dyn_obs_blocks, goals, res_m=res_m,
        )
        results["merged_all"] = merged_obs
        _print_experiment_summary(results, loc_name, goals)

        print("\n[4/5] Generating output figures ...")
        fig0_base_map(base_grid, loc_name, obs_ratio);                  print("  fig0 (base map)")
        fig1_per_session_paths(base_grid, results);                     print("  fig1 (per-session paths)")
        fig2_fixed_pair_traversal(base_grid, results);                  print("  fig2 (fixed pair traversal)")
        fig4_temporal(base_grid, sessions_poses, sessions_obs, results, dyn_obs_blocks); print("  fig4_temporal")
        fig6_summary(results, loc_name, obs_ratio);                      print("  fig6_summary")
        print("  Generating map_growth.gif ...")
        create_gif(results["merged_all"], sessions_poses, results);     print("  map_growth.gif")

        save_snapshots(OUTPUT_DIR / "data", sessions_poses, sessions_obs,
                       merged_obs, results)
        print(f"  Snapshots saved to {OUTPUT_DIR / 'data'}")

    total_time = time.time() - t0
    print(f"\nTotal elapsed: {total_time:.1f}s ({total_time/60:.1f} min)")


def _place_obstacles(base_grid, sessions_poses, seeds):
    dyn_obs_blocks = []
    route_s1_cells = [(p[0], p[1]) for p in sessions_poses[0]]
    obs_candidates = []
    for off in range(-60, 61, 20):
        obs = place_dynamic_obstacle(route_s1_cells, base_grid, block_h=50, block_w=100, offset=off)
        test_grid = base_grid.copy()
        test_grid[obs[0]:obs[2], obs[1]:obs[3]] = 1
        if _check_connectivity(test_grid, seeds):
            obs_candidates.append(obs)
        if len(obs_candidates) >= 2:
            break
    return obs_candidates


def _print_experiment_summary(results, loc_name, goals):
    reachable_flags = ["YES" if v else "no" for v in results["goal_success_mat"]]
    print(f"  A1 reachable: {reachable_flags}")
    print(f"  A1 cumulative: {[f'{r*100:.0f}%' for r in results['reachable_ratios']]}")
    print("\n[5/5] Quantitative Summary")
    print("=" * 80)
    print(f"MAP SOURCE: {loc_name}")
    print(f"  res={RES} m/cell (default)")
    print(f"SESSIONS: {N_SESSIONS} | GOALS: {len(goals)} (1 per session) | PLANNER: A* | FOV: {FOV_HALF_DEG*2}deg/{FOV_RANGE_M}m")
    print("=" * 80)
    hdr = f"{'k':<4} {'Config':<15} {'Cov(m2)':<10} {'Free(m2)':<10} {'Free%':<8} {'Reach':<8} {'Cumul%':<8} {'MetricR':<9} {'TopoR':<9} {'Nodes':<7}"
    print(hdr); print("-" * 80)
    for k in range(N_SESSIONS):
        cfg = f"+S{k+1}" if k > 0 else "S1 [single]"
        mr = results["metric_ratios"][k]; tr = results["topo_ratios"][k]
        rs = "YES" if results["goal_success_mat"][k] else "no"
        print(f"{k+1:<4} {cfg:<15} {results['cov_area_m2'][k]:<10.0f} {results['cov_free_m2'][k]:<10.0f} "
              f"{results['cov_free_m2'][k]/max(results['cov_area_m2'][k],1)*100:<8.1f} {rs:<8} "
              f"{results['reachable_ratios'][k]*100:<8.0f} "
              f"{f'{mr:.2f}' if not np.isnan(mr) else 'N/A':<9} {f'{tr:.2f}' if not np.isnan(tr) else 'N/A':<9} "
              f"{results['node_counts'][k]:<7}")
    print("-" * 80)
    c = "COLLISION" if results['stale_collides'] else "SAFE"
    u = "COLLISION" if results['updated_collides'] else "SAFE"
    print(f"B4: path_before={results['stale_len']:.1f}m, path_after={results['updated_len']:.1f}m, "
          f"new_reachable={results['new_reachable']}, stale={c}, updated={u}")
    print()

    # Optimality detail table: per-session pair + fixed pair across k
    fixed_ratios = results.get("fixed_pair_ratios", [float("nan")] * N_SESSIONS)
    fixed_success = results.get("fixed_pair_success", [False] * N_SESSIONS)
    fixed_good = [k for k in range(N_SESSIONS) if fixed_success[k]]
    if fixed_good:
        f_first = fixed_good[0]
        f_last = fixed_good[-1]
        print(f"Optimality — Fixed Pair (start={results['fixed_pair'][0]}, goal={results['fixed_pair'][1]}):")
        print(f"  traverse first k={f_first+1}")
        print(f"  k={f_first+1} ratio={fixed_ratios[f_first]:.2f}  →  k={f_last+1} ratio={fixed_ratios[f_last]:.2f}")
    print()
    opt_hdr = f"{'k':<4} {'pair-reach':<12} {'per-pair ratio':<16} {'fixed-pair ratio':<18} {'fixed-pair reach':<17}"
    print(opt_hdr)
    print("-" * 70)
    for k in range(N_SESSIONS):
        mr = results["metric_ratios"][k]
        mr_str = f"{mr:.2f}" if not np.isnan(mr) else "N/A"
        fr = fixed_ratios[k]
        fr_str = f"{fr:.2f}" if not np.isnan(fr) else "N/A"
        ps = "YES" if results["goal_success_mat"][k] else "no"
        fs = "YES" if fixed_success[k] else "no"
        print(f"{k+1:<4} {ps:<12} {mr_str:<16} {fr_str:<18} {fs:<17}")
    print("-" * 70)
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MMS Benchmark / OSM Map Generator")
    parser.add_argument("--mode", type=str, default="benchmark",
                        choices=["benchmark", "map_only"])
    parser.add_argument("--map", type=str, default="osm",
                        choices=["osm", "synthetic", "both"])
    parser.add_argument("--lat", type=float, default=None,
                        help="Center latitude for benchmark osm mode (overrides CENTER_LAT)")
    parser.add_argument("--lon", type=float, default=None,
                        help="Center longitude for benchmark osm mode (overrides CENTER_LON)")
    parser.add_argument("--area_m", type=int, default=None,
                        help="Side length in meters for square OSM download area (overrides AREA_M)")
    parser.add_argument("--width_m", type=float)
    parser.add_argument("--length_m", type=float)
    parser.add_argument("--res_m", type=float, default=None,
                        help="Grid resolution m/cell (default 0.5 benchmark, 1.0 map_only)")
    parser.add_argument("--expand_cells", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "map_only":
        if None in (args.lat, args.lon, args.width_m, args.length_m):
            sys.exit("map_only 需要 --lat --lon --width_m --length_m")
        if not (-90 <= args.lat <= 90):
            sys.exit("--lat 需 ∈ [-90, 90]")
        if not (-180 <= args.lon <= 180):
            sys.exit("--lon 需 ∈ [-180, 180]")
        if not (50 <= args.width_m <= 5000):
            sys.exit("--width_m 需 ∈ [50, 5000]")
        if not (50 <= args.length_m <= 5000):
            sys.exit("--length_m 需 ∈ [50, 5000]")
        res_m_final = args.res_m if args.res_m is not None else 1.0
        if not (0.1 <= res_m_final <= 10.0):
            sys.exit("--res_m 需 ∈ [0.1, 10.0]")
        if not (0 <= args.expand_cells <= 20):
            sys.exit("--expand_cells 需 ∈ [0, 20]")
        run_map_only(args.lat, args.lon, args.width_m, args.length_m,
                     res_m=res_m_final, expand_cells=args.expand_cells)
    else:
        bench_lat = args.lat if args.lat is not None else CENTER_LAT
        bench_lon = args.lon if args.lon is not None else CENTER_LON
        bench_area = args.area_m if args.area_m is not None else AREA_M
        bench_res = args.res_m if args.res_m is not None else RES
        main(args.map, lat=bench_lat, lon=bench_lon, area_m=bench_area, res_m=bench_res)
