# Frontier-Based Goal-Directed Exploration + Topological Map Design

> **Created:** 2025-06-13
> **Updated:** 2025-06-14 — Final spec: PCD loading, fixed (start,goal), softmax perturbation, 3-figure visualization
> **Status:** Approved, final spec. Implementing `frontier_explore_benchmark.py`
> **Context:** Multi-Session Mapping Benchmark — octa_maze PCD demo

---

## 1. Motivation

### 1.1 Problem with Current Implementation

The current per-session simulation has a fundamental design flaw:

- **Goal pairs are sampled AFTER all sessions complete** (`sample_goals()` is called post-hoc).
- **Robot paths are goal-agnostic** — `make_session_route()` generates random closed polygon routes around a seed point, with no awareness of the goal.
- **Reachability is trivially broken** — the robot never tries to reach the goal, so `reachable=NO` for most sessions is an artifact of the design, not a meaningful experimental result.

This makes the benchmark circular: goals are sampled from the merged k=10 observation map's free space, and then we evaluate whether sessions can reach those goals. Since the robot never knew about the goals during exploration, the answer is "no unless the goal happens to be in the visited area."

### 1.2 Correct Concept

A multi-session topometric mapping benchmark should measure:

> Given a fixed (start, goal) pair, as more sessions explore and merge their topological maps, does the shortest path on the merged topometric map converge to the ground-truth optimal path?

This is directly analogous to real-world robot navigation:
1. Session 1 explores, builds a partial map
2. The partial map can be queried for a start→goal path (suboptimal due to incomplete knowledge)
3. Session 2 explores a different area, merging with session 1's map
4. The merged map now finds a shorter path for the same (start, goal) pair
5. After N sessions, the path approaches the ground-truth optimum

**Both start and goal are fixed across all sessions.** Session diversity comes from exploration strategy perturbation (different initial yaw + softmax frontier selection temperature), not from different starting positions. This enables direct comparison of path optimality as the merged topometric map grows across sessions.

---

## 2. Design Overview

### 2.1 Key Components

| Component | Description |
|-----------|-------------|
| **Frontier-based Explorer** | Per-session exploration loop: move toward frontiers, update observations, check if goal is reachable |
| **Goal Checker** | After each observation update, run A* on the known map; if start→goal is reachable, navigate to goal and terminate |
| **Topological Map Builder** | Extract keyframes from the exploration trajectory; store as NetworkX graph with spatial positions |
| **Cross-session Map Merger** | Merge topological maps from all sessions, adding cross-session edges where keyframes are close and line-of-sight is clear |
| **Topometric Path Planner** | Query shortest path on the merged topological graph for the fixed (start, goal) pair |
| **Optimality Evaluator** | Compare topometric path length vs. ground-truth A* path length on the full base map |

### 2.2 Data Flow

```
                   ┌──────────┐
                   │ fix pair │  (start, goal) given via CLI or auto-sampled
                   │  save to │  distance >= DIST_MIN cells, GT-reachable
                   │ file.json│
                   └────┬─────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Session 1        Session 2    ...  Session K
   ┌─────────┐    ┌─────────┐       ┌─────────┐
   │Frontier │    │Frontier │       │Frontier │
   │Explore  │    │Explore  │       │Explore  │
   │+Goal Chk│    │+Goal Chk│       │+Goal Chk│
   └────┬────┘    └────┬────┘       └────┬────┘
        │               │               │
        ▼               ▼               ▼
    topo_map_1      topo_map_2       topo_map_K
        │               │               │
        └───────────────┼───────────────┘
                        │  merge_topometric_graphs(sessions[:k+1])
                        ▼
                 merged_topo_map[k]
                        │
                        │  nx.shortest_path(merged_topo_map, start, goal)
                        ▼
                 topo_path_len[k]
                        │
                        │  ratio[k] = topo_path_len[k] / GT_len
                        ▼
                 optimality_curve (k=1..K approaches 1.0)
```

---

## 3. Frontier-Based Exploration Algorithm

### 3.1 Main Loop

```
Input:  start, goal, base_grid (for GT oracle only)
Output: trajectory poses, session_obs, subgraph

obs = np.zeros_like(base_grid, int8)  # -1=free, 1=obstacle, 0=unknown
current = start
trajectory = [start]
MAX_STEPS = grid_h * grid_w * coverage_budget  # e.g., 300*300*0.5 = 45000

for step in range(MAX_STEPS):
    # 1. FOV update
    obs = observe_single_fov(obs, current, base_grid)

    # 2. Goal reachability check
    pg = obs_free_planning_grid(obs)
    pg[start], pg[goal] = 0, 0  # force endpoints free (with inflate_radius clearance)
    path_to_goal, _ = astar(inflate(pg), current, goal)
    if path_to_goal is not None:
        trajectory.extend(path_to_goal)
        break  # SUCCESS

    # 3. Frontier selection
    frontier_cells = find_frontiers(obs)  # unknown cells adjacent to free cells
    if len(frontier_cells) == 0:
        break  # explored everything reachable

    next_frontier = select_nearest_frontier(frontier_cells, current)

    # 4. Navigate to frontier
    path_to_frontier, _ = astar(inflate(pg), current, next_frontier)
    if path_to_frontier is None:
        break  # stuck

    current = path_to_frontier[-1]  # move to farthest reachable cell on path
    trajectory.extend(path_to_frontier)
    current = trajectory[-1]
```

### 3.2 Frontier Definition

```
frontier = unknown_cell AND (UNIQUE(neighbor_4) contains free_cell)
```

That is: a cell whose value is 0 (unknown) in `obs`, and at least one of its 4-connected neighbors is -1 (observed free). Obstacle neighbors do not qualify.

### 3.3 Frontier Selection Strategy (with Perturbation)

The base strategy is nearest-frontier-first (Yamauchi 1997), augmented with **softmax-temperature perturbation** to produce diverse exploration paths across sessions.

**Algorithm:**

```
# 1. Filter top-N candidates by Euclidean distance
candidates = frontiers[argsort(euclidean_dist(current, f))][:FRONTIER_TOP_N]

# 2. Compute negative distances to the candidates on the known map
dists = [astar_len(inflate(pg), current, c) for c in candidates]

# 3. Softmax selection with per-session temperature T_k
logits = -np.array(dists) / T_k
probs  = softmax(logits)           # stable via exp(logits - max(logits))
next   = rng.choice(candidates, p=probs)
```

**Temperature schedule across sessions:**

```
T_k = FRONTIER_TEMP_MIN + k * (FRONTIER_TEMP_MAX - FRONTIER_TEMP_MIN) / max(K - 1, 1)
```

| k | T_k | Behavior |
|---|-----|----------|
| 0 (first session) | 0.5 | Near-greedy: almost always picks the nearest frontier |
| K-1 (last session) | 5.0 | Near-uniform: explores more varied areas |

**Initial yaw perturbation:** each session k starts exploration facing direction `k * (2π / K)` radians, further differentiating the initial FOV coverage.

**Performance optimization for large maps:** first filter frontiers by Euclidean distance (e.g., top 20 nearest), then compute A\* distances only for those candidates. On small grids like the maze (71×71), compute A\* for all frontiers directly.

### 3.4 Goal Check

After each FOV update, construct the partial planning grid:

```
pg = 1 - (obs == -1).astype(np.uint8)
# Clear inflate_radius neighborhood around start/goal:
for pt in [start, goal]:
    pg[pt[0]-R:pt[0]+R+1, pt[1]-R:pt[1]+R+1] = 0
inf_pg = inflate(pg)
path, _ = astar(inf_pg, current_position, goal)
```

If `path is not None`, the goal is reachable on the current partial map → navigate to it and terminate the session.

### 3.5 Observation Model

- FOV: 90° sector (FOV_HALF_DEG = 45°) pointing along robot yaw
- Range: FOV_RANGE_M (8.0 m for maze, 15.0 m for OSM-scale)
- Resolution: res_m (0.5 m/cell for octa_maze)

---

## 4. Topological Map Construction

`build_topometric_subgraph(trajectory)` extracts keyframes from the exploration trajectory:

| Event | Threshold |
|-------|-----------|
| New node inserted | Translation > TRANS_THRESH_M OR Rotation > ROT_THRESH_RAD |
| Edge between consecutive nodes | Weight = Euclidean distance (m) |

The resulting graph is a NetworkX Graph with node attributes `(x, y, yaw)`.

---

## 5. Cross-Session Map Merger

`merge_topometric_graphs(subgraphs[:k+1])`:
1. Relabel nodes from each subgraph with global offsets to avoid ID collision
2. Union all edges within each subgraph
3. For each pair of nodes from **different sessions**: if Euclidean distance < CROSS_DIST_M AND line-of-sight on `base_grid` is obstacle-free → add cross-session edge

Cross-session edges enable path planning across areas explored by different sessions.

---

## 6. Path Planning and Evaluation

### 6.1 Ground Truth Path

```
GT_path, GT_len = astar(inflate(base_grid), start, goal)
```

Computed once on the full known map. Represents the theoretical optimal path.

### 6.2 Topometric Path

For each cumulative merge level k:

```
topo = merged_topos_all[k]
start_node = nearest topo node to start (distance < TOPO_SNAP_DIST_M)
goal_node  = nearest topo node to goal  (distance < TOPO_SNAP_DIST_M)
topo_path_len = nx.shortest_path_length(topo, start_node, goal_node, weight="weight")
```

If no connecting path exists, the pair is considered unreachable at level k.

### 6.3 Optimality Metric

```
ratio[k] = topo_path_len[k] / GT_len
```

| ratio | Interpretation |
|-------|---------------|
| `= 1.0` | Topometric path matches ground-truth optimal |
| `> 1.0` | Topometric path is longer (suboptimal — incomplete map knowledge) |
| `NaN / inf` | Unreachable at this merge level |

**Expected behavior:** `ratio[k]` decreases monotonically (or non-increasing) as k grows, approaching 1.0.

---

## 7. Fixed Pair Selection

### 7.1 Constraints

- Both start and goal must lie on `inflate(base_grid) == 0` (safe free space)
- Both start and goal must have `base_grid == 0` (not on obstacle)
- Euclidean distance >= **DIST_MIN cells**
- GT-reachable: `astar(inflate(base_grid), start, goal)` must return a valid path

### 7.2 Selection Mode

**Mode A — CLI-provided (required for STL maze):**
- `--start R C --goal R C` given as command-line arguments
- Validated at startup; script exits with error message if invalid/unreachable
- Saved to `output/maze/fixed_pair.json` for reproducibility

**Mode B — Auto-sampled (fallback for general use):**
1. Sample candidates from `inflate(base_grid)` free cells
2. Filter by distance constraint
3. Score by GT path length / distance (longer paths = more interesting)
4. Select the highest-scoring pair
5. Fallback to lower distance threshold if primary fails
6. Save to `output/<mode>/fixed_pair.json`

---

## 8. Visualization

All figures at **300 dpi**.

### 8.1 fig1: Session Exploration Paths + Merged Graph

**Layout:** 2×3 panels (K=5: sessions 0–4 + Merged), size 18×12 inches.

**Session k subplot (panels 1–5):**
- Background: `base_grid` (obstacle=black, free=light gray)
- Topo nodes: blue solid dots (keyframes from session k only)
- Topo edges: blue thin lines (session k intra-session edges)
- Shortest path on **session k alone** topo graph: orange thick solid line (start→goal)
- If unreachable on session k's topo graph → label "unreachable", no orange line
- Start: green circle; Goal: red X
- Title: `Session {k}  T={T_k:.1f}  nodes={N}  reach=YES/NO`

**Merged subplot (panel 6):**
- Background: `base_grid`
- All session topo nodes: blue dots
- Intra-session edges: blue thin lines
- Cross-session edges: yellow thin lines
- Shortest path on merged topo graph: orange thick solid line
- GT optimal path: orange dashed line (reference)
- Start: green circle; Goal: red X
- Title: `Merged (k=0..{K-1})  topo={len:.1f}m  GT={gt_len:.1f}m  ratio={ratio:.2f}`

**Output:** `output/octa_maze/fig1_session_exploration.png`

### 8.2 fig2: Optimality Curve

**Layout:** single plot, size 8×5 inches.

- X-axis: Cumulative session count k (1 → K)
- Y-axis: Optimality ratio = `topo_path_len[k] / GT_len`
- Blue line + dots: ratio per cumulative merge level k
- Red dashed horizontal line: `ratio = 1.0` (GT optimal reference)
- Each data point annotated with value (e.g. `r=1.23`)
- Unreachable at level k → gray dashed segment + "∞" label
- Title: `Path Optimality vs. Number of Sessions`
- Legend: `topo ratio`, `GT optimal (1.0)`

**Output:** `output/octa_maze/fig2_optimality_curve.png`

### 8.3 fig3: Coverage Growth

**Layout:** 2×3 panels (K=5: cumulative k=1..5 + summary line plot), size 18×12 inches.

**Cumulative coverage subplot k (panels 1–5):**
- Background: `base_grid` (obstacle=black)
- Historical coverage (sessions 0..k-1): green semi-transparent overlay (`merged_obs[:k] == -1`)
- Current session k new coverage: **cyan** semi-transparent overlay (distinct from history)
- Current session k exploration trajectory: white thin line
- Start: green circle; Goal: red X
- Annotation: `k={k}  new_cov={new_pct:.0f}%  total_cov={total_pct:.0f}%  area={cov_m2:.0f}m²`

**Summary line plot (panel 6):**
- X-axis: Cumulative session count (1 → K)
- Y-axis: Coverage percentage (`cov_free_m2 / total_free_m2 × 100`)
- Blue line + dots: cumulative coverage curve
- Cyan bars or area fill: per-session new coverage contribution
- Title: `Cumulative Coverage Growth`

**Output:** `output/octa_maze/fig3_reachability_coverage.png`

### 8.4 Output Directory Structure

```
output/octa_maze/
├── base_map.npy
├── fixed_pair.json
├── fig1_session_exploration.png
├── fig2_optimality_curve.png
├── fig3_reachability_coverage.png
└── data/
    ├── session_0_poses.npy
    ├── session_0_obs.npy
    ├── ...
    ├── session_4_poses.npy
    ├── session_4_obs.npy
    ├── merged_obs_k0.npy
    ├── merged_obs_k1.npy
    ├── ...
    ├── topo_graph_k0.json
    ├── ...
    └── metrics.json
```

---

## 9. Implementation Scope (General)

### 9.1 New Functions

| Function | Responsibility |
|----------|---------------|
| `find_frontiers(obs)` | Return list of frontier cell coordinates |
| `select_frontier(frontiers, current, obs, rng, temperature, top_n)` | Softmax-temperature frontier selection with perturbation |
| `frontier_explore_session(start, goal, base_grid, rng, initial_yaw, frontier_temperature, res_m)` | Run one session of frontier exploration; return (poses, obs, subgraph) |
| `load_pcd_grid(pcd_path, resolution, height_slice, height_tolerance, dilate)` | Parse PCD, crop height slice, rasterize → 2D occupancy grid |
| `topo_path_length(topo_graph, start, goal, res_m)` | Shortest path on topo graph for given pair |
| `fig1_session_exploration(...)` | 2×3 panels: per-session topo graph paths + merged graph |
| `fig2_optimality_curve(...)` | Line plot: optimality ratio vs. session count |
| `fig3_reachability_coverage(...)` | 2×3 panels: cumulative coverage growth + summary line |

### 9.2 Removed Functions (from OSM benchmark)

| Function | Reason |
|----------|--------|
| `make_session_route()` | Replaced by `frontier_explore_session()` |
| `make_channel_route()` | Synthetic-only; out of scope for frontier exploration |
| `pick_best_fixed_pair()` | Absorbed into new `sample_goals()` with distance constraint |
| `_plan_on_merged_obs()` | No longer needed (topo graph path replaces grid A\* for estimation) |

---

## 10. Key Constants (General / OSM-scale)

| Constant | Value | Description |
|----------|-------|-------------|
| `FRONTIER_DIST_MIN` | 150 | Min cell distance between start and goal |
| `FRONTIER_DIST_FALLBACK` | 100 | Fallback distance if 150 fails |
| `MAX_STEPS_COVERAGE_BUDGET` | 0.5 | Max exploration steps = area × budget |
| `FOV_HALF_DEG` | 45.0 | Half field-of-view angle |
| `FOV_RANGE_M` | 15.0 | Sensor range in meters |
| `TRANS_THRESH_M` | 7.0 | Topo node translation threshold |
| `ROT_THRESH_RAD` | 1.047 (60°) | Topo node rotation threshold |
| `CROSS_DIST_M` | 10.0 | Max distance for cross-session edges |
| `INFLATE_RADIUS` | 3 | Obstacle inflation radius (cells) |
| `TOPO_SNAP_DIST_M` | 5.0 | Max distance to snap start/goal to nearest topo node |
| `FRONTIER_TEMP_MIN` | 0.5 | Softmax temperature for session 0 (near-greedy) |
| `FRONTIER_TEMP_MAX` | 5.0 | Softmax temperature for session K-1 (near-uniform) |
| `FRONTIER_TOP_N` | 5 | Number of nearest frontier candidates for softmax |

---

## 11. Octa Maze Implementation Plan (`frontier_explore_benchmark.py`)

### 11.1 Purpose

A standalone demonstration script using the `octa_maze.pcd` point cloud (35×35m garden maze, 250k points) to prove the core concept: all K=5 exploration sessions start from the same (start, goal) pair but use different exploration behavior (perturbed initial yaw + softmax-temperature frontier selection). As sessions accumulate, the shortest path on the merged topometric map converges toward the ground-truth optimal.

The script is self-contained (no OSM or internet dependency), runs in the `opennavmap` conda environment or any Python 3.8+ with numpy/scipy/matplotlib/networkx.

### 11.2 File Structure

```
benchmark_mms/
  frontier_explore_benchmark.py    ← new standalone script
  data/
    octa_maze.pcd                  ← 250k points, ASCII PCD
    octa_maze.stl                  ← original mesh (not used by script)
    stl2pcd_occupancy.py           ← reference: PCD→grid parameters
  output/
    octa_maze/                     ← output directory
      base_map.npy                 ← 71×71 uint8 occupancy grid
      fixed_pair.json              ← start/goal grid + world coords
      fig1_session_exploration.png ← 2×3 panels, 300dpi, 18×12in
      fig2_optimality_curve.png    ← single plot, 300dpi, 8×5in
      fig3_reachability_coverage.png ← 2×3 panels, 300dpi, 18×12in
      data/
        session_0_poses.npy
        session_0_obs.npy
        ...
        merged_obs_k0.npy
        merged_obs_k1.npy
        ...
        topo_graph_k0.json
        ...
        metrics.json
```

### 11.3 PCD Loading → 2D Grid Map

**Source:** `data/octa_maze.pcd` (ASCII PCD v0.7, 250,000 points, FIELDS x y z)

**Coordinate frame:** XZ is the ground plane (0–35 m × 0–35 m). Y is height (0–3.5 m). Grid maps `row = z / RES`, `col = x / RES`.

**Conversion parameters (from `stl2pcd_occupancy.py`):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `height_slice` | 2.0 m | Crop Y at this height for 2D projection |
| `height_tolerance` | 0.3 m | Keep points where `abs(Y - height_slice) ≤ tolerance`, i.e. Y ∈ [1.7, 2.3] |
| `resolution` | 0.5 m/cell | Output grid resolution |
| `dilate` | 1 px | Binary dilation to thicken walls |

**Expected grid shape:** 35 m / 0.5 m + 1 = **71×71 cells**

**Validation:** obstacle ratio must be in [0.04, 0.95]

```python
def load_pcd_grid(pcd_path, resolution=0.5,
                  height_slice=2.0, height_tolerance=0.3,
                  dilate=1):
    """Load ASCII PCD, crop Y-height slice, rasterize to 2D grid.
    
    Returns: grid (uint8, 0=free 1=obstacle),
             x_range=(x_min, x_max), z_range=(z_min, z_max)
    """
    pts = np.loadtxt(pcd_path, skiprows=10)       # skip 10-line PCD header
    x_range = (pts[:,0].min(), pts[:,0].max())     # (0.0, 35.0)
    z_range = (pts[:,2].min(), pts[:,2].max())     # (0.0, 35.0)
    # Height-based crop on Y
    mask = np.abs(pts[:,1] - height_slice) <= height_tolerance
    pts_slice = pts[mask]
    grid = generate_occupancy_grid(pts_slice, x_range, z_range, resolution)
    if dilate > 0:
        grid = binary_dilation(grid, structure=numpy.ones((3,3), bool)).astype(np.uint8)
    return grid, x_range, z_range
```

### 11.4 Maze-Specific Constants

Scaled proportionally from the 1000×1000 @ 0.5 m/cell OSM benchmark to the 71×71 grid (scale factor ≈ 71/1000 = 0.071):

| Constant | Maze Value | OSM Value | Description |
|----------|------------|-----------|-------------|
| `N_SESSIONS` (K) | **5** | 10 | Number of exploration sessions |
| `GRID_RES_M` | **0.5** | 0.5 | Grid resolution (m/cell) |
| `GRID_SHAPE` | **71×71** | 1000×1000 | Grid dimensions |
| `MAP_SIZE_M` | **35×35 m** | 500×500 m | Real-world map size |
| `PCD_HEIGHT_SLICE` | **2.0** | — | Height (Y) to crop for 2D grid (m) |
| `PCD_HEIGHT_TOL` | **0.3** | — | Tolerance band around height_slice (m) |
| `PCD_DILATE` | **1** | — | Binary dilation radius (px) for walls |
| `FOV_HALF_DEG` | **45.0** | 45.0 | Half field-of-view angle |
| `FOV_RANGE_M` | **8.0** | 15.0 | Sensor range (meters) |
| `TRANS_THRESH_M` | **2.0** | 7.0 | Topo node translation threshold |
| `ROT_THRESH_RAD` | **1.047 (60°)** | 1.047 (60°) | Topo node rotation threshold |
| `CROSS_DIST_M` | **3.0** | 10.0 | Max distance for cross-session edges |
| `INFLATE_RADIUS` | **1** | 2 | Obstacle inflation radius (cells) — maze corridors are narrow (2-4 cells wide) |
| `FRONTIER_DIST_MIN` | **30** | 150 | Min cell distance start↔goal |
| `FRONTIER_DIST_FALLBACK` | **15** | 100 | Fallback distance |
| `TOPO_SNAP_DIST_M` | **3.0** | 5.0 | Snap start/goal to nearest topo node |
| `MAX_STEPS_COVERAGE_BUDGET` | **0.5** | 0.5 | Max steps = h × w × budget |
| `FRONTIER_TEMP_MIN` | **0.5** | — | Softmax temp for session 0 (near-greedy) |
| `FRONTIER_TEMP_MAX` | **5.0** | — | Softmax temp for session K-1 (near-uniform) |
| `FRONTIER_TOP_N` | **5** | — | Number of nearest frontier candidates |

### 11.5 Session Configuration

- **K = 5**: default, overridable via `--k`
- **Fixed (start, goal)**: ALL K sessions share the **same** start and goal, provided via CLI `--start R C --goal R C` in grid coordinates:
  - start: world `(x=2.5, y=2.0, z=3.5)` → grid `(row=7, col=5)` at 0.5 m/cell
  - goal:  world `(x=32.0, y=2.0, z=32.5)` → grid `(row=65, col=64)` at 0.5 m/cell
  - Euclidean distance ≈ 82.7 cells (~41.4 m) — satisfies FRONTIER_DIST_MIN=30
- **Session diversity via exploration perturbation**: same start/goal, but different exploration behavior per session:
  - **RNG**: `np.random.default_rng(master_seed + k)` per session
  - **Initial yaw**: `k * (2π / K)` radians — session 0 faces +X (east), session 1 72° rotated, etc.
  - **Frontier selection softmax temperature**: `T_k = T_MIN + k * (T_MAX - T_MIN) / max(K-1, 1)`
    - session 0: T=0.5 (near-greedy: picks nearest frontier with high probability)
    - session 4: T=5.0 (near-uniform: explores varied areas)
  - **Frontier candidates**: top-N=5 nearest frontiers by Euclidean distance, then softmax over `-dist / T_k`

### 11.6 CLI Interface

```bash
# Required: provide fixed start and goal in grid coordinates (row, col)
python frontier_explore_benchmark.py --start 7 5 --goal 65 64

# Optional overrides
python frontier_explore_benchmark.py --start 7 5 --goal 65 64 \
    --res_m 0.5 --k 5 --seed 42 --output_dir ./output/octa_maze
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--start R C` | Yes | — | Fixed start cell (row, col) in grid coords |
| `--goal R C` | Yes | — | Fixed goal cell (row, col) in grid coords |
| `--res_m` | No | 0.5 | Grid resolution (m/cell) |
| `--k` | No | 5 | Number of sessions |
| `--seed` | No | 42 | Master random seed |
| `--output_dir` | No | `./output/octa_maze` | Output directory |

### 11.7 Dependencies

```bash
# Core (required):
pip install numpy matplotlib scipy networkx

# Optional:
pip install imageio       # for map_growth.gif; fails gracefully if missing
```

No OSM-related dependencies (osmnx, geopandas, shapely). No trimesh needed (PCD loaded via numpy). No internet required.

### 11.8 Execution Flow

```
1. Load data/octa_maze.pcd → crop Y∈[1.7, 2.3] → rasterize XZ at 0.5 m/cell
   → dilate 1 px → base_grid (71×71, uint8) → save base_map.npy
2. Validate --start (7,5) and --goal (65,64): both free + GT-reachable
3. Save fixed_pair.json:
   {"start":[7,5], "goal":[65,64],
    "world_start":[2.5,2.0,3.5], "world_goal":[32.0,2.0,32.5]}
4. For k in 0..K-1:
   a. rng_k = np.random.default_rng(seed + k)
      initial_yaw_k = k * (2π / K)
      temperature_k = T_MIN + k * (T_MAX - T_MIN) / max(K-1, 1)
   b. Run frontier_explore_session(fixed_start, fixed_goal, base_grid,
                                    rng=rng_k,
                                    initial_yaw=initial_yaw_k,
                                    frontier_temperature=temperature_k)
   c. Build topometric subgraph from trajectory
   d. Merge subgraph with previous sessions
   e. Evaluate topo_path for (fixed_start, fixed_goal) at each merge level
5. Generate figures (all 300 dpi):
   - fig1_session_exploration.png   (2×3 panels, 18×12 in)
   - fig2_optimality_curve.png      (single plot, 8×5 in)
   - fig3_reachability_coverage.png (2×3 panels, 18×12 in)
6. Save all data snapshots to output/octa_maze/data/
7. Print quantitative summary to terminal:
   k  temp  initial_yaw  nodes  reachable  topo_len(m)  GT(m)  ratio  cov_free_m2

### 11.9 Expected Results

- `ratio[0]` (session 0 alone, T=0.5, greedy): should be > 1.0 or unreachable — single session has incomplete knowledge
- `ratio[k]` decreases monotonically as k grows, approaching 1.0
- Early sessions (0–1, low temperature) cover direct corridor; later sessions (3–4, high temperature) explore side branches
- By k=4 (merged all 5), `ratio` should be close to 1.0 (near-optimal path on merged topo map)
- `node_count` and `cov_free_m2` increase with k
- `cov_pct` (coverage percentage) grows, with cyan overlays in fig3 showing diminishing new coverage in later sessions

### 11.10 Self-Contained Design

The script is **entirely standalone** — it copies in all shared utility functions from `multisession_sim_osm.py` (A\*, inflate, build_topometric_subgraph, merge_topometric_graphs, observe_batch_fov, _line_free, _add_subgraph_to_merged) rather than importing them. This avoids coupling to the OSM benchmark code and ensures the script can be moved or shared independently.
