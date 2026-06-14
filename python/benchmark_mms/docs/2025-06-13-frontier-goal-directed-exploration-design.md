# Frontier-Based Goal-Directed Exploration + Topological Map Design

> **Created:** 2025-06-13
> **Updated:** 2025-06-14 — Added STL maze implementation plan (Section 11)
> **Status:** Approved, implementing for STL maze
> **Context:** Multi-Session Mapping Benchmark

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

**The goal is fixed across all sessions, while each session's starting position is randomly sampled from the full free space.** This means the final evaluation pair (start_for_eval, fixed_goal) uses start_for_eval = the same start used by all sessions' trial, allowing direct comparison of path optimality as the merged map grows.

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

### 3.3 Frontier Selection Strategy

Select the frontier cell with the **shortest A\* path distance** from the current position on the known map:

```
next = argmin( astar_len(current, f) for f in frontiers )
```

This is the nearest-frontier-first strategy (Yamauchi 1997), which minimizes travel distance between frontiers.

**Performance optimization for large maps:** first filter frontiers by Euclidean distance (e.g., top 20 nearest), then compute A\* distances only for those candidates. On small grids like the maze (72×72), compute A\* for all frontiers directly.

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
- Range: sensor-specific (see Section 11 for maze-specific values)
- Resolution: res_m (0.5 m/cell for STL maze)

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

### 8.1 fig0: Base Map

Single panel showing:
- Full base_grid occupancy map (free = light, obstacle = dark)
- Fixed start (green circle) and goal (red X)
- GT path (orange dashed line)
- Title with map stats (grid size, resolution, free/obs ratio)

### 8.2 fig1: Per-Session Frontier Exploration

K panels (grid layout depends on K: 1×5 for K=5, 5×2 for K=10). Each panel k shows:
- **Background:** `sessions_obs[k]` — only what session k observed (unvisited = black)
- **Exploration trajectory:** white line showing the frontier-to-frontier path
- **Topo nodes:** blue dots (keyframes from session k only)
- **Topo edges:** white transparent lines
- **Session start:** green circle (random per session)
- **Fixed Goal:** red X
- **GT path:** orange dashed line for reference
- **Title:** `k=X  topo-reach=YES/NO  topo_len=...m  GT=...m  ratio=...`

### 8.3 fig2: Fixed Pair Across Incremental Merge

K panels. Each panel k shows:
- **Background:** `merged_all[k]` — cumulative merge of sessions 0..k
- **Merged topo graph:** nodes (blue) + intra-session edges (white) + cross-session edges (yellow)
- **Fixed start/goal:** green circle / red X
- **Topo path:** orange thick line
- **GT path:** orange dashed
- **Title:** `k=X  reach=YES/NO  topo=...m  GT=...m  ratio=...`

### 8.4 fig6: Summary Table

K-row quantitative summary table with columns:
k, session_start, reachable, topo_len, GT_len, ratio, nodes, cov_free_m2

### 8.5 map_growth.gif

K frames (one per cumulative session), FPS=3. Each frame: merged map + current session trajectory + stats.

### 8.6 Output Files

| File | Description |
|------|-------------|
| `fig0_maze_map.png` | Full maze occupancy map with start/goal/GT path |
| `fig1_per_session_exploration.png` | K-panel frontier exploration per session |
| `fig2_fixed_pair_merge.png` | K-panel incremental merge with topo paths |
| `fig6_summary_table.png` | Quantitative summary |
| `map_growth.gif` | Map growth animation |

---

## 9. Implementation Scope (General)

### 9.1 New Functions

| Function | Responsibility |
|----------|---------------|
| `find_frontiers(obs)` | Return list of frontier cell coordinates |
| `select_nearest_frontier(frontiers, current, obs)` | Pick closest frontier by A\* distance |
| `frontier_explore_session(start, goal, base_grid, rng, res_m)` | Run one session of frontier exploration; return (poses, obs, subgraph) |
| `topo_path_length(topo_graph, start, goal, res_m)` | Shortest path on topo graph for given pair |
| `fig1_per_session_exploration(...)` | New fig1 with frontier trajectory + topo path |
| `fig2_fixed_pair_merge(...)` | Updated fig2 with topo-based paths |

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

---

## 11. STL Maze Implementation Plan (`frontier_explore_benchmark.py`)

### 11.1 Purpose

A standalone demonstration script using the `octa_maze.stl` file (35×35m garden maze) to prove the core concept: as K=5 exploration sessions accumulate, the shortest path on the merged topometric map from a fixed start to a fixed goal converges toward the ground-truth optimal.

The script is self-contained (no OSM or internet dependency), runs in the `opennavmap` conda environment or any Python 3.8+ with numpy/scipy/matplotlib/networkx.

### 11.2 File Structure

```
benchmark_mms/
  frontier_explore_benchmark.py    ← new standalone script
  output/
    maze/
      base_map.npy
      fixed_pair.json
      fig0_maze_map.png
      fig1_per_session_exploration.png
      fig2_fixed_pair_merge.png
      fig6_summary_table.png
      map_growth.gif
      data/
        session_01_poses.npy
        session_01_obs.npy
        ...
        merged_obs_k01.npy
        topo_graph_k01.json
        metrics.json
```

### 11.3 STL Parsing → 2D Grid Map

**Source:** `/data/octa_maze.stl` (binary STL, ASCII-sig header `solid garden_maze2`, 1478 triangles)

**Coordinate frame:** The STL uses Y for height (0–3.5m with walls at Y≈3.5m) and XZ for the floor plane (0–35m × 0–35m). Walls are vertical faces with `|normal_Y| < 0.1`.

**Method:**
1. Parse binary STL using `struct` (zero external dependencies — fallback to `trimesh` if available)
2. Identify wall triangles: faces where `|normal_Y| < 0.1` (vertical faces)
3. Project wall footprints onto the XZ plane (rasterize each wall triangle's bounding box)
4. Resolution: 0.5 m/cell → 70×70 grid + 2-cell border = **72×72 cells**
5. Add 2-cell border walls
6. Validate: obstacle ratio must be in [0.04, 0.95]

**Grid coordinates:** `grid[row][col]` where `row = z / RES`, `col = x / RES`

```python
def load_stl_grid(stl_path, res_m=0.5):
    """Parse binary STL and rasterize walls to 2D occupancy grid.
    
    Returns: grid (uint8, 0=free 1=obstacle), shape=(grid_h, grid_w)
    """
    # Pure-Python struct-based STL parser
    # Extract wall faces (|normal_Y| < 0.1)
    # Rasterize bounding boxes of wall triangles to XZ grid
    # Add border walls
    # Return validated grid
```

### 11.4 Maze-Specific Constants

Scaled proportionally from the 1000×1000 @ 0.5 m/cell OSM benchmark to the 72×72 grid (scale factor ≈ 72/1000 = 0.072):

| Constant | Maze Value | OSM Value | Description |
|----------|------------|-----------|-------------|
| `N_SESSIONS` (K) | **5** | 10 | Number of exploration sessions |
| `GRID_RES_M` | **0.5** | 0.5 | Grid resolution (m/cell) |
| `GRID_SHAPE` | **72×72** | 1000×1000 | Grid dimensions |
| `MAP_SIZE_M` | **35×35 m** | 500×500 m | Real-world map size |
| `FOV_HALF_DEG` | **45.0** | 45.0 | Half field-of-view angle |
| `FOV_RANGE_M` | **8.0** | 15.0 | Sensor range (meters) |
| `TRANS_THRESH_M` | **2.0** | 7.0 | Topo node translation threshold |
| `ROT_THRESH_RAD` | **1.047 (60°)** | 1.047 (60°) | Topo node rotation threshold |
| `CROSS_DIST_M` | **3.0** | 10.0 | Max distance for cross-session edges |
| `INFLATE_RADIUS` | **2** | 3 | Obstacle inflation radius (cells) |
| `FRONTIER_DIST_MIN` | **30** | 150 | Min cell distance start↔goal |
| `FRONTIER_DIST_FALLBACK` | **15** | 100 | Fallback distance |
| `TOPO_SNAP_DIST_M` | **3.0** | 5.0 | Snap start/goal to nearest topo node |
| `MAX_STEPS_COVERAGE_BUDGET` | **0.5** | 0.5 | Max steps = h × w × budget |

### 11.5 Session Configuration

- **K = 5**: default, overridable via `--k`
- **Fixed goal**: provided via CLI `--goal R C`, same for all K sessions
- **Fixed evaluation start**: same as fixed goal's pair start, provided via CLI `--start R C`
- **Per-session start**: uniformly random from `inflate(base_grid)` free cells, using `np.random.default_rng(session_id)` for reproducibility
- Each session starts exploration from its random start, with awareness of the fixed goal for the goal-check loop

### 11.6 CLI Interface

```bash
# Required: provide fixed start and goal in grid coordinates (row, col)
python frontier_explore_benchmark.py --start 5 5 --goal 60 60

# Optional overrides
python frontier_explore_benchmark.py --start 5 5 --goal 60 60 \
    --res_m 0.5 --k 5 --seed 42 --output_dir ./output/maze
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--start R C` | Yes | — | Fixed start cell (row, col) for the evaluation pair |
| `--goal R C` | Yes | — | Fixed goal cell (row, col) |
| `--res_m` | No | 0.5 | Grid resolution (m/cell) |
| `--k` | No | 5 | Number of sessions |
| `--seed` | No | 42 | Master random seed |
| `--output_dir` | No | `./output/maze` | Output directory |

### 11.7 Dependencies

```bash
# Core (required):
pip install numpy matplotlib scipy networkx

# Optional (better STL rendering):
pip install trimesh       # falls back to pure-Python struct parser if unavailable
pip install imageio       # for map_growth.gif; fails gracefully if missing
```

No OSM-related dependencies (osmnx, geopandas, shapely). No internet required.

### 11.8 Execution Flow

```
1. Parse octa_maze.stl → 72×72 occupancy grid (base_map.npy)
2. Validate CLI-provided --start and --goal (free + GT-reachable)
3. Save fixed_pair.json
4. For k in 1..K:
   a. Sample random session_start from free cells
   b. Run frontier_explore_session(session_start, fixed_goal, base_grid)
   c. Build topometric subgraph from trajectory
   d. Merge subgraph with previous sessions
   e. Evaluate topo_path for fixed evaluation pair (start, fixed_goal)
5. Generate output figures (fig0, fig1, fig2, fig6, map_growth.gif)
6. Save all data snapshots
7. Print quantitative summary to terminal
```

### 11.9 Expected Results

- `ratio[1]` should be > 1.0 (or unreachable) — session 1 alone has incomplete knowledge
- `ratio[k]` should decrease monotonically as k grows, approaching 1.0
- By k=5, `ratio` should be close to 1.0 (near-optimal path on merged topo map)
- `node_count` and `cov_free_m2` should increase with k

### 11.10 Self-Contained Design

The script is **entirely standalone** — it copies in all shared utility functions from `multisession_sim_osm.py` (A\*, inflate, build_topometric_subgraph, merge_topometric_graphs, observe_batch_fov, _line_free, _add_subgraph_to_merged) rather than importing them. This avoids coupling to the OSM benchmark code and ensures the script can be moved or shared independently.
