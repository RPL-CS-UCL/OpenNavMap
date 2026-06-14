# Frontier-Based Goal-Directed Exploration + Topological Map Design

> **Created:** 2025-06-13  
> **Status:** Approved, pending implementation plan  
> **Context:** Multi-Session Mapping Benchmark (`multisession_sim_osm.py`)

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
                   │ fix pair │  (start, goal) sampled once before all sessions
                   │  save to │  distance >= 150 cells, GT-reachable
                   │ file.json│
                   └────┬─────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Session 1        Session 2    ...  Session N
   ┌─────────┐    ┌─────────┐       ┌─────────┐
   │Frontier │    │Frontier │       │Frontier │
   │Explore  │    │Explore  │       │Explore  │
   │+Goal Chk│    │+Goal Chk│       │+Goal Chk│
   └────┬────┘    └────┬────┘       └────┬────┘
        │               │               │
        ▼               ▼               ▼
    topo_map_1      topo_map_2       topo_map_N
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
                 optimality_curve (k=1..N approaches 1.0)
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

### 3.5 Observation Model (unchanged)

- FOV: 90° sector (FOV_HALF_DEG = 45°) pointing along robot yaw
- Range: 15 meters (FOV_RANGE_M = 15.0)
- Resolution: per-session res_m (default 1.0 m/cell)

---

## 4. Topological Map Construction (unchanged)

`build_topometric_subgraph(trajectory)` extracts keyframes from the exploration trajectory:

| Event | Threshold |
|-------|-----------|
| New node inserted | Translation > TRANS_THRESH_M (7.0 m) OR Rotation > ROT_THRESH_RAD (60°) |
| Edge between consecutive nodes | Weight = Euclidean distance (m) |

The resulting graph is a NetworkX Graph with node attributes `(x, y, yaw)`.

---

## 5. Cross-Session Map Merger (unchanged)

`merge_topometric_graphs(subgraphs[:k+1])`:
1. Relabel nodes from each subgraph with global offsets to avoid ID collision
2. Union all edges within each subgraph
3. For each pair of nodes from **different sessions**: if Euclidean distance < CROSS_DIST_M (10 m) AND line-of-sight on `base_grid` is obstacle-free → add cross-session edge

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
start_node = nearest topo node to start (distance < 5.0 m)
goal_node  = nearest topo node to goal  (distance < 5.0 m)
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
- Euclidean distance >= **150 cells** (150 m @ 1 m/cell resolution)
- GT-reachable: `astar(inflate(base_grid), start, goal)` must return a valid path

### 7.2 Sampling Strategy

1. Sample ~200 candidates from `inflate(base_grid)` free cells
2. Filter by distance constraint (>= 150 cells)
3. Score by GT path length / distance (longer paths = more interesting exploration)
4. Select the pair with the highest score
5. If sampling fails at 150 cells threshold, fallback to >= 100 cells
6. Save to `output/real/fixed_pair.json` for reproducibility

---

## 8. Visualization

### 8.1 fig1: Per-Session Frontier Exploration

10 panels (5×2 grid). Each panel k shows:
- **Background:** `sessions_obs[k]` — only what session k observed (unvisited = black)
- **Exploration trajectory:** white line showing the frontier-to-frontier path
- **Topo nodes:** blue dots (keyframes from session k only)
- **Topo edges:** white transparent lines
- **Start:** green circle; **Goal:** red X
- **Topo path:** orange line from start to goal on this session's topo graph
- **GT path:** orange dashed line for reference
- **Title:** `k=X  topo-reach=YES/NO  topo_len=...m  GT=...m  ratio=...`

### 8.2 fig2: Fixed Pair Across Incremental Merge

10 panels (5×2 grid). Each panel k shows:
- **Background:** `merged_all[k]` — cumulative merge of sessions 0..k
- **Merged topo graph:** nodes (blue) + edges (white + cross-session in yellow)
- **Fixed start/goal:** green circle / red X
- **Topo path:** orange thick line
- **GT path:** orange dashed
- **Title:** `k=X  reach=YES/NO  topo=...m  GT=...m  ratio=...`

### 8.3 Output Files

| File | Description |
|------|-------------|
| `fig0_osm_base_map.png` | Full OSM occupancy map |
| `fig1_per_session_exploration.png` | 10-panel frontier exploration per session |
| `fig2_fixed_pair_merge.png` | 10-panel incremental merge with topo paths |
| `fig4_temporal_adaptability.png` | Temporal adaptability (unchanged) |
| `fig6_summary_table.png` | Quantitative summary |
| `map_growth.gif` | Map growth animation |

---

## 9. Implementation Scope

### 9.1 New Functions

| Function | Responsibility |
|----------|---------------|
| `find_frontiers(obs)` | Return list of frontier cell coordinates |
| `select_nearest_frontier(frontiers, current, obs)` | Pick closest frontier by A\* distance |
| `frontier_explore_session(start, goal, base_grid, res_m)` | Run one session of frontier exploration; return (poses, obs, subgraph) |
| `topo_path_length(topo_graph, start, goal, res_m)` | Shortest path on topo graph for given pair |
| `fig1_per_session_exploration(...)` | New fig1 with frontier trajectory + topo path |
| `fig2_fixed_pair_merge(...)` | Updated fig2 with topo-based paths |

### 9.2 Modified Functions

| Function | Change |
|----------|--------|
| `sample_goals()` | Increase distance threshold to 150 cells; save pair to file |
| `run_experiments()` | Replace A\*-on-partial-grid with topo-graph shortest path; call `frontier_explore_session()` per session |
| `main()` | Pass fixed pair to session generation; wire new fig functions |
| `_print_experiment_summary()` | Add topo-based reachability columns |

### 9.3 Removed Functions

| Function | Reason |
|----------|--------|
| `make_session_route()` | Replaced by `frontier_explore_session()` |
| `make_channel_route()` | Synthetic-only; out of scope for frontier exploration |
| `pick_best_fixed_pair()` | Absorbed into new `sample_goals()` with distance constraint |
| `_plan_on_merged_obs()` | No longer needed (topo graph path replaces grid A\* for estimation) |

---

## 10. Key Constants

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
