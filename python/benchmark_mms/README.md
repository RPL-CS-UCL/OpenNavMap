# benchmark_mms — Multi-Session Frontier Exploration Benchmark

This benchmark evaluates goal-directed frontier exploration and multi-session
topometric map merging on `octa_maze`, `duplex_office`, and `tunnel` point-cloud
maps. The current experiment compares normal repeated exploration with a
day-change setting where an artificial obstacle blocks part of the map for the
first sessions and is removed in later sessions.

## Overview

Each environment runs `K` sessions from a fixed `(start, goal)` pair. Session
diversity comes from different initial yaw values while the frontier softmax
temperature is fixed by the run script, currently `--temperature 2.5`.

The goal is to test whether cumulative topometric merging can recover useful
navigation structure across sessions, improve the topometric shortest path
toward the ground-truth path, and expose the impact of environment changes on
reachability and explored area.

The benchmark measures:

- shortest path length on the cumulative topometric map,
- optimality ratio `topo_len / GT_len`,
- reachability across cumulative sessions,
- cumulative explored free-space area in square meters (`m²`).

Coordinates follow the current convention:

- PCD ground plane: `X/Y`
- PCD height axis: `Z`
- grid column axis: `X`
- grid row axis: `Y`
- CLI `--start` and `--goal`: world coordinates `(col_m, row_m)`

## Requirements

```bash
conda activate opennavmap
pip install numpy matplotlib scipy networkx
```

## Approach

- Convert each PCD into a 2D occupancy grid using `X/Y` as the ground plane and
  `Z` as height.
- Run `K` frontier-exploration sessions from the same start/goal pair with fixed
  temperature `2.5` and different initial yaw values.
- Build a per-session topometric graph from the explored trajectory, then merge
  graphs cumulatively using distance, line-of-sight, and A* reachability checks.
- For day-change runs, inject a rectangular obstacle block in world coordinates
  for day0 sessions and remove it from `--day_change` onward.
- Plot cumulative path optimality, reachability, and explored free-space area in
  `m²`; export all maps, ratios, coverage, and topometric graphs under `data/`.

## Recommended Runs

Run from the repository root:

```bash
cd /Titan/code/robohike_ws/src/opennavmap

bash python/benchmark_mms/scripts/run_duplex_office.sh
bash python/benchmark_mms/scripts/run_octa_maze.sh
bash python/benchmark_mms/scripts/run_tunnel.sh

bash python/benchmark_mms/scripts/run_duplex_office_daychange.sh
bash python/benchmark_mms/scripts/run_octa_maze_daychange.sh
bash python/benchmark_mms/scripts/run_tunnel_daychange.sh
```

Current script settings:

| Script | Resolution | Sessions | Start `(col,row)` | Goal `(col,row)` | Temperature |
|--------|------------|----------|-------------------|------------------|-------------|
| `run_duplex_office.sh` | `0.2 m` | `5` | `(1.5, 2.5)` | `(19, 18)` | `2.5` |
| `run_octa_maze.sh` | `0.2 m` | `5` | `(2.5, 4)` | `(30, 33)` | `2.5` |
| `run_tunnel.sh` | `0.3 m` | `10` | `(-15, -5)` | `(201, 132)` | `2.5` |
| `run_duplex_office_daychange.sh` | `0.2 m` | `5` | `(1.5, 2.5)` | `(19, 18)` | `2.5` |
| `run_octa_maze_daychange.sh` | `0.2 m` | `5` | `(2.5, 4)` | `(30, 33)` | `2.5` |
| `run_tunnel_daychange.sh` | `0.3 m` | `10` | `(-15, -5)` | `(201, 132)` | `2.5` |

`run_tunnel.sh` and `run_tunnel_daychange.sh` set `--max_steps 8000` to avoid
the area-based default step budget on the large tunnel map.

Day-change scripts use `--day_change 2`, so sessions `0` and `1` run on day0
with the obstacle block, while later sessions run on the clear day1 map.

## Latest Results

Results from the current scripts after the merged-panel correction:

| Run | Reachable | Final ratio | Final coverage |
|-----|-----------|-------------|----------------|
| `duplex_office` | `5/5` | `1.0377` | `259.00 m²` (`87.018%`) |
| `duplex_office_daychange` | `3/5` | `1.0000` | `265.48 m²` (`89.195%`) |
| `octa_maze` | `5/5` | `1.0584` | `663.72 m²` (`78.487%`) |
| `octa_maze_daychange` | `3/5` | `1.0637` | `681.48 m²` (`80.587%`) |
| `tunnel` | `10/10` | `1.0104` | `5111.19 m²` (`6.840%`) |
| `tunnel_daychange` | `10/10` | `1.0104` | `5270.67 m²` (`7.054%`) |

Day-change runs intentionally show early failures in `duplex_office` and
`octa_maze` because the day0 obstacle blocks the route. After the obstacle is
removed, cumulative merging recovers reachability and the topometric path ratio
approaches the ground-truth path.

## Direct CLI Usage

```bash
python python/benchmark_mms/frontier_explore_benchmark.py \
  --pcd python/benchmark_mms/data/duplex_office.pcd \
  --output_dir python/benchmark_mms/output/duplex_office \
  --start 1.5 2.5 \
  --goal 19 18 \
  --res_m 0.2 \
  --k 5 \
  --seed 42 \
  --temperature 2.5
```

## CLI Arguments

| Argument | Description |
|----------|-------------|
| `--start COL_M ROW_M` | Start position in world coordinates along `(col_axis,row_axis)` |
| `--goal COL_M ROW_M` | Goal position in world coordinates along `(col_axis,row_axis)` |
| `--res_m` | Occupancy-grid resolution in meters per cell |
| `--k` | Number of sessions |
| `--seed` | Master random seed |
| `--temperature` | Fixed frontier softmax temperature for all sessions |
| `--pcd` | Input PCD path |
| `--output_dir` | Output directory |
| `--col_axis` | PCD field index for grid columns, default `0` (`X`) |
| `--row_axis` | PCD field index for grid rows, default `1` (`Y`) |
| `--height_axis` | PCD field index for height slice, default `2` (`Z`) |
| `--dilate` | Obstacle dilation radius in pixels, default `PCD_DILATE=1` |
| `--max_steps` | Optional per-session step limit |
| `--obstacle_block COL_MIN ROW_MIN COL_MAX ROW_MAX` | Optional day0 obstacle block in world coordinates |
| `--day_change` | First session index that switches from day0 blocked map to day1 clear map |

## Outputs

Each run writes to `python/benchmark_mms/output/<environment>/`.

Top-level outputs:

| File | Description |
|------|-------------|
| `fig1_session_exploration.png` | Per-session observations, trajectories, topo graphs, and merged map |
| `fig2_optimality_curve.png` | Cumulative path optimality ratio curve |
| `fig3_reachability_coverage.png` | Cumulative coverage growth plotted in `m²` |
| `fixed_pair.json` | Start/goal metadata and GT path length |
| `base_map.npy` | Occupancy grid copy for compatibility |

Data outputs under `data/`:

| File | Description |
|------|-------------|
| `base_map.npy` | Occupancy grid used by the run |
| `metrics.json` | Run metadata, ratios, node counts, trajectory lengths |
| `ratios.json` | `gt_len_m` and cumulative `topo_len / GT_len` ratios |
| `coverage.json` | `cum_m2`, `new_m2`, `cum_pct`, `new_pct` per session |
| `session_<k>_poses.npy` | Session trajectory `(row, col, yaw)` |
| `session_<k>_obs.npy` | Session observation grid (`-1=free`, `1=obstacle`, `0=unknown`) |
| `merged_obs_k<k>.npy` | Cumulative observation grid through session `k` |
| `topomap_k<k>.npz` | Single-session topo nodes, edges, edge weights, start/goal node IDs |
| `topomap_merged_k<k>.npz` | Cumulative merged topo map through session `k` |

`topomap_*.npz` contains:

- `nodes_xy`: node coordinates in meters,
- `edges`: integer node-index pairs,
- `edge_weights`: edge path lengths in meters,
- `start_node` / `goal_node` for session maps when available,
- `start_nodes` / `goal_nodes` for merged maps when available.

Generated result directories are ignored by git via `.gitignore`.

## Implementation Notes

- FOV observations use ray casting: obstacle cells terminate visibility, so cells
  behind obstacles are not marked as visible.
- A* uses 8-neighbor Euclidean costs and a closed set.
- Frontier navigation uses local-window A* for short-range planning on large maps.
- The first topo node is the true start. The goal node is recorded only if the
  session actually reaches the goal.
- Non-adjacent nodes inside one session can receive loop-closure edges if they
  satisfy the same distance, line-of-sight, and A* reachability checks used for
  cross-session merge edges.
- `fig3_reachability_coverage.png` reports cumulative free-space coverage in
  square meters, with percentages retained in annotations and `coverage.json`.
- Day-change `fig1_session_exploration.png` shows the blocked region on day0 and
  the removed region on day1/merged panels; merged-panel display uses a clear
  grid for visualization while stored metrics remain tied to the active grid.

## Tests

```bash
cd /Titan/code/robohike_ws/src/opennavmap
pytest python/benchmark_mms/tests/test_frontier_benchmark.py -q
```
