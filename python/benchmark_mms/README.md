# benchmark_mms

Multi-session frontier exploration benchmark for `duplex_office`, `octa_maze`,
and `tunnel` point-cloud maps.

## Goal

This experiment evaluates whether cumulative topometric map merging improves
goal-directed navigation over repeated sessions. It compares:

- baseline runs on a fixed map,
- day-change runs where an artificial obstacle blocks the route in sessions `0`
  and `1`, then is removed from session `2` onward.

Metrics are cumulative reachability, topometric shortest-path ratio
`topo_len / GT_len`, and explored free-space area in `m²`.

## Method

- Convert each PCD to a 2D occupancy grid using `X/Y` as ground plane and `Z` as
  height.
- Run multiple frontier-exploration sessions from the same start/goal pair with
  fixed temperature `2.5` and different initial yaws.
- Build one topometric graph per session and merge graphs cumulatively using
  distance, line-of-sight, and A* reachability checks.
- For day-change runs, apply `--obstacle_block COL_MIN ROW_MIN COL_MAX ROW_MAX`
  before `--day_change`, then switch back to the clear map.

Coordinates for `--start`, `--goal`, and `--obstacle_block` are world
`(col_m, row_m)` coordinates.

## Usage

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

Direct example:

```bash
python python/benchmark_mms/frontier_explore_benchmark.py \
  --pcd python/benchmark_mms/data/duplex_office.pcd \
  --output_dir python/benchmark_mms/output/duplex_office \
  --start 1.5 2.5 --goal 19 18 \
  --res_m 0.2 --k 5 --seed 42 --temperature 2.5
```

## Results

Latest results after the merged-panel visualization correction:

| Run | Reachable | Final ratio | Final coverage |
|-----|-----------|-------------|----------------|
| `duplex_office` | `5/5` | `1.0377` | `259.00 m²` (`87.018%`) |
| `duplex_office_daychange` | `3/5` | `1.0000` | `265.48 m²` (`89.195%`) |
| `octa_maze` | `5/5` | `1.0584` | `663.72 m²` (`78.487%`) |
| `octa_maze_daychange` | `3/5` | `1.0637` | `681.48 m²` (`80.587%`) |
| `tunnel` | `10/10` | `1.0104` | `5111.19 m²` (`6.840%`) |
| `tunnel_daychange` | `10/10` | `1.0104` | `5270.67 m²` (`7.054%`) |

The day-change runs intentionally fail early in `duplex_office` and `octa_maze`
because the inserted obstacle blocks the route. After the obstacle is removed,
merged topometric maps recover reachability and improve the path ratio.

## Outputs

Each run writes to `python/benchmark_mms/output/<run_name>/`:

- `fig1_session_exploration.png`: session observations, trajectories, topo maps,
  and merged map.
- `fig2_optimality_curve.png`: cumulative path ratio curve.
- `fig3_reachability_coverage.png`: cumulative coverage in `m²`.
- `data/metrics.json`, `data/ratios.json`, `data/coverage.json`: numeric results.
- `data/topomap_k*.npz`, `data/topomap_merged_k*.npz`: exported topo graphs.

Generated output directories are ignored by git.

## Test

```bash
pytest python/benchmark_mms/tests/test_frontier_benchmark.py -q
```
