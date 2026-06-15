# benchmark_mms — Frontier Exploration Benchmark

Multi-session frontier-based goal-directed exploration benchmark on the Octa Maze.

## Overview

K sessions each start from the same `(start, goal)` pair and explore using frontier
selection with per-session softmax-temperature perturbation. As sessions accumulate,
the topometric map path from start to goal converges toward the ground-truth optimal.

## Requirements

```bash
conda activate opennavmap
pip install numpy matplotlib scipy networkx
```

## Usage

```bash
cd python/benchmark_mms
python frontier_explore_benchmark.py
```

With explicit arguments:

```bash
python frontier_explore_benchmark.py \
    --start 7 5 --goal 62 65 \
    --k 5 --seed 42 \
    --output_dir output/octa_maze
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--start R C` | `7 5` | Start cell in grid coordinates (row, col) |
| `--goal R C` | `62 65` | Goal cell in grid coordinates (row, col) |
| `--k` | `5` | Number of exploration sessions |
| `--seed` | `42` | Master random seed |
| `--res_m` | `0.5` | Grid resolution (m/cell) |
| `--pcd` | `data/octa_maze.pcd` | Path to point cloud file |
| `--output_dir` | `output/octa_maze` | Output directory |

## Output Files

All outputs written to `output/octa_maze/`:

| File | Description |
|------|-------------|
| `fig1_session_exploration.png` | Per-session topo graph + merged exploration map |
| `fig2_optimality_curve.png` | Path optimality ratio per cumulative session count |
| `fig3_reachability_coverage.png` | Cumulative coverage growth with per-session trajectory |
| `fixed_pair.json` | Fixed (start, goal) metadata and GT path length |
| `data/metrics.json` | Per-session metrics (ratio, nodes, traj length) |

## Data

`data/octa_maze.pcd` — 250k-point ASCII PCD of a 35×35 m octagonal maze.

Grid parameters after loading:
- Resolution: 0.5 m/cell → 71×71 grid
- Height slice: Y = 2.0 ± 0.3 m
- Obstacle ratio: ~29% (no pre-dilation)

## World Coordinates

The PCD uses XZ as the ground plane (Y = height):

| Point | World (x, y, z) | Grid (row, col) |
|-------|-----------------|-----------------|
| Start | (2.5, 2.0, 3.5) | (7, 5) |
| Goal  | (32.5, 2.0, 31.0) | (62, 65) |

## Session Diversity

All sessions share the same `(start, goal)` pair. Exploration diversity is achieved via:

- **Initial yaw**: `k × (2π / K)` — session 0 faces east, each subsequent session rotates
- **Frontier temperature** `T_k`: linearly from `T_min=0.5` (greedy) to `T_max=5.0`
  (near-uniform), controlled via softmax over Euclidean distance to frontier candidates

## Run Tests

```bash
cd /path/to/opennavmap   # repo root
conda run -n opennavmap python -m pytest python/benchmark_mms/tests/ -v
```

## Design Notes

See `docs/2025-06-13-frontier-goal-directed-exploration-design.md` for the full
algorithm specification including frontier exploration pseudocode, topometric graph
construction, and path optimality evaluation.
