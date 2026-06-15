# A* Edge Weight Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement.

**Goal:** Replace straight-line Manhattan edge weights in `build_topometric_subgraph` and `merge_topometric_graphs` with actual A* path lengths on the inflated occupancy grid, so `topo_path_len >= GT_path_len` always holds (ratio >= 1.0).

**Architecture:** Two functions modified: (1) `build_topometric_subgraph` gets optional `base_grid` parameter, uses A* when provided; (2) `merge_topometric_graphs` computes A* path length between cross-session node pairs, skipping unreachable ones.

**Tech Stack:** Python 3.8+, numpy, heapq, networkx, scipy

---

## File Structure

| File | Role |
|------|------|
| `python/benchmark_mms/frontier_explore_benchmark.py` | Modify `build_topometric_subgraph` (line ~340), `merge_topometric_graphs` (lines ~400-403), `main()` call site (line ~957) |
| `python/benchmark_mms/tests/test_frontier_benchmark.py` | Append 3 new tests |
