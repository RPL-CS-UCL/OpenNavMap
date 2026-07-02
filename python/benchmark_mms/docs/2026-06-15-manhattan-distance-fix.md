# Manhattan Distance Consistency Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all Euclidean distance computations in `frontier_explore_benchmark.py` with Manhattan distance, increase cross-session edge threshold to 5m, and remove the line-of-sight obstacle check for graph edge connections.

**Architecture:** All five changes are in one file and form a coherent unit — they convert the entire path length metric from Euclidean to Manhattan. A* uses 4-direction movement with Manhattan heuristic; topo graph edge weights use Manhattan; snap distances use Manhattan; cross-session edge check drops `_line_free` and uses Manhattan distance ≤ 5 m.

**Tech Stack:** Python 3.8+, numpy, networkx, heapq, pytest

---

## File Structure

| File | Role |
|------|------|
| `python/benchmark_mms/frontier_explore_benchmark.py` | Main script — 5 surgical edits across 5 functions |
| `python/benchmark_mms/tests/test_frontier_benchmark.py` | Existing test file — append 6 new tests |

---

### Task 1: Manhattan Distance — All Five Changes + Tests

**Files:**
- Modify: `python/benchmark_mms/frontier_explore_benchmark.py` — lines 43, 117-125, 337+342, 402-408, 428-429
- Modify: `python/benchmark_mms/tests/test_frontier_benchmark.py` — append 6 new tests

**Detailed changes:**

**Change A — `CROSS_DIST_M` (line 43):** `3.0` → `5.0`

**Change B — `astar` (lines 117-125):** Remove `diag = res * np.sqrt(2)` and diagonal directions. Replace dirs with 4-direction only: `[(-1,0,res), (1,0,res), (0,-1,res), (0,1,res)]`. Change initial heuristic from `res * np.hypot(...)` to `res * (abs(Δr) + abs(Δc))`. Change loop heuristic (line 143) to `res * (abs(nb[0]-goal[0]) + abs(nb[1]-goal[1]))`.

**Change C — `build_topometric_subgraph` edge weight (line 342):** `res * np.hypot(r - pr, c - pc)` → `res * (abs(r - pr) + abs(c - pc))`. Line 337 (translation threshold) stays Euclidean.

**Change D — `merge_topometric_graphs` (lines 402-408):** `np.hypot(xi-xj, yi-yj)` → `abs(xi-xj) + abs(yi-yj)`. Remove entire `_line_free` check block — just add the edge directly when distance < cross_dist.

**Change E — `topo_path_length` snap distances (lines 428-429):** `np.hypot(...)` → `abs(...) + abs(...)`.

**New test cases to append to test_frontier_benchmark.py:**

```python
def test_astar_distance_is_manhattan():
    """A* on a clear 10×10 grid must return Manhattan distance (4-dir only)."""
    grid = np.zeros((10, 10), dtype=np.uint8)
    _, dist = feb.astar(grid, (0, 0), (3, 4))
    expected = (3 + 4) * feb.GRID_RES_M
    assert abs(dist - expected) < 1e-9, f"Expected {expected}, got {dist}"


def test_astar_no_diagonal_movement():
    """A* must not cut diagonals — every step must be axis-aligned."""
    grid = np.zeros((10, 10), dtype=np.uint8)
    path, _ = feb.astar(grid, (0, 0), (2, 2))
    assert path is not None
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        assert (r0 == r1) or (c0 == c1), f"Diagonal step from ({r0},{c0}) to ({r1},{c1})"


def test_topo_subgraph_edge_weight_is_manhattan():
    """build_topometric_subgraph edge weight: Manhattan, not Euclidean."""
    # Δr=3, Δc=4 → Manhattan=(3+4)*0.5=3.5 m, Euclidean=hypot(3,4)*0.5=2.5 m
    # TRANS_THRESH_M=2.0 m, Manhattan 3.5 m > 2.0 m → triggers new node
    poses = [(0, 0, 0.0), (3, 4, 0.0)]
    G = feb.build_topometric_subgraph(poses, res=feb.GRID_RES_M)
    assert G.number_of_edges() == 1
    weight = list(G.edges(data=True))[0][2]["weight"]
    expected = (3 + 4) * feb.GRID_RES_M
    assert abs(weight - expected) < 1e-9, f"Expected {expected}, got {weight}"


def test_merge_connects_nodes_through_obstacle():
    """merge_topometric_graphs connects nodes even if base_grid has obstacle in between."""
    base_grid = np.zeros((5, 5), dtype=np.uint8)
    base_grid[2, 2] = 1
    G1 = nx.Graph(); G1.add_node(0, x=0.5, y=0.5, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=1.5, y=1.5, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], base_grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() >= 1, "Nodes across obstacle must be connected"


def test_merge_distance_threshold_is_5m():
    """merge_topometric_graphs connects nodes up to 5 m Manhattan distance."""
    base_grid = np.zeros((20, 20), dtype=np.uint8)
    G1 = nx.Graph(); G1.add_node(0, x=0.0, y=0.0, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=4.5, y=0.0, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], base_grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() >= 1, "Nodes 4.5 m apart must connect (threshold=5 m)"


def test_merge_does_not_connect_beyond_5m():
    """merge_topometric_graphs rejects nodes > 5 m Manhattan distance."""
    base_grid = np.zeros((20, 20), dtype=np.uint8)
    G1 = nx.Graph(); G1.add_node(0, x=0.0, y=0.0, yaw=0.0)
    G2 = nx.Graph(); G2.add_node(0, x=5.5, y=0.0, yaw=0.0)
    merged = feb.merge_topometric_graphs([G1, G2], base_grid, res=feb.GRID_RES_M)
    assert merged.number_of_edges() == 0, "Nodes > 5 m apart must NOT be connected"
```

**Commit message:** `fix: Manhattan distance throughout, 5m edge threshold, no line-of-sight check`
