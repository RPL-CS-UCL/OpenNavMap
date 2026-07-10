# PCD Dilate + Fixed Start/Goal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement.

**Goal:** Set `PCD_DILATE=1` (expand walls by 1 pixel on load), reduce `INFLATE_RADIUS=0` (dilate already provides safety), use fixed start `(7,5)` / goal `(62,65)` with proper world coordinates, and redo the benchmark.

**Architecture:** Four changes in `frontier_explore_benchmark.py` + update `fixed_pair.json` + run benchmark.

**Tech Stack:** Python 3.8+, numpy, scipy, networkx, pytest

---

## File Structure

| File | Role |
|------|------|
| `python/benchmark_mms/frontier_explore_benchmark.py` | 4 edits: line 44, line 53, lines 853-856, line 916 |
| `python/benchmark_mms/tests/test_frontier_benchmark.py` | Append 2 new constant tests |
| `python/benchmark_mms/output/octa_maze/fixed_pair.json` | Updated by benchmark re-run |

---

### Task: Dilate + Fixed Pair + Redo Experiment

- [ ] **Step 1: Write failing tests**

Append to `python/benchmark_mms/tests/test_frontier_benchmark.py`:

```python
def test_pcd_dilate_is_1():
    """PCD_DILATE must be 1 (walls expanded by 1 pixel on load)."""
    assert feb.PCD_DILATE == 1, f"Expected 1, got {feb.PCD_DILATE}"


def test_inflate_radius_is_0():
    """INFLATE_RADIUS must be 0 (dilated PCD already provides safety margin)."""
    assert feb.INFLATE_RADIUS == 0, f"Expected 0, got {feb.INFLATE_RADIUS}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Titan/code/robohike_ws/src/opennavmap && conda run -n opennavmap python -m pytest python/benchmark_mms/tests/test_frontier_benchmark.py -k "dilate or inflate_radius" -v
```

Expected: both FAIL (current PCD_DILATE=0, INFLATE_RADIUS=1).

- [ ] **Step 3: Apply four code changes**

**Change A** (line ~44, `INFLATE_RADIUS`):
```python
INFLATE_RADIUS = 0
```

**Change B** (line ~53, `PCD_DILATE`):
```python
PCD_DILATE = 1
```

**Change C** (lines ~853-856, `--start` / `--goal`):
Change `required=True` to `required=False` and add `default` values:
```python
    parser.add_argument("--start", type=int, nargs=2, default=[7, 5],
                        metavar=("R", "C"),
                        help="Fixed start cell (row, col) [default: 7 5]")
    parser.add_argument("--goal", type=int, nargs=2, default=[62, 65],
                        metavar=("R", "C"),
                        help="Fixed goal cell (row, col) [default: 62 65]")
```

**Change D** (line ~916, `world_goal` hardcode + add dynamic world_start):
Replace:
```python
        "world_start": [2.5, 2.0, 3.5], "world_goal": [32.0, 2.0, 32.5],
```
With (using `x_range` and `z_range` from `load_pcd_grid` return):
```python
        "world_start": [round(start[1] * res + float(x_range[0]), 1),
                        2.0,
                        round(start[0] * res + float(z_range[0]), 1)],
        "world_goal":  [round(goal[1] * res + float(x_range[0]), 1),
                        2.0,
                        round(goal[0] * res + float(z_range[0]), 1)],
```

Note: `x_range` and `z_range` are already in scope — `main()` calls `load_pcd_grid` which returns `(grid, x_range, z_range)`. Verify by reading the code; variable names may differ slightly (e.g., the return destructuring).

- [ ] **Step 4: Run all 17 tests**

```bash
cd /Titan/code/robohike_ws/src/opennavmap && conda run -n opennavmap python -m pytest python/benchmark_mms/tests/test_frontier_benchmark.py -v
```

Expected: all 17 tests PASS (15 existing + 2 new).

- [ ] **Step 5: Run full benchmark (no arguments — uses defaults)**

```bash
cd /Titan/code/robohike_ws/src/opennavmap && conda run -n opennavmap python python/benchmark_mms/frontier_explore_benchmark.py --k 5 --seed 42 2>&1
```

Expected: GT path ~60.0m, all 5 sessions reachable, fig1/fig2/fig3 generated.

- [ ] **Step 6: Commit**

```bash
cd /Titan/code/robohike_ws/src/opennavmap
git add python/benchmark_mms/frontier_explore_benchmark.py python/benchmark_mms/tests/test_frontier_benchmark.py
git commit -m "fix: PCD_DILATE=1, INFLATE_RADIUS=0, fixed start/goal (7,5)->(62,65)"
```
