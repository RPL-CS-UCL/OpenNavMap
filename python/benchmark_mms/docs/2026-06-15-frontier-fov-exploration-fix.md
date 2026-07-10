# Frontier Exploration Fix — FOV, Exploration Strategy, Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix FOV parameters (60deg/5m), session exploration coverage (session 0-2 unreachable), and observation-based visualization colors in `frontier_explore_benchmark.py`.

**Architecture:** Three independent fixes in one file: (1) reduce FOV constants, (2) add goal-directed frontier bias so low-temperature sessions still explore toward the goal, (3) replace `base_grid`-background in fig1/fig3 with obs-based rendering (grey=unknown, white=free, black=obstacle).

**Tech Stack:** Python 3.8+, numpy, scipy, matplotlib, networkx, pytest

---

## File Structure

| File | Role |
|------|------|
| `python/benchmark_mms/frontier_explore_benchmark.py` | Main script — modify constants (lines 38-40), `select_frontier` (lines 252-308), `fig1_session_exploration` (lines 586-700), `fig3_reachability_coverage` (lines 742-822) |
| `python/benchmark_mms/tests/test_frontier_benchmark.py` | New test file — unit tests for FOV, frontier selection bias, and obs rendering helpers |

---

### Task 1: Update FOV Constants and Test

**Files:**
- Modify: `python/benchmark_mms/frontier_explore_benchmark.py:38-40`
- Create: `python/benchmark_mms/tests/test_frontier_benchmark.py`

**Steps:**

1. Create `python/benchmark_mms/tests/test_frontier_benchmark.py` with test content (see below)
2. Run tests to confirm they fail (FOV_HALF_DEG is currently 45.0, FOV_RANGE_M is 8.0)
3. Update constants in `frontier_explore_benchmark.py`: FOV_HALF_DEG=30.0, FOV_RANGE_M=5.0 (FOV_HALF_RAD auto-computed)
4. Run tests to confirm they pass
5. Commit

**Test file content (test_frontier_benchmark.py):**

```python
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmark_mms.frontier_explore_benchmark as feb


def test_fov_half_deg_is_30():
    """Camera FOV must be 60 degrees full-angle (30 degrees half-angle)."""
    assert feb.FOV_HALF_DEG == 30.0, f"Expected 30.0, got {feb.FOV_HALF_DEG}"


def test_fov_range_m_is_5():
    """Camera maximum depth must be 5 metres."""
    assert feb.FOV_RANGE_M == 5.0, f"Expected 5.0, got {feb.FOV_RANGE_M}"


def test_fov_half_rad_consistent_with_deg():
    """FOV_HALF_RAD must equal np.radians(FOV_HALF_DEG)."""
    assert abs(feb.FOV_HALF_RAD - np.radians(30.0)) < 1e-9
```

**Constant changes in frontier_explore_benchmark.py:**
- Line 38: `FOV_HALF_DEG = 30.0` (was 45.0)
- Line 39: `FOV_HALF_RAD = np.radians(FOV_HALF_DEG)` (unchanged, auto-updates)
- Line 40: `FOV_RANGE_M = 5.0` (was 8.0)

---

### Task 2: Goal-Directed Frontier Bias (Fix Session 0-2 Exploration)

**Root cause:** Low-temperature sessions (T=0.5) always pick the nearest frontier using Euclidean distance, so the robot stays near the start indefinitely. The fix adds a **goal-direction bias** as an additive bonus to the softmax logit: frontiers whose free-neighbor points toward the goal receive a small positive logit boost, nudging selection toward the goal over time.

**Files:**
- Modify: `python/benchmark_mms/frontier_explore_benchmark.py:252-308` (select_frontier)
- Modify: `python/benchmark_mms/frontier_explore_benchmark.py` around line 496 (call site in frontier_explore_session)
- Modify: `python/benchmark_mms/tests/test_frontier_benchmark.py` (add test cases)

**Steps:**

1. Add test cases to `test_frontier_benchmark.py` for goal-directed selection (see below)
2. Run tests to confirm they fail (old select_frontier doesn't accept `goal` parameter)
3. Replace `select_frontier` function with new implementation (see below)
4. Update the call site in `frontier_explore_session` to pass `goal=goal, goal_bias=0.5`
5. Run all tests to confirm they pass
6. Run full benchmark to verify sessions reach goal more often
7. Commit

**New select_frontier implementation (complete replacement of lines 252-308):**

```python
def select_frontier(
    frontiers: list[tuple[int, int]],
    current: tuple[int, int],
    obs: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
    top_n: int,
    inf_pg: np.ndarray,
    res: float = GRID_RES_M,
    frontier_free_neighbors: list[tuple[int, int]] | None = None,
    goal: tuple[int, int] | None = None,
    goal_bias: float = 0.5,
) -> tuple[int, int] | None:
    if not frontiers:
        return None

    f_arr = np.array(frontiers)
    cr, cc = current
    eucl_dists = np.abs(f_arr[:, 0] - cr) + np.abs(f_arr[:, 1] - cc)
    order = np.argsort(eucl_dists)
    top_k = min(top_n, len(frontiers))

    targets = []
    for idx in order[:top_k]:
        fr, fc = frontiers[idx]
        tgt = frontier_free_neighbors[idx] if frontier_free_neighbors is not None else (fr, fc)
        if tgt == (cr, cc):
            continue
        targets.append(tgt)

    if not targets:
        return _fallback_select(order, frontiers, frontier_free_neighbors, inf_pg, current, res)

    tgt_arr = np.array(targets)
    eucl_to_targets = np.abs(tgt_arr[:, 0] - cr) + np.abs(tgt_arr[:, 1] - cc)
    logits = -eucl_to_targets / max(temperature, 1e-6)

    if goal is not None:
        goal_r, goal_c = goal
        dist_to_goal = max(np.hypot(goal_r - cr, goal_c - cc), 1e-6)
        dot = ((tgt_arr[:, 0] - cr) * (goal_r - cr) +
               (tgt_arr[:, 1] - cc) * (goal_c - cc))
        direction_score = dot / dist_to_goal
        direction_score /= max(np.abs(direction_score).max(), 1e-6)
        logits += goal_bias * direction_score

    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    chosen_idx = rng.choice(len(targets), p=probs)

    _, length = astar(inf_pg, current,
                      (int(targets[chosen_idx][0]), int(targets[chosen_idx][1])), res)
    if length >= float("inf"):
        return _fallback_select(order, frontiers, frontier_free_neighbors, inf_pg, current, res)

    return (int(targets[chosen_idx][0]), int(targets[chosen_idx][1]))
```

**Call site update (in frontier_explore_session, replace the select_frontier call):**

```python
        next_f = select_frontier(frontiers, (r, c), obs, rng,
                                 frontier_temperature, top_n, inf_pg, res,
                                 frontier_free_neighbors=free_neighbors,
                                 goal=goal, goal_bias=0.5)
```

**New test cases (add to test_frontier_benchmark.py):**

```python
def _make_simple_grid(h=15, w=15):
    return np.zeros((h, w), dtype=np.uint8)


def test_select_frontier_prefers_goal_direction():
    rng = np.random.default_rng(0)
    frontiers = [(2, 7), (10, 7)]
    free_neighbors = [(3, 7), (9, 7)]
    current = (5, 7)
    goal = (12, 7)
    obs = np.full((15, 15), -1, dtype=np.int8)
    inf_pg = np.zeros((15, 15), dtype=np.uint8)

    toward_count = 0
    for _ in range(200):
        result = feb.select_frontier(
            frontiers, current, obs, rng,
            temperature=0.5, top_n=5, inf_pg=inf_pg,
            frontier_free_neighbors=free_neighbors,
            goal=goal, goal_bias=0.5,
        )
        if result == (9, 7):
            toward_count += 1

    assert toward_count > 140, f"Expected >140/200 toward-goal, got {toward_count}"


def test_select_frontier_no_bias_behaves_as_nearest():
    rng = np.random.default_rng(42)
    frontiers = [(5, 6), (10, 6)]
    free_neighbors = [(5, 7), (10, 7)]
    current = (5, 8)
    obs = np.full((15, 15), -1, dtype=np.int8)
    inf_pg = np.zeros((15, 15), dtype=np.uint8)

    nearest_count = 0
    for _ in range(200):
        result = feb.select_frontier(
            frontiers, current, obs, rng,
            temperature=0.5, top_n=5, inf_pg=inf_pg,
            frontier_free_neighbors=free_neighbors,
            goal=None, goal_bias=0.5,
        )
        if result == (5, 7):
            nearest_count += 1

    assert nearest_count > 150, f"Expected >150/200 nearest, got {nearest_count}"
```

---

### Task 3: Observation-Based Visualization (Grey/White/Black Background)

**Spec:** Each session subplot shows the robot's **partial observation** as background:
- Grey (`#6B7280`): unknown / not yet observed (obs == 0)
- White (`#F3F4F6`): observed free (obs == -1)
- Black (`#1F2937`): observed obstacle (obs == 1)

**Files:**
- Modify: `python/benchmark_mms/frontier_explore_benchmark.py` — add `STYLE_UNKNOWN` constant and `obs_to_rgb` helper (near line 530), update `fig1_session_exploration` per-session background (line 610), update `fig3_reachability_coverage` background (lines 764-774)
- Modify: `python/benchmark_mms/tests/test_frontier_benchmark.py` (add obs_to_rgb tests)

**Steps:**

1. Add test cases for `obs_to_rgb` (see below)
2. Run tests to confirm they fail (obs_to_rgb not defined)
3. Add `STYLE_UNKNOWN` constant and `obs_to_rgb` helper
4. Run helper tests to confirm they pass
5. Update fig1 per-session background to use obs_to_rgb
6. Update fig3 background to use obs-based rendering (grey/white/black with cyan overlay for new coverage this session)
7. Run all tests to confirm they pass
8. Run full benchmark
9. Commit

**New constant (after line 530 STYLE_OBS):**

```python
STYLE_UNKNOWN = np.array([107, 114, 128]) / 255.0
```

**New helper function (after constant block):**

```python
def obs_to_rgb(obs: np.ndarray) -> np.ndarray:
    H, W = obs.shape
    rgb = np.empty((H, W, 3), dtype=np.float32)
    rgb[:] = STYLE_UNKNOWN
    rgb[obs == -1] = STYLE_FREE
    rgb[obs == 1] = STYLE_OBS
    return rgb
```

**fig1 update:** In the per-session for-loop (around line 610), replace `_draw_base_grid(ax, base_grid)` with:
```python
        ax.imshow(obs_to_rgb(all_obs[k]), origin="upper", interpolation="none")
```

**fig3 update:** In the per-session for-loop (around lines 762-774), replace the background drawing with:
```python
        ax.set_facecolor(BG_COLOR)
        ax.imshow(obs_to_rgb(all_obs[k]), origin="upper", interpolation="none")

        prev_mask = cum_free_mask.copy()
        new_free = (all_obs[k] == -1)
        cum_free_mask |= new_free

        new_this_session = new_free & ~prev_mask
        if new_this_session.any():
            cov_layer = np.zeros((*base_grid.shape, 4))
            cov_layer[new_this_session, :] = (*_COV_NEW_RGB, ALPHA_COV_NEW)
            ax.imshow(cov_layer, origin="upper", interpolation="none", zorder=2)
```

**New test cases (add to test_frontier_benchmark.py):**

```python
def test_obs_to_rgb_unknown_is_grey():
    obs = np.zeros((5, 5), dtype=np.int8)
    rgb = feb.obs_to_rgb(obs)
    expected = np.array([107, 114, 128]) / 255.0
    np.testing.assert_allclose(rgb[2, 2], expected, atol=1e-6)


def test_obs_to_rgb_free_is_white():
    obs = np.full((5, 5), -1, dtype=np.int8)
    rgb = feb.obs_to_rgb(obs)
    expected = np.array([243, 244, 246]) / 255.0
    np.testing.assert_allclose(rgb[2, 2], expected, atol=1e-6)


def test_obs_to_rgb_obstacle_is_black():
    obs = np.full((5, 5), 1, dtype=np.int8)
    rgb = feb.obs_to_rgb(obs)
    expected = np.array([31, 41, 55]) / 255.0
    np.testing.assert_allclose(rgb[2, 2], expected, atol=1e-6)


def test_obs_to_rgb_mixed():
    obs = np.zeros((3, 3), dtype=np.int8)
    obs[0, 0] = -1
    obs[1, 1] = 1
    rgb = feb.obs_to_rgb(obs)
    white = np.array([243, 244, 246]) / 255.0
    black = np.array([31, 41, 55]) / 255.0
    grey  = np.array([107, 114, 128]) / 255.0
    np.testing.assert_allclose(rgb[0, 0], white, atol=1e-6)
    np.testing.assert_allclose(rgb[1, 1], black, atol=1e-6)
    np.testing.assert_allclose(rgb[2, 2], grey,  atol=1e-6)
```
