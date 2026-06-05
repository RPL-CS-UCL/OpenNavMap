# benchmark_mms Multi-Session Mapping 评测 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `python/benchmark_mms/` 下实现三个离线评测脚本，量化单会话 vs 多会话建图在 Matterport3D 上的导航增益（可达性、路径最优性、长期更新），对应论文 Q5/E5。

**Architecture:** 假设各 $M_k$ 合并地图已由 `map_merge_pipeline.py` 离线生成；本模块只加载 `PointGraph`（`trav` 边）、固定 task pairs、调用 `dijk_shortest_path` 计算 metric，不依赖 ROS。GT 测地距离从 Matterport3D `.ply` mesh 用 `trimesh` 计算。

**Tech Stack:** Python 3.8, `trimesh`, `numpy`, `matplotlib`, `json`; 复用 `python/utils/utils_shortest_path.py` 和 `python/point_graph.py`。

---

## 文件规划

| 文件 | 职责 |
|------|------|
| `python/benchmark_mms/__init__.py` | 空包标记 |
| `python/benchmark_mms/REQUIREMENTS.md` | 需求说明（已存在） |
| `python/benchmark_mms/parser.py` | 公共 argparse |
| `python/benchmark_mms/dataloader.py` | 加载 PointGraph、task pairs、GT 测地距离 |
| `python/benchmark_mms/evaluation.py` | 三类 metric 计算函数 |
| `python/benchmark_mms/run_reachability.py` | E5-A1 入口：可达比例 vs k |
| `python/benchmark_mms/run_path_optimality.py` | E5-A2+A3 入口：路径长/GT 比值 + 拓扑对照 |
| `python/benchmark_mms/run_longterm.py` | E5-B4 入口：通路变化 before/after |
| `python/test/test_benchmark_mms.py` | 单元测试 |
| `scripts/run_benchmark_mms_reachability.sh` | shell 入口 A1 |
| `scripts/run_benchmark_mms_path_optimality.sh` | shell 入口 A2 |
| `scripts/run_benchmark_mms_longterm.sh` | shell 入口 B4 |

---

## Task 1: 包初始化 + parser.py

**Files:**
- Create: `python/benchmark_mms/__init__.py`
- Create: `python/benchmark_mms/parser.py`

- [ ] **步骤 1：创建 `__init__.py`**

```python
# python/benchmark_mms/__init__.py
```

- [ ] **步骤 2：创建 `parser.py`**

```python
# python/benchmark_mms/parser.py
import argparse
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--merged_map_dirs",
        type=str,
        nargs="+",
        required=True,
        help="有序列表：M_1, M_2, ..., M_K 的合并地图目录（对应 k=1..K）",
    )
    parser.add_argument(
        "--task_pairs_file",
        type=str,
        required=True,
        help="JSON 文件，包含固定的 (start_node_id, goal_node_id) 列表",
    )
    parser.add_argument(
        "--gt_mesh_path",
        type=str,
        default=None,
        help="Matterport3D GT mesh .ply 路径（用于测地距离；为 None 时跳过 GT 比值）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出目录（CSV + 图表）",
    )
    parser.add_argument(
        "--edge_type",
        type=str,
        default="trav",
        choices=["trav", "odom", "covis"],
        help="用于规划的图边类型",
    )
    parser.add_argument("--debug", action="store_true", help="打印详细日志")
    return parser.parse_args()
```

- [ ] **步骤 3：提交**

```bash
git add python/benchmark_mms/__init__.py python/benchmark_mms/parser.py
git commit -m "feat(mms): add benchmark_mms package and parser"
```

---

## Task 2: dataloader.py

**Files:**
- Create: `python/benchmark_mms/dataloader.py`
- Test: `python/test/test_benchmark_mms.py`（Task 5 统一测试，此处先写接口）

- [ ] **步骤 1：创建 `dataloader.py`**

```python
# python/benchmark_mms/dataloader.py
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import numpy as np
from pathlib import Path
from typing import Optional

from point_graph import PointGraphLoader, PointGraph


def load_graph(map_dir: str, edge_type: str = "trav") -> PointGraph:
    """从已合并地图目录加载 PointGraph。"""
    return PointGraphLoader.load_data(Path(map_dir), edge_type)


def load_task_pairs(task_pairs_file: str) -> list[tuple[int, int]]:
    """加载固定的 (start_node_id, goal_node_id) 列表。

    JSON 格式：[{"start": 0, "goal": 42}, ...]
    """
    with open(task_pairs_file, "r") as f:
        raw = json.load(f)
    return [(item["start"], item["goal"]) for item in raw]


def generate_task_pairs(
    graph: PointGraph,
    n_pairs: int,
    seed: int = 42,
    min_distance: float = 5.0,
) -> list[tuple[int, int]]:
    """在图中随机采样 n_pairs 对 (start, goal)，要求两点欧氏距离 >= min_distance。

    用于首次生成固定任务集合并保存为 JSON。
    """
    rng = np.random.default_rng(seed)
    node_ids = list(graph.nodes.keys())
    pairs: list[tuple[int, int]] = []
    max_attempts = n_pairs * 100
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        s_id, g_id = rng.choice(node_ids, size=2, replace=False)
        s_node = graph.get_node(int(s_id))
        g_node = graph.get_node(int(g_id))
        dist = float(np.linalg.norm(s_node.trans - g_node.trans))
        if dist >= min_distance:
            pairs.append((int(s_id), int(g_id)))
        attempts += 1
    return pairs


def save_task_pairs(pairs: list[tuple[int, int]], out_file: str) -> None:
    """将 task pairs 保存为 JSON。"""
    data = [{"start": s, "goal": g} for s, g in pairs]
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)


def load_gt_geodesic(
    mesh_path: str,
    pos_start: np.ndarray,
    pos_goal: np.ndarray,
) -> float:
    """用 trimesh 计算两 3D 点在 mesh 表面上的测地最短路长度（米）。

    若 trimesh 未安装或 mesh_path 为 None，返回 np.nan。
    """
    if mesh_path is None:
        return np.nan
    try:
        import trimesh
        import trimesh.graph
    except ImportError:
        return np.nan

    mesh = trimesh.load(mesh_path, process=False)
    # 将 3D 位置映射到 mesh 最近顶点
    _, idx_s = trimesh.proximity.closest_point(mesh, pos_start.reshape(1, 3))
    _, idx_g = trimesh.proximity.closest_point(mesh, pos_goal.reshape(1, 3))
    # 构造顶点邻接图，用 Dijkstra 求测地距离
    adjacency = trimesh.graph.vertex_adjacency_graph(mesh)
    import networkx as nx
    try:
        dist = nx.shortest_path_length(
            adjacency, source=int(idx_s[0]), target=int(idx_g[0]), weight="weight"
        )
    except nx.NetworkXNoPath:
        dist = np.nan
    return float(dist)
```

- [ ] **步骤 2：提交**

```bash
git add python/benchmark_mms/dataloader.py
git commit -m "feat(mms): add dataloader for graph, task pairs, GT geodesic"
```

---

## Task 3: evaluation.py

**Files:**
- Create: `python/benchmark_mms/evaluation.py`

- [ ] **步骤 1：创建 `evaluation.py`**

```python
# python/benchmark_mms/evaluation.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import numpy as np
from typing import Optional

from point_graph import PointGraph
from utils.utils_shortest_path import dijk_shortest_path
from utils.base_node import BaseNode


def compute_reachability(
    graph: PointGraph,
    task_pairs: list[tuple[int, int]],
) -> dict:
    """计算可达比例。

    Returns:
        {
            "reachable_ratio": float,   # [0, 1]
            "reachable_count": int,
            "total_count": int,
        }
    """
    reachable = 0
    total = 0
    for start_id, goal_id in task_pairs:
        s = graph.get_node(start_id)
        g = graph.get_node(goal_id)
        if s is None or g is None:
            continue
        total += 1
        dist, _ = dijk_shortest_path(graph, s, g)
        if dist < float("inf"):
            reachable += 1
    ratio = reachable / total if total > 0 else 0.0
    return {"reachable_ratio": ratio, "reachable_count": reachable, "total_count": total}


def _topological_path_length(graph: PointGraph, path: list[BaseNode]) -> float:
    """将 path 节点列表转换为物理路径长度（米）。"""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += float(np.linalg.norm(path[i].trans - path[i + 1].trans))
    return total


def compute_path_optimality(
    graph_metric: PointGraph,
    graph_topological: PointGraph,
    task_pairs: list[tuple[int, int]],
    gt_geodesics: Optional[list[float]] = None,
) -> dict:
    """计算路径最优性（metric 规划 vs 拓扑规划 vs GT 测地距离）。

    graph_metric:       物理距离加权边的图（用于度量规划）
    graph_topological:  跳数加权边的图（用于拓扑规划）；若为 None 则跳过

    Returns:
        {
            "node_count": int,
            "metric_path_lengths": list[float],     # 每对任务的度量路径长
            "topo_path_lengths": list[float],        # 拓扑路径物理长
            "gt_geodesics": list[float],             # GT 测地距离（nan 若不可用）
            "metric_ratio_mean": float,              # mean(metric/GT)，排除 nan 和 inf
            "topo_ratio_mean": float,
        }
    """
    node_count = graph_metric.get_num_node()
    metric_lengths: list[float] = []
    topo_lengths: list[float] = []
    gts = gt_geodesics if gt_geodesics is not None else [np.nan] * len(task_pairs)

    for i, (start_id, goal_id) in enumerate(task_pairs):
        s = graph_metric.get_node(start_id)
        g = graph_metric.get_node(goal_id)
        if s is None or g is None:
            metric_lengths.append(np.nan)
            topo_lengths.append(np.nan)
            continue

        metric_dist, _ = dijk_shortest_path(graph_metric, s, g)
        metric_lengths.append(float(metric_dist))

        if graph_topological is not None:
            s_t = graph_topological.get_node(start_id)
            g_t = graph_topological.get_node(goal_id)
            if s_t is not None and g_t is not None:
                _, topo_path = dijk_shortest_path(graph_topological, s_t, g_t)
                topo_lengths.append(_topological_path_length(graph_metric, [
                    graph_metric.get_node(n.id) for n in topo_path
                    if graph_metric.get_node(n.id) is not None
                ]))
            else:
                topo_lengths.append(np.nan)
        else:
            topo_lengths.append(np.nan)

    def _safe_ratio(lengths: list[float], geodesics: list[float]) -> float:
        ratios = []
        for l, g in zip(lengths, geodesics):
            if np.isnan(l) or np.isnan(g) or g == 0 or l == float("inf"):
                continue
            ratios.append(l / g)
        return float(np.mean(ratios)) if ratios else np.nan

    return {
        "node_count": node_count,
        "metric_path_lengths": metric_lengths,
        "topo_path_lengths": topo_lengths,
        "gt_geodesics": list(gts),
        "metric_ratio_mean": _safe_ratio(metric_lengths, list(gts)),
        "topo_ratio_mean": _safe_ratio(topo_lengths, list(gts)),
    }


def compute_longterm_delta(
    graph_before: PointGraph,
    graph_after: PointGraph,
    task_pairs: list[tuple[int, int]],
) -> dict:
    """计算通路变化前后的 path length 变化与新增可达 goal 数。

    Returns:
        {
            "path_lengths_before": list[float],
            "path_lengths_after": list[float],
            "new_reachable_count": int,     # after 可达但 before 不可达的任务数
            "path_length_delta_mean": float, # mean(after - before)，仅统计两者均可达的对
        }
    """
    lengths_before: list[float] = []
    lengths_after: list[float] = []
    new_reachable = 0

    for start_id, goal_id in task_pairs:
        sb = graph_before.get_node(start_id)
        gb = graph_before.get_node(goal_id)
        sa = graph_after.get_node(start_id)
        ga = graph_after.get_node(goal_id)

        d_before = float("inf")
        d_after = float("inf")

        if sb is not None and gb is not None:
            d_before, _ = dijk_shortest_path(graph_before, sb, gb)
        if sa is not None and ga is not None:
            d_after, _ = dijk_shortest_path(graph_after, sa, ga)

        if d_before == float("inf") and d_after < float("inf"):
            new_reachable += 1

        lengths_before.append(float(d_before))
        lengths_after.append(float(d_after))

    both_reachable = [
        (b, a) for b, a in zip(lengths_before, lengths_after)
        if b < float("inf") and a < float("inf")
    ]
    delta_mean = float(np.mean([a - b for b, a in both_reachable])) if both_reachable else np.nan

    return {
        "path_lengths_before": lengths_before,
        "path_lengths_after": lengths_after,
        "new_reachable_count": new_reachable,
        "path_length_delta_mean": delta_mean,
    }
```

- [ ] **步骤 2：提交**

```bash
git add python/benchmark_mms/evaluation.py
git commit -m "feat(mms): add evaluation metrics (reachability, path optimality, longterm)"
```

---

## Task 4: 三个入口脚本

**Files:**
- Create: `python/benchmark_mms/run_reachability.py`
- Create: `python/benchmark_mms/run_path_optimality.py`
- Create: `python/benchmark_mms/run_longterm.py`

- [ ] **步骤 1：创建 `run_reachability.py`**

```python
# python/benchmark_mms/run_reachability.py
"""
Usage:
  python python/benchmark_mms/run_reachability.py \
    --merged_map_dirs /data/M_1 /data/M_2 /data/M_3 \
    --task_pairs_file /data/task_pairs.json \
    --out_dir /data/results/reachability \
    --edge_type trav
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import csv
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from benchmark_mms.parser import parse_arguments
from benchmark_mms.dataloader import load_graph, load_task_pairs
from benchmark_mms.evaluation import compute_reachability


def main() -> None:
    args = parse_arguments()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    task_pairs = load_task_pairs(args.task_pairs_file)
    results = []

    for k, map_dir in enumerate(args.merged_map_dirs, start=1):
        graph = load_graph(map_dir, args.edge_type)
        metrics = compute_reachability(graph, task_pairs)
        metrics["k"] = k
        metrics["map_dir"] = map_dir
        results.append(metrics)
        print(f"k={k}: reachable_ratio={metrics['reachable_ratio']:.3f} "
              f"({metrics['reachable_count']}/{metrics['total_count']})")

    # 保存 CSV
    csv_path = Path(args.out_dir) / "reachability.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "reachable_ratio", "reachable_count", "total_count", "map_dir"])
        writer.writeheader()
        writer.writerows(results)

    # 绘图
    ks = [r["k"] for r in results]
    ratios = [r["reachable_ratio"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, ratios, marker="o")
    ax.set_xlabel("Sessions k")
    ax.set_ylabel("Reachable Goal Ratio")
    ax.set_title("Reachability vs Number of Sessions")
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(Path(args.out_dir) / "reachability.pdf"))
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：创建 `run_path_optimality.py`**

```python
# python/benchmark_mms/run_path_optimality.py
"""
Usage:
  python python/benchmark_mms/run_path_optimality.py \
    --merged_map_dirs /data/M_1 /data/M_2 /data/M_3 \
    --task_pairs_file /data/task_pairs.json \
    --gt_mesh_path /data/scene.ply \
    --out_dir /data/results/path_optimality \
    --edge_type trav
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import csv
import copy
import numpy as np
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from benchmark_mms.parser import parse_arguments
from benchmark_mms.dataloader import load_graph, load_task_pairs, load_gt_geodesic
from benchmark_mms.evaluation import compute_path_optimality
from utils.utils_shortest_path import dijk_shortest_path


def _make_topological_graph(graph):
    """复制图并将所有边权重置为 1（跳数）。"""
    import copy
    g_topo = copy.deepcopy(graph)
    for node in g_topo.nodes.values():
        new_edges = {}
        for key, (neighbor, _) in node._edges.items():
            topo_neighbor = g_topo.get_node(neighbor.id)
            if topo_neighbor is not None:
                new_edges[key] = (topo_neighbor, 1.0)
        node._edges = new_edges
    return g_topo


def main() -> None:
    args = parse_arguments()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    task_pairs = load_task_pairs(args.task_pairs_file)
    results = []

    # 第一次循环：从第一张图采样 GT 测地距离（节点位置已知）
    first_graph = load_graph(args.merged_map_dirs[0], args.edge_type)
    gt_geodesics: list[float] = []
    for start_id, goal_id in task_pairs:
        s = first_graph.get_node(start_id)
        g = first_graph.get_node(goal_id)
        if s is None or g is None:
            gt_geodesics.append(float("nan"))
        else:
            gt_geodesics.append(load_gt_geodesic(args.gt_mesh_path, s.trans, g.trans))

    for k, map_dir in enumerate(args.merged_map_dirs, start=1):
        graph = load_graph(map_dir, args.edge_type)
        graph_topo = _make_topological_graph(graph)
        metrics = compute_path_optimality(graph, graph_topo, task_pairs, gt_geodesics)
        metrics["k"] = k
        results.append(metrics)
        print(f"k={k}: nodes={metrics['node_count']}, "
              f"metric_ratio={metrics['metric_ratio_mean']:.3f}, "
              f"topo_ratio={metrics['topo_ratio_mean']:.3f}")

    # 保存 CSV
    csv_path = Path(args.out_dir) / "path_optimality.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "node_count", "metric_ratio_mean", "topo_ratio_mean"])
        writer.writeheader()
        writer.writerows([{k: v for k, v in r.items() if k in ["k", "node_count", "metric_ratio_mean", "topo_ratio_mean"]} for r in results])

    # 绘图
    ks = [r["k"] for r in results]
    metric_ratios = [r["metric_ratio_mean"] for r in results]
    topo_ratios = [r["topo_ratio_mean"] for r in results]
    node_counts = [r["node_count"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ks, metric_ratios, marker="o", label="Metric planning")
    ax1.plot(ks, topo_ratios, marker="s", linestyle="--", label="Topological planning")
    ax1.axhline(1.0, color="gray", linestyle=":", label="GT optimal")
    ax1.set_xlabel("Sessions k")
    ax1.set_ylabel("Path Length / GT Geodesic")
    ax1.set_title("Path Optimality vs Sessions")
    ax1.legend()
    ax1.grid(True)

    ax2.bar(ks, node_counts)
    ax2.set_xlabel("Sessions k")
    ax2.set_ylabel("Node Count")
    ax2.set_title("Map Size vs Sessions")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(str(Path(args.out_dir) / "path_optimality.pdf"))
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 3：创建 `run_longterm.py`**

```python
# python/benchmark_mms/run_longterm.py
"""
Usage:
  python python/benchmark_mms/run_longterm.py \
    --merged_map_dirs /data/epoch1_M_k /data/epoch2_M_k \
    --task_pairs_file /data/task_pairs.json \
    --out_dir /data/results/longterm \
    --edge_type trav

merged_map_dirs 须恰好为 2 项：[before, after]
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import csv
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from benchmark_mms.parser import parse_arguments
from benchmark_mms.dataloader import load_graph, load_task_pairs
from benchmark_mms.evaluation import compute_longterm_delta


def main() -> None:
    args = parse_arguments()
    if len(args.merged_map_dirs) != 2:
        raise ValueError("run_longterm.py 需要恰好 2 个 --merged_map_dirs：[before, after]")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    task_pairs = load_task_pairs(args.task_pairs_file)
    graph_before = load_graph(args.merged_map_dirs[0], args.edge_type)
    graph_after = load_graph(args.merged_map_dirs[1], args.edge_type)

    metrics = compute_longterm_delta(graph_before, graph_after, task_pairs)
    print(f"New reachable goals after update: {metrics['new_reachable_count']}")
    print(f"Mean path length delta (after - before): {metrics['path_length_delta_mean']:.3f} m")

    # 保存 CSV
    csv_path = Path(args.out_dir) / "longterm.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "path_length_before", "path_length_after", "delta"])
        writer.writeheader()
        for i, (b, a) in enumerate(zip(metrics["path_lengths_before"], metrics["path_lengths_after"])):
            writer.writerow({"task_id": i, "path_length_before": b, "path_length_after": a, "delta": a - b})

    # 绘图：before/after 对比散点
    fig, ax = plt.subplots(figsize=(6, 5))
    finite_before = [b for b, a in zip(metrics["path_lengths_before"], metrics["path_lengths_after"])
                     if b < float("inf") and a < float("inf")]
    finite_after = [a for b, a in zip(metrics["path_lengths_before"], metrics["path_lengths_after"])
                    if b < float("inf") and a < float("inf")]
    ax.scatter(finite_before, finite_after, alpha=0.6)
    lim = max(max(finite_before, default=1), max(finite_after, default=1)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", label="no change")
    ax.set_xlabel("Path Length Before (m)")
    ax.set_ylabel("Path Length After (m)")
    ax.set_title(f"Long-term Update: new_reachable={metrics['new_reachable_count']}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(Path(args.out_dir) / "longterm.pdf"))
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：提交**

```bash
git add python/benchmark_mms/run_reachability.py \
        python/benchmark_mms/run_path_optimality.py \
        python/benchmark_mms/run_longterm.py
git commit -m "feat(mms): add three benchmark entry scripts (reachability, path_optimality, longterm)"
```

---

## Task 5: 单元测试

**Files:**
- Create: `python/test/test_benchmark_mms.py`

- [ ] **步骤 1：编写测试（用小型内存图，不需要真实数据集）**

```python
# python/test/test_benchmark_mms.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import json
import tempfile
import numpy as np
import unittest

from utils.base_graph import BaseGraph
from utils.base_node import BaseNode
from benchmark_mms.evaluation import (
    compute_reachability,
    compute_path_optimality,
    compute_longterm_delta,
)
from benchmark_mms.dataloader import generate_task_pairs, save_task_pairs, load_task_pairs


def _make_linear_graph(n: int, edge_weight: float = 1.0):
    """构造 n 个节点的线形图，节点位置沿 x 轴排列，相邻节点之间有边。"""
    g = BaseGraph(map_root=None, edge_type="trav")
    nodes = []
    for i in range(n):
        node = BaseNode(i, trans=np.array([float(i), 0.0, 0.0]))
        g.add_node(node)
        nodes.append(node)
    for i in range(n - 1):
        g.add_edge_undirected(nodes[i], nodes[i + 1], edge_weight)
    return g, nodes


class TestReachability(unittest.TestCase):
    def test_all_reachable(self):
        g, _ = _make_linear_graph(5)
        pairs = [(0, 4), (1, 3)]
        result = compute_reachability(g, pairs)
        self.assertEqual(result["reachable_ratio"], 1.0)
        self.assertEqual(result["reachable_count"], 2)

    def test_disconnected(self):
        g, nodes = _make_linear_graph(4)
        # 移除 node 2，断开连通性
        g.remove_node(nodes[2])
        pairs = [(0, 3)]
        result = compute_reachability(g, pairs)
        self.assertEqual(result["reachable_ratio"], 0.0)

    def test_missing_node_skipped(self):
        g, _ = _make_linear_graph(3)
        pairs = [(0, 99)]  # node 99 不存在
        result = compute_reachability(g, pairs)
        self.assertEqual(result["total_count"], 0)


class TestPathOptimality(unittest.TestCase):
    def test_metric_ratio_approaches_1(self):
        # 单条直线路径，边权 = 1m，GT 测地 = 4m（0→4，4 步）
        g, _ = _make_linear_graph(5, edge_weight=1.0)
        pairs = [(0, 4)]
        gt = [4.0]
        result = compute_path_optimality(g, None, pairs, gt)
        self.assertAlmostEqual(result["metric_ratio_mean"], 1.0, places=5)

    def test_node_count(self):
        g, _ = _make_linear_graph(7)
        result = compute_path_optimality(g, None, [(0, 6)], [6.0])
        self.assertEqual(result["node_count"], 7)


class TestLongtermDelta(unittest.TestCase):
    def test_new_reachable_after_update(self):
        g_before, nodes_b = _make_linear_graph(4)
        # 断开 before 图
        g_before.remove_node(nodes_b[2])

        g_after, _ = _make_linear_graph(4)
        pairs = [(0, 3)]
        result = compute_longterm_delta(g_before, g_after, pairs)
        self.assertEqual(result["new_reachable_count"], 1)

    def test_path_shortens(self):
        # before: 0-1-2-3（直线，长 3m）
        # after: 0-1-2-3 + shortcut 0-3（长 1m，但我们添加权重 1.5 < 3）
        g_before, nodes_b = _make_linear_graph(4, edge_weight=1.0)
        g_after, nodes_a = _make_linear_graph(4, edge_weight=1.0)
        g_after.add_edge_undirected(nodes_a[0], nodes_a[3], 1.5)
        pairs = [(0, 3)]
        result = compute_longterm_delta(g_before, g_after, pairs)
        self.assertLess(result["path_length_delta_mean"], 0)  # after < before


class TestDataloader(unittest.TestCase):
    def test_generate_and_save_load(self):
        g, _ = _make_linear_graph(20)
        pairs = generate_task_pairs(g, n_pairs=5, min_distance=3.0)
        self.assertLessEqual(len(pairs), 5)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            tmp_path = f.name
        save_task_pairs(pairs, tmp_path)
        loaded = load_task_pairs(tmp_path)
        self.assertEqual(pairs, loaded)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试，确认通过**

```bash
python -m pytest python/test/test_benchmark_mms.py -v
```

预期输出（全部 PASS）：
```
test_all_reachable PASSED
test_disconnected PASSED
test_missing_node_skipped PASSED
test_metric_ratio_approaches_1 PASSED
test_node_count PASSED
test_new_reachable_after_update PASSED
test_path_shortens PASSED
test_generate_and_save_load PASSED
```

- [ ] **步骤 3：提交**

```bash
git add python/test/test_benchmark_mms.py
git commit -m "test(mms): add unit tests for evaluation and dataloader"
```

---

## Task 6: Shell 入口脚本

**Files:**
- Create: `scripts/run_benchmark_mms_reachability.sh`
- Create: `scripts/run_benchmark_mms_path_optimality.sh`
- Create: `scripts/run_benchmark_mms_longterm.sh`

- [ ] **步骤 1：创建 `run_benchmark_mms_reachability.sh`**

```bash
#!/bin/bash
# Usage: bash scripts/run_benchmark_mms_reachability.sh <SCENE> <K_MAX>
# Example: bash scripts/run_benchmark_mms_reachability.sh s17DRP5sb8fy 5

set -euo pipefail

readonly SCENE=${1:?"Usage: $0 <SCENE> <K_MAX>"}
readonly K_MAX=${2:?"Usage: $0 <SCENE> <K_MAX>"}
readonly PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc_private"
readonly DATA_ROOT="/Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/${SCENE}"
readonly OUT_DIR="${DATA_ROOT}/benchmark_mms/reachability"

# 构造 M_1 .. M_K 的地图目录列表
MAP_DIRS=()
for ((k=1; k<=K_MAX; k++)); do
    MAP_DIRS+=("${DATA_ROOT}/merged_M${k}/merge_finalmap")
done

python "${PROJECT_PATH}/python/benchmark_mms/run_reachability.py" \
    --merged_map_dirs "${MAP_DIRS[@]}" \
    --task_pairs_file "${DATA_ROOT}/task_pairs.json" \
    --out_dir "${OUT_DIR}" \
    --edge_type trav

echo "Done. Results in ${OUT_DIR}"
```

- [ ] **步骤 2：创建 `run_benchmark_mms_path_optimality.sh`**

```bash
#!/bin/bash
# Usage: bash scripts/run_benchmark_mms_path_optimality.sh <SCENE> <K_MAX> [GT_MESH]
# Example: bash scripts/run_benchmark_mms_path_optimality.sh s17DRP5sb8fy 5 /data/mesh.ply

set -euo pipefail

readonly SCENE=${1:?"Usage: $0 <SCENE> <K_MAX> [GT_MESH]"}
readonly K_MAX=${2:?"Usage: $0 <SCENE> <K_MAX> [GT_MESH]"}
readonly GT_MESH=${3:-""}
readonly PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc_private"
readonly DATA_ROOT="/Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/${SCENE}"
readonly OUT_DIR="${DATA_ROOT}/benchmark_mms/path_optimality"

MAP_DIRS=()
for ((k=1; k<=K_MAX; k++)); do
    MAP_DIRS+=("${DATA_ROOT}/merged_M${k}/merge_finalmap")
done

MESH_ARG=""
[ -n "$GT_MESH" ] && MESH_ARG="--gt_mesh_path ${GT_MESH}"

python "${PROJECT_PATH}/python/benchmark_mms/run_path_optimality.py" \
    --merged_map_dirs "${MAP_DIRS[@]}" \
    --task_pairs_file "${DATA_ROOT}/task_pairs.json" \
    --out_dir "${OUT_DIR}" \
    --edge_type trav \
    $MESH_ARG

echo "Done. Results in ${OUT_DIR}"
```

- [ ] **步骤 3：创建 `run_benchmark_mms_longterm.sh`**

```bash
#!/bin/bash
# Usage: bash scripts/run_benchmark_mms_longterm.sh <SCENE> <BEFORE_DIR> <AFTER_DIR>
# Example: bash scripts/run_benchmark_mms_longterm.sh s17DRP5sb8fy \
#   /data/epoch1/merge_finalmap /data/epoch2/merge_finalmap

set -euo pipefail

readonly SCENE=${1:?"Usage: $0 <SCENE> <BEFORE_DIR> <AFTER_DIR>"}
readonly BEFORE_DIR=${2:?"Usage: $0 <SCENE> <BEFORE_DIR> <AFTER_DIR>"}
readonly AFTER_DIR=${3:?"Usage: $0 <SCENE> <BEFORE_DIR> <AFTER_DIR>"}
readonly PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc_private"
readonly DATA_ROOT="/Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/${SCENE}"
readonly OUT_DIR="${DATA_ROOT}/benchmark_mms/longterm"

python "${PROJECT_PATH}/python/benchmark_mms/run_longterm.py" \
    --merged_map_dirs "${BEFORE_DIR}" "${AFTER_DIR}" \
    --task_pairs_file "${DATA_ROOT}/task_pairs.json" \
    --out_dir "${OUT_DIR}" \
    --edge_type trav

echo "Done. Results in ${OUT_DIR}"
```

- [ ] **步骤 4：加执行权限并提交**

```bash
chmod +x scripts/run_benchmark_mms_reachability.sh \
         scripts/run_benchmark_mms_path_optimality.sh \
         scripts/run_benchmark_mms_longterm.sh
git add scripts/run_benchmark_mms_*.sh
git commit -m "feat(mms): add shell entry scripts for benchmark_mms"
```

---

## 自检

### Spec 覆盖检查

| 需求 | 覆盖任务 |
|------|---------|
| A1 可达比例 vs k | Task 3 `compute_reachability` + Task 4 `run_reachability.py` |
| A2 路径长/GT 比值 vs k | Task 3 `compute_path_optimality` + Task 4 `run_path_optimality.py` |
| A3 拓扑 vs 度量对照 | Task 3 `topo_ratio_mean` + Task 4 `_make_topological_graph` |
| B4 通路变化 before/after | Task 3 `compute_longterm_delta` + Task 4 `run_longterm.py` |
| 固定 task pairs（apples-to-apples） | Task 2 `generate_task_pairs` + `save/load_task_pairs` |
| GT 测地距离 | Task 2 `load_gt_geodesic`（trimesh）|
| 不依赖 ROS | 所有脚本均无 `rospy` import |
| shell 入口与现有 benchmark 一致 | Task 6 |

### 类型一致性检查

- `compute_reachability` 接受 `list[tuple[int, int]]` → 与 `load_task_pairs` 返回类型一致 ✓
- `compute_path_optimality` 接受 `PointGraph` → `load_graph` 返回 `PointGraph` ✓
- `dijk_shortest_path` 接受 `BaseNode` → `graph.get_node(int)` 返回 `BaseNode` ✓
- `run_longterm.py` 的 `merged_map_dirs` 个数检查在 `main()` 开头 ✓
