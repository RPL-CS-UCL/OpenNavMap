# benchmark_mms — Multi-Session Mapping 评测需求说明

## 背景

本模块对应论文 `experiment_vnav.tex` 的 `sec:exp_single_vs_multi`（Q5/E5）：**单会话 vs 多会话建图对 Image-goal Visual Navigation 的增益**。

定性声明：`large-scale, long-term operations rely on scalable, globally consistent mapping`。本实验将该 claim 量化为两条轴：

- **large-scale（空间轴）**：更多会话覆盖更大区域 → 提升可达性与路径最优性
- **long-term（时间轴）**：跨时间更新地图 → 反映环境变化对导航的影响

> 本节定性为 "system demonstration rather than a standalone contribution"，故实验保持简洁、可量化，不过度追求完备性。

---

## 实验设置

**环境**：Matterport3D 仿真场景（有 GT mesh → 可查询测地最短路）

**会话定义**：同一场景内 $N$ 条部分重叠轨迹，各自独立采集（start/end point 随机生成）。  
- $S$：单会话（仅第 1 条轨迹）  
- $M_k$：前 $k$ 条轨迹合并后的地图（$k = 1, 2, \ldots, K$）

**任务集合**：固定一组贯穿全图的 `(start_node, goal_node)` image pair，保存为 JSON，所有实验共用同一批任务（apples-to-apples）。

**前置条件**：各 $M_k$ 的合并地图已由 `map_merge_pipeline.py` 离线生成并保存至磁盘；本模块只读图、跑规划、算 metric。

---

## 实验内容（优先级排序）

### A1. 可达性（Reachability）— 核心，必做

**目标**：随 $k$ 增长，统计可规划到达的 goal 比例。  
**Metric**：`reachable_ratio(k)` = 可连通到 goal 的任务数 / 总任务数  
**预期曲线**：随 $k$ 单调上升并趋于饱和  
**实现**：在 `trav` 图上对每个 (start, goal) 调用 `dijk_shortest_path`，`inf` 距离视为不可达

### A2. 路径最优性（Path Optimality）— 核心，必做

**目标**：多会话增加 cross-session shortcut 边 → 路径更短更优。  
**Metrics**：
- `metric_ratio(k)` = 估计度量路径长 / GT 测地最短路长（越接近 1.0 越优）
- `node_count(k)` = 合并图的节点数

**预期曲线**：`metric_ratio` 随 $k$ 单调下降逼近 1.0  
**GT 测地距离来源**：从 Matterport3D `.ply` mesh 用 `trimesh` 计算两点间测地线长度

### A3. 拓扑 vs 度量规划对照（对照实验，折入 A2）

**目标**：在同一张 $M_k$ 地图上，比较纯拓扑规划（最少跳数）与度量规划（物理距离加权 Dijkstra）的路径长度差异。  
**Metric**：`topological_ratio(k)` = 拓扑路径物理长 / GT 测地距离  
**预期**：`topological_ratio > metric_ratio`，说明无全局度量一致性会选错路

### B4. 长期地图更新（Long-term）— 选做，代表 long-term 轴

**目标**：环境变化（通路打开）→ 更新地图后路径骤降 + 新增可达 goal。  
**设置**：
- `epoch-1`：某通道封闭（对应部分 `trav` 边权重置为 `inf` 或删除）
- `epoch-2`：该通道打开（对应新会话数据合并，恢复 `trav` 边）

**Metrics**：
- `path_length_before`、`path_length_after`（before/after 两个数字）
- `new_reachable_goals`（打开通道后新增可达 goal 数）

---

## Metric 汇总表（统一口径）

| Metric | 单位 | 说明 |
|--------|------|------|
| `reachable_ratio` | [0,1] | 可达 goal 比例 |
| `metric_ratio` | 无量纲 | 估计度量路径长 / GT 测地距离 |
| `topological_ratio` | 无量纲 | 拓扑路径物理长 / GT 测地距离 |
| `node_count` | 整数 | 合并图节点数 |
| `path_length` | 米 | 绝对路径长（A2/B4 辅助） |

---

## 结果呈现形式

每个实验的输出均为：**一张 PDF 图（主视觉）+ 一个 CSV 辅助表**。图用于论文插图，CSV 用于数据存档与按需制表。

---

### A1 可达性 — 折线图

主图（`reachability.pdf`）：`reachable_ratio` vs `k`，单条折线，y 轴 [0, 1]。

```
Reachable Goal Ratio
1.0 |              ●───●
0.8 |         ●
0.6 |    ●
0.4 | ●
    +──────────────────── Sessions k
      1    2    3    4    5
```

辅助表（`reachability.csv`，论文图说明下附关键行）：

| k | reachable_ratio | reachable_count | node_count |
|---|----------------|----------------|------------|
| 1 | 0.42           | 21 / 50        | 312        |
| 3 | 0.78           | 39 / 50        | 891        |
| 5 | 0.95           | 47 / 50        | 1423       |

---

### A2+A3 路径最优性 — 双子图 + 辅助表

主图（`path_optimality.pdf`）：左子图为 `metric_ratio` 与 `topological_ratio` vs `k`（加 y=1.0 参考线），右子图为 `node_count` 柱状图。

```
Path Length / GT Geodesic        Node Count
2.0 | ◇                         1500 |       ▓
1.6 |     ◇                     1000 |    ▓  ▓
1.2 | ●       ◇                  500 | ▓  ▓  ▓  ▓
1.0 |.....●───●───●                0 +────────────
    +──────────────── k                1  2  3  4
      1  2  3  4
  ● metric_ratio   ◇ topo_ratio
```

辅助表（`path_optimality.csv`）：

| k | metric_ratio | topo_ratio | node_count |
|---|-------------|------------|------------|
| 1 | 1.82        | 2.31       | 312        |
| 3 | 1.24        | 1.87       | 891        |
| 5 | 1.06        | 1.61       | 1423       |

核心论点：`metric_ratio` 随 $k$ 下降逼近 1.0，`topo_ratio` 始终高于 `metric_ratio`，证明度量一致性对规划质量的必要性。

---

### B4 长期更新 — 散点图 + before/after 汇总表

主图（`longterm.pdf`）：散点图，横轴 `path_length_before`，纵轴 `path_length_after`，对角虚线为"无变化"参考线；点落在对角线下方表示更新后路径更短。

```
Path Length After (m)
 30 |  ·  ·
 20 |    · ·  ·
 10 |      ·        ·   ·
  0 +───────────────────── Path Length Before (m)
    0    10   20   30
      (-- 对角线: no change --)
```

辅助汇总表（附于图说明，`longterm.csv` 存完整逐任务数据）：

| 指标                        | 值       |
|-----------------------------|----------|
| New reachable goals         | 7 / 50   |
| Mean path length before     | 18.4 m   |
| Mean path length after      | 14.1 m   |
| Mean delta (after − before) | −4.3 m   |

---

## 诚实性 framing

多会话并非"总是更好"：地图变大、merge 有引入误差风险。正确表述：

> "多会话在 coverage / path optimality 上有明确、可量化增益，node culling（Q4）控制规模代价。"

逻辑链：Q2（merge 精度）→ Q4（culling 控规模）→ Q5（导航增益）。

---

## 文件规划

```
python/benchmark_mms/
├── REQUIREMENTS.md         # 本文件
├── __init__.py
├── parser.py               # argparse
├── dataloader.py           # 加载 PointGraph + task pairs + GT 测地距离
├── evaluation.py           # metric 计算（reachability / path optimality / longterm）
├── run_reachability.py     # E5-A1 入口
├── run_path_optimality.py  # E5-A2+A3 入口
└── run_longterm.py         # E5-B4 入口

scripts/
├── run_benchmark_mms_reachability.sh
├── run_benchmark_mms_path_optimality.sh
└── run_benchmark_mms_longterm.sh
```

---

## 依赖

- `python/point_graph.py` — `PointGraphLoader`, `PointGraph`
- `python/utils/utils_shortest_path.py` — `dijk_shortest_path`
- `python/utils/base_graph.py` — `BaseGraph`
- `trimesh` — GT 测地距离计算（`pip install trimesh`）
- `matplotlib` — 曲线图输出（已有）
- 不依赖 ROS
