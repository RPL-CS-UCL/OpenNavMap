# 拓扑节点来源说明

## 概述

图中蓝色大圆点是 **拓扑度量地图（topometric map）的节点**，与 A\* 路径搜索的展开节点无关。

---

## 节点来源 1：Session 内关键帧（intra-session keyframes）

**产生函数：** `build_topometric_subgraph(poses, res)` — 见 `multisession_sim_osm.py`

每个 session 的机器人轨迹是 `[(row, col, yaw), ...]`。
遍历轨迹时，当满足以下任一条件时，将当前位姿加入拓扑图作为一个节点：

| 触发条件 | 阈值常量 | 值 |
|----------|----------|----|
| 平移距离 > 阈值 | `TRANS_THRESH_M` | 7.0 m |
| 旋转角度 > 阈值 | `ROT_THRESH_RAD` | 60° |

相邻节点之间自动添加一条边，权重为两点之间的欧氏距离（米）。

```
poses: [ p0, p1, p2, ..., pN ]
         ↑       ↑          ↑
       node0   node1  ...  nodeM   ← 每当平移>7m 或 旋转>60° 时新建节点
         └───────┘    ←    edge (weight = 欧氏距离)
```

这些节点代表 **机器人行走轨迹的里程碑位置**，不是 A\* 搜索中展开的 grid cell。

---

## 节点来源 2：跨 session 合并边（inter-session cross edges）

**产生函数：** `merge_topometric_graphs(subgraphs, base_grid)` 和 `_add_subgraph_to_merged()`

随着 session 数 k 增加，前 k 个 session 的 subgraph 被逐步合并为一个大图。
合并时，如果来自不同 session 的两个节点满足：

1. 空间距离 < `CROSS_DIST_M = 10.0 m`
2. 两节点之间的直连线段不穿过 base_grid 上的障碍物（`_line_free` 检测）

则在这两个节点之间添加一条 cross-session 边。

```
Session 1 subgraph:   A─B─C
Session 2 subgraph:   D─E─F

合并后（若 B 与 E 距离 < 10m 且连线无障碍）:
  A─B─C
    │
    E─F─D   ← B─E 为跨 session 边
```

**结论：** k 越大，累积的 session 越多，图上蓝色节点越密集，跨 session 连接越丰富。

---

## 节点来源 3：A\* 搜索（不显示）

A\* 在 occupancy grid 上搜索路径时，会临时展开大量 grid cell（open list / closed list），但这些中间状态 **不会加入拓扑图**，也不会在图上显示。

A\* 只返回最终路径 `[(row, col), ...]`，即图中绿色线段。

---

## 可视化对应

| 图中元素 | 来源 |
|----------|------|
| 蓝色大圆点 | 拓扑节点（intra + inter session keyframes） |
| 白色半透明细线 | 拓扑边（节点之间的连接） |
| 绿色粗线 | A\* 在当前 merged partial map 上搜索到的最短路径 |
| 橙色虚线 | A\* 在完整 base_grid 上搜索到的 GT 最优路径（参考） |
| 绿色圆点 | start point |
| 红色 X | goal point |

---

## 相关常量（`multisession_sim_osm.py`）

```python
TRANS_THRESH_M  = 7.0      # 节点采样平移阈值（米）
ROT_THRESH_RAD  = 1.047    # 节点采样旋转阈值（60°）
CROSS_DIST_M    = 10.0     # 跨 session 边最大距离（米）
INFLATE_RADIUS  = 3        # 障碍物膨胀半径（格）
RES             = 0.5      # 默认分辨率（米/格）
```
