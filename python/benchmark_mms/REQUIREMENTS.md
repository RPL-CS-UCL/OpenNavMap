# 需求：基于真实 OSM 占据栅格的多会话建图仿真

## 背景与目标

本仿真服务于一篇关于**众包多会话建图**（OpenNavMap）的学术论文，用于支撑可扩展机器人导航的核心论断。审稿人提出的问题是：

> "导航实验如何支撑论文的论断？OpenNavMap 具体解决了什么问题？"

本文的核心论点是：

> **OpenNavMap 解决了单会话建图（single-session mapping）的两个根本性局限——空间覆盖（spatial coverage）与时序适应性（temporal adaptability）——这两点直接提升下游导航的成功率。**

因此实验的统一对照是：

- **基线（single-session）**：仅会话 1（k=1）观测构建的地图。
- **多会话（multi-session）**：会话 1 累计合并会话 2、3、…、10（k=2→10）后的地图。

整个论证需要一个**最小但有说服力**的纯 Python 仿真，以 **OpenStreetMap 真实室外环境**作为占据地图来源，在其上运行两组聚焦实验（空间覆盖扩展 + 时序适应），并以统一口径的 metric 量化"多会话相对单会话的收益"。

---

## 统一实验载体与 GT 定义（重要，全文一致性基准）

为保证全文 metric 口径一致，所有实验统一构建在**同一个 OSM 占据栅格仿真**之上，不引入 Matterport3D mesh 或外部 `trav` 拓扑图数据。

### 1. 地图载体

- **占据栅格（occupancy grid）**：从 OSM 栅格化得到的 `base_map`（1000×1000，0.5 m/cell），`0=free`、`1=obstacle`。
- **会话观测合并图（merged map）**：每个会话沿其（随机、不等长）路径用**相机 FOV 仿真**（60° 视场角、7 m 距离、朝向=轨迹前进方向）观测，得到 `-1=已知自由`、`1=已知障碍`、`0=未知`。`k` 个会话的累计合并图记为 `merged_k`。
- **topometric 子图（topometric submap）**：在每个会话的轨迹上抽取**带度量位姿的拓扑节点**（节点 = `(x, y, yaw)` 米制位姿）与**度量边**（边 = 相对位姿 + 物理长度，米）。多会话合并 = 各子图并集 + 跨会话连接边（cross-session shortcut）。该 topometric 图用于 `topological_ratio` 与 `node_count` 指标，使其语义与栅格载体一致。
  - **节点建立阈值**：遍历轨迹位姿序列，当相对上一个节点的平移距离超过 **7 m** 或旋转角度差超过 **60°** 时，新建一个关键帧节点。
  - **跨子图连接条件**：来自不同子图的两个节点，若欧氏距离 < **10 m**，且两节点之间的连线在 `base_map` 上不经过任何障碍物（栅格逐格检测），则添加可通行 shortcut 边。

### 2. Ground Truth（GT）的唯一来源

**GT 一律定义在完整真实 `base_map`（含动态障碍的真值世界）上**，而非任何会话的观测地图：

- **GT 可达性**：在 `base_map`（膨胀后）上对 (start, goal) 跑 **A\***，能连通即 GT 可达。
- **GT 测地最短路长 `gt_geodesic`**：在 `base_map`（膨胀后）上的 8 连通 **A\***（直线步 = `RES`，对角步 = `RES * √2`，启发式 = 欧氏距离 × `RES`）最短路物理长度（单位：米）。

这样 `metric_ratio` 与 `topological_ratio` 的分母统一为 `gt_geodesic`，全文口径一致。

### 3. k 的语义

`k ∈ {1, 2, 3, ..., 10}` 表示累计会话数，共 **10 个会话**，每个会话对应一个独立区域（zone）。k=1 即单会话基线；k=2…10 为多会话累计。所有随 k 变化的曲线均以 k=1 为基线参照，横轴为 1…10。

---

## 环境与依赖

Python 3.8+，conda 或 venv。安装：

```bash
pip install numpy matplotlib scipy imageio osmnx geopandas shapely networkx
```

- **不依赖 ROS、Gazebo、rasterio、GDAL**
- 所有地理操作通过 `osmnx` + `shapely` + 手写 NumPy 栅格化完成
- 拓扑图操作使用 `networkx`
- 所有输出为静态 PNG 图 + 一个 GIF

---

## 步骤 1 — 下载并栅格化真实 OSM 地图

### 位置选择

以**深圳湾 / 南山科技园**作为地图中心：

```python
CENTER_LAT = 22.5076   # Nanshan, Shenzhen
CENTER_LON = 113.9437
AREA_M     = 1200      # 从 OSM 查询 1200×1200 m（过滤后裁剪到 500×500 m）
```

该区域混合了园区建筑、开放广场与道路，适合演示多区域导航。若此处 OSM 数据稀疏，回退到：

```python
# 回退 1：香港科技大学校园
CENTER_LAT, CENTER_LON = 22.3364, 114.2637

# 回退 2：MIT 校园，剑桥市 MA（OSM 覆盖密集，可保证有数据）
CENTER_LAT, CENTER_LON = 42.3601, -71.0942
```

### 下载流程

```python
import osmnx as ox
import numpy as np
from shapely.geometry import box
from shapely.affinity import affine_transform

# 1. 下载建筑轮廓
tags = {"building": True}
gdf_buildings = ox.features_from_point(
    (CENTER_LAT, CENTER_LON),
    tags=tags,
    dist=AREA_M // 2
)

# 2. 下载可行走路径 / 道路（作为自由空间参考）
G_walk = ox.graph_from_point(
    (CENTER_LAT, CENTER_LON),
    dist=AREA_M // 2,
    network_type="all"
)
```

### 栅格化为占据栅格

栅格规格：
- **栅格尺寸**：1000 × 1000 cells
- **分辨率**：0.5 m/cell → 500 × 500 m 真实世界
- **坐标系**：投影到 UTM（米制），使用 `gdf.to_crs(gdf.estimate_utm_crs())`

栅格化逻辑（纯 NumPy + Shapely，不用 rasterio）：

```python
# 在投影坐标（米）下定义包围盒
# cx, cy = 下载区域的 UTM 质心
GRID_W, GRID_H = 1000, 1000
RES = 0.5  # 米 / cell

x_min = cx - (GRID_W * RES) / 2
y_min = cy - (GRID_H * RES) / 2

grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)  # 0=free, 1=obstacle

for geom in gdf_buildings_proj.geometry:
    if geom is None or geom.is_empty:
        continue
    # 将几何边界转换为栅格索引
    minx, miny, maxx, maxy = geom.bounds
    col0 = int((minx - x_min) / RES)
    row0 = int((y_min + GRID_H * RES - maxy) / RES)  # y 轴翻转
    col1 = int((maxx - x_min) / RES)
    row1 = int((y_min + GRID_H * RES - miny) / RES)
    # 用 shapely 做细粒度 cell 检查
    for r in range(max(0, row0), min(GRID_H, row1 + 1)):
        for c in range(max(0, col0), min(GRID_W, col1 + 1)):
            cell_x = x_min + c * RES + RES / 2
            cell_y = (y_min + GRID_H * RES) - r * RES - RES / 2
            from shapely.geometry import Point
            if geom.contains(Point(cell_x, cell_y)):
                grid[r, c] = 1
```

> **性能提示**：1000×1000 栅格下内层循环对大建筑明显变慢。**强烈建议**用 `np.mgrid` + `shapely.vectorized.contains` 向量化，或用 `shapely.STRtree` 批处理（若耗时 >60 s）。如有 `rasterio.features.rasterize` 可用则更快，但**不得**作为硬依赖。

### 后处理

栅格化之后：
1. 添加 2 格边界墙：`grid[0:2,:]=1; grid[-2:,:]=1; grid[:,0:2]=1; grid[:,-2:]=1`
2. 校验栅格至少含 **15% 自由 cell** 与 **5% 障碍 cell**。否则（过稀或过密）调整 `AREA_M` 或换回退位置。
3. 保存 `grid` 为 `output/base_map.npy`（可复现）。
4. 保存预览图 `output/fig0_osm_base_map.png`，展示原始 OSM 占据栅格（深色背景，free=浅色，obstacle=深色）。

---

## 步骤 2 — 区域识别（自动）

不写死区域，**从真实地图自动识别 10 个空间上分离的可导航区域**（每个区域对应一个会话）：

```python
def find_zones(grid, n_zones=10):
    """
    寻找 n_zones 个种子点，满足：
    - 位于自由空间（grid == 0）
    - 远离障碍（用距离变换）
    - 彼此充分分离（>90 cells，约 45 m；10 个种子需在 500×500 m 内铺开）
    返回 (row, col) 种子点列表，每个区域一个。
    """
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(grid == 0)
    # 按"距障碍距离"排序（最安全的 cell 优先）
    candidates = np.argwhere(dist > 5)  # 距任意障碍 >2.5 m
    seeds = []
    for _ in range(n_zones):
        if len(candidates) == 0:
            break
        scores = dist[candidates[:, 0], candidates[:, 1]]
        best = candidates[np.argmax(scores)]
        seeds.append(tuple(best))
        # 移除距该种子 90 cells 内的候选
        dists_to_best = np.abs(candidates[:, 0] - best[0]) + np.abs(candidates[:, 1] - best[1])
        candidates = candidates[dists_to_best > 90]
    return seeds
```

将这些 `seeds` 作为每个会话路径的**锚点**。每个会话围绕其种子探索周边区域。若 `find_zones()` 返回少于 10 个种子（地图过于杂乱），回退到将栅格分为 `≈√10` 网格（如 4×3）的子区域，在每个子区域采样最空旷的 cell 作为种子，凑足 10 个。

---

## 步骤 3 — 会话路径生成

### 路径规划基础设施

```python
import heapq
from scipy.ndimage import binary_dilation

def inflate(grid, radius=3):
    struct = np.ones((2*radius+1, 2*radius+1), bool)
    return binary_dilation(grid.astype(bool), structure=struct).astype(np.uint8)

def astar(grid, start, goal, res=0.5):
    """
    8 连通 A*。直线步=res，对角步=res*√2，启发式=欧氏距离*res（admissible）。
    返回 (path, length_m)；不可达返回 (None, inf)。
    全文统一规划算法（A1 可达性、A2 度量路径、GT 测地距离均用本函数）。
    """
    if grid[start] or grid[goal]:
        return None, float("inf")
    H, W = grid.shape
    diag = res * np.sqrt(2)
    dirs = [(-1,0,res),(1,0,res),(0,-1,res),(0,1,res),
            (-1,-1,diag),(-1,1,diag),(1,-1,diag),(1,1,diag)]
    def h(a):  # 欧氏启发式（米）
        return res * np.hypot(a[0]-goal[0], a[1]-goal[1])
    g = {start: 0.0}
    parent = {}
    pq = [(h(start), start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]; path.append(cur)
            path.reverse()
            return path, g[goal]
        for dr, dc, step in dirs:
            nb = (cur[0]+dr, cur[1]+dc)
            if 0<=nb[0]<H and 0<=nb[1]<W and not grid[nb]:
                ng = g[cur] + step
                if ng < g.get(nb, float("inf")):
                    g[nb] = ng; parent[nb] = cur
                    heapq.heappush(pq, (ng + h(nb), nb))
    return None, float("inf")
```

> **统一口径**：`astar` 同时返回路径与米制长度，因此 `metric_ratio` 的分子（`merged_k` 规划栅格路径长）、`gt_geodesic` 的分母（`base_map` 路径长）由同一函数计算，保证可比。会话路点连接亦使用 `astar` 取其路径部分。

### 会话路点生成（随机、不等长）

每个会话 `i`（种子 `(sr, sc)`）的轨迹**带随机性且长度互不相同**，以模拟众包采集的差异：

1. 用会话专属随机种子 `rng = np.random.default_rng(seed=sess_id)`，保证可复现。
2. 随机路点数 `n_points ∈ [4, 8]`、随机基准半径 `radius ∈ [80, 200] cells`，每个会话各异 → 轨迹长度自然不等。
3. 围绕种子按角度采样路点，对**每个路点的角度与半径施加随机扰动**（非规则圆形），全部吸附到膨胀栅格中的自由 cell。
4. 用 `astar` 顺序连接路点，拼接为完整轨迹；记录沿途位姿 `(r, c, yaw)`（`yaw` 由相邻轨迹点差分得到，供 FOV 覆盖使用）。
5. **不强制统一的最小长度**；仅要求路点间 `astar` 均可达，各会话长短不一。

```python
def make_session_route(seed, grid, sess_id):
    """生成带随机性、长度不定的单会话轨迹，返回位姿列表 [(r,c,yaw), ...]。"""
    rng = np.random.default_rng(sess_id)
    inf_grid = inflate(grid)
    free_cells = np.argwhere(inf_grid == 0)
    n_points = int(rng.integers(4, 9))           # 各会话路点数不同
    radius   = int(rng.integers(80, 201))        # 各会话半径不同 → 长度不一
    wps = [tuple(seed)]
    for k in range(n_points):
        angle = 2*np.pi*k/n_points + rng.uniform(-0.4, 0.4)  # 角度扰动
        rad_k = radius * rng.uniform(0.6, 1.2)               # 半径扰动
        tr = int(np.clip(seed[0] + rad_k*np.sin(angle), 5, grid.shape[0]-5))
        tc = int(np.clip(seed[1] + rad_k*np.cos(angle), 5, grid.shape[1]-5))
        d = np.abs(free_cells[:,0]-tr) + np.abs(free_cells[:,1]-tc)
        wps.append(tuple(free_cells[np.argmin(d)]))
    wps.append(tuple(seed))                      # 闭合回种子

    # A* 连接路点，拼接轨迹并计算朝向 yaw
    route = []
    for a, b in zip(wps[:-1], wps[1:]):
        seg, _ = astar(inf_grid, a, b)
        if seg is None:
            continue
        route.extend(seg if not route else seg[1:])
    poses = []
    for j, (r, c) in enumerate(route):
        nxt = route[min(j+1, len(route)-1)]
        yaw = np.arctan2(nxt[0]-r, nxt[1]-c)    # 前进方向 yaw
        poses.append((r, c, yaw))
    return poses
```

### Topometric 子图抽取（用于 topological_ratio / node_count）

每个会话的轨迹位姿列表 `[(r, c, yaw), ...]` 直接用于抽取 **topometric 子图**：节点携带度量位姿 `(x, y, yaw)`（米制坐标），边携带相对位姿与物理长度（米）。

**节点建立规则**：遍历轨迹位姿序列，当与上一个关键帧节点的**平移距离 > 7 m** 或**旋转角度差 > 60°** 时，触发新建节点——与均匀下采样相比，此规则天然适应直道少节点、弯道多节点的非匀速探索场景。

**跨子图连接规则**：合并时，来自不同子图的两节点若满足：(1) 欧氏距离 < 10 m，且 (2) 两节点连线在 `base_map` 上逐格检测**无障碍物**，则添加可通行 shortcut 边。

```python
import networkx as nx

TRANS_THRESH_M  = 7.0            # 平移阈值（米）
ROT_THRESH_RAD  = np.radians(60) # 旋转阈值（60°）
CROSS_DIST_M    = 10.0           # 跨子图连接距离上限（米）

def build_topometric_subgraph(poses, res=0.5):
    """
    从单会话轨迹位姿抽取 topometric 子图。
    节点建立触发条件：相对上一节点平移 > TRANS_THRESH_M 或旋转差 > ROT_THRESH_RAD。
    边：相邻节点间的里程计边，边权 = 欧氏物理距离（米），属性 rel_pose=(dx,dy,dyaw)。
    返回 networkx.Graph，节点属性 (x, y, yaw)（米制）。
    """
    G = nx.Graph()
    if not poses:
        return G
    # 首节点
    r0, c0, y0 = poses[0]
    G.add_node(0, x=c0*res, y=r0*res, yaw=y0)
    node_idx = 0
    prev = (r0, c0, y0)

    for r, c, yaw in poses[1:]:
        pr, pc, py = prev
        trans = res * np.hypot(r-pr, c-pc)
        rot   = abs(np.arctan2(np.sin(yaw-py), np.cos(yaw-py)))
        if trans > TRANS_THRESH_M or rot > ROT_THRESH_RAD:
            node_idx += 1
            G.add_node(node_idx, x=c*res, y=r*res, yaw=yaw)
            dist_m = res * np.hypot(r-pr, c-pc)
            G.add_edge(node_idx-1, node_idx,
                       weight=dist_m,
                       rel_pose=(c-pc, r-pr, yaw-py))
            prev = (r, c, yaw)
    return G

def _line_free(r0, c0, r1, c1, base_grid):
    """Bresenham 直线逐格检测，全程无障碍返回 True。"""
    pts = set()
    dr, dc = abs(r1-r0), abs(c1-c0)
    sr, sc = (1 if r1>r0 else -1), (1 if c1>c0 else -1)
    r, c = r0, c0
    if dr > dc:
        err = dr // 2
        while r != r1:
            pts.add((r, c))
            err -= dc
            if err < 0: c += sc; err += dr
            r += sr
    else:
        err = dc // 2
        while c != c1:
            pts.add((r, c))
            err -= dr
            if err < 0: r += sr; err += dc
            c += sc
    pts.add((r1, c1))
    return all(base_grid[p] == 0 for p in pts
               if 0 <= p[0] < base_grid.shape[0] and 0 <= p[1] < base_grid.shape[1])

def merge_topometric_graphs(subgraphs, base_grid, res=0.5):
    """
    多会话 topometric 图合并：
    1. 所有子图并集（节点 ID 全局唯一，不同子图加偏移区分）
    2. 跨会话 shortcut 边：不同子图节点满足以下两个条件则连接：
       - 欧氏距离 < CROSS_DIST_M（10 m）
       - 两节点连线在 base_grid 上逐格无障碍（_line_free）
    返回合并后的 networkx.Graph。
    """
    merged = nx.Graph()
    offset = 0
    subgraph_node_sets = []   # 记录每个子图的节点 ID 集合（偏移后）
    for G in subgraphs:
        mapping = {n: n+offset for n in G.nodes}
        merged.update(nx.relabel_nodes(G, mapping))
        subgraph_node_sets.append({n+offset for n in G.nodes})
        offset += G.number_of_nodes() + 1

    # 跨子图 shortcut：仅检测来自不同子图的节点对
    all_nodes = [(n, d['x'], d['y']) for n, d in merged.nodes(data=True)]
    for si in range(len(subgraph_node_sets)):
        for sj in range(si+1, len(subgraph_node_sets)):
            ni_list = [(n,x,y) for n,x,y in all_nodes if n in subgraph_node_sets[si]]
            nj_list = [(n,x,y) for n,x,y in all_nodes if n in subgraph_node_sets[sj]]
            for ni, xi, yi in ni_list:
                for nj, xj, yj in nj_list:
                    d = np.hypot(xi-xj, yi-yj)
                    if d < CROSS_DIST_M and not merged.has_edge(ni, nj):
                        # 转换回栅格坐标做连线检测
                        ri, ci = int(yi/res), int(xi/res)
                        rj, cj = int(yj/res), int(xj/res)
                        if _line_free(ri, ci, rj, cj, base_grid):
                            merged.add_edge(ni, nj, weight=d,
                                            rel_pose=(xj-xi, yj-yi, 0.0))
    return merged
```

`node_count(k)` = 合并图节点数；`topological_ratio(k)` = 合并图上 `networkx.shortest_path_length`（weight='weight'）/ `gt_geodesic`。

### Exp 2 / B4 的动态障碍

生成会话路径后，定义 **2 个动态障碍**，满足：
- 分别只在会话 2 和会话 3 出现
- 放置在**会话 1 机器人路径**沿线或附近（使过时的 S1 地图"误以为"该处自由）
- 为 ~24×48 cells 的自由空间内矩形块（约 12×24 m，适配 500×500 m 场景）

```python
def place_dynamic_obstacle(route_s1, base_grid, block_h=24, block_w=48, offset=60):
    """
    沿 S1 路径取一点，略微偏移，放置矩形障碍。
    返回障碍矩形 (r0, c0, r1, c1)。
    """
    mid_idx = len(route_s1) // 2 + offset
    mid_r, mid_c = route_s1[min(mid_idx, len(route_s1)-1)]
    r0 = max(2, mid_r - block_h//2)
    c0 = max(2, mid_c - block_w//2)
    r1 = min(base_grid.shape[0]-2, r0 + block_h)
    c1 = min(base_grid.shape[1]-2, c0 + block_w)
    return (r0, c0, r1, c1)
```

---

## 实验内容（优先级排序）

全部实验统一构建在 OSM 占据栅格载体上，GT 来源见"统一实验载体与 GT 定义"。

### A1. 可达性（Reachability）— 核心，必做

**目标**：随会话数 `k` 增长，统计可规划到达的 goal 比例，量化空间覆盖收益。
**Metric**：`reachable_ratio(k)` = 在 `merged_k` 规划地图上可用 **A\*** 连通到 goal 的任务数 / 总任务数
**评测任务**：自动生成 **20 对** (start, goal)（覆盖近距、跨区等类型，见下"目标对生成"），全部在完整 `base_map`（GT）上保证可达。
**实现**：将 `merged_k` 转为规划栅格（unknown 视为障碍：`pg = (merged_k != -1).astype(uint8)`），膨胀后对每个 (start, goal) 调用 `astar`；返回 `inf` 视为不可达。
**预期曲线**：`reachable_ratio` 随 `k`（1→10）单调上升并趋于饱和，k=1 为最低基线。

#### 目标对生成（20 对）

自动生成 **20 对** goal，覆盖不同类型的任务：
- 4 对在同一区域内（近距，各随机选 2 个区域）
- 8 对跨相邻区域（zone i → zone i+1，取随机 4 对相邻区域组合）
- 8 对跨远距区域（zone i → zone j，|i-j| ≥ 4，取 4 对）

对每对 `(start, goal)`：在 **k=10 全会话合并地图与 base_map** 上均验证 `astar` 可达；若不可达则重采。

### A2. 路径最优性（Path Optimality）— 核心，必做

**目标**：多会话引入 cross-session shortcut 边 → 路径更短、更逼近 GT 测地最优。
**Metrics**：
- `metric_ratio(k)` = `merged_k` 规划栅格 A* 路径长 / `gt_geodesic`（越接近 1.0 越优）
- `topological_ratio(k)` = 合并 topometric 图最短路物理长 / `gt_geodesic`
- `node_count(k)` = 合并 topometric 图节点数

**GT 测地距离 `gt_geodesic`**：在 `base_map`（膨胀后）上对同一 (start, goal) 调用 `astar` 所得物理长（米）。
**统计口径**：在 A1 中"k=10 全部可达"的任务子集上计算 A2 各 ratio 的均值，保证跨 k 可比。
**预期曲线**：`metric_ratio` 与 `topological_ratio` 随 `k` 单调下降逼近 1.0；`node_count` 随 `k` 上升。

### B4. 长期地图更新（Long-term）— 选做，代表 long-term 轴

**目标**：环境变化（通路打开 / 关闭）→ 更新地图后路径骤降 + 新增可达 goal，演示时序适应性。本实验复用 Exp 2 的动态障碍设定，并补充 before/after 量化。
**设置**：
- `epoch-1`（stale，单会话 S1）：通道被动态障碍封闭，对应规划栅格 cell 置为障碍。
- `epoch-2`（updated，k=10 全会话合并）：新会话观测合并，通道恢复为已知自由 / 恢复 topometric 边。

**固定 start/goal**：`start` = Zone 1 种子（S1 锚点），`goal` = Zone 2 种子（S2 锚点）。这是一条 S1 部分观测为自由、但真值世界中被动态障碍阻断的跨区路径。

**Metrics**：
- `path_length_before`、`path_length_after`（米；before/after 两个数值）
- `new_reachable_goals`（更新后相对单会话新增的可达 goal 数）
- **安全性判定**：`stale` 地图规划路径与真值世界（含动态障碍）碰撞检测 → `COLLISION`；`updated` 地图绕行 → `SAFE`

**预期结果**：`path_length_after < path_length_before`（或 before 不可达→after 可达），`new_reachable_goals > 0`，且 stale=COLLISION、updated=SAFE。

---

## Metric 汇总表（统一口径）

所有 ratio 的分母统一为 `gt_geodesic`（定义在 `base_map` GT 上）；所有规划在 `merged_k` 规划栅格或合并拓扑图上进行。

| Metric | 单位 | 所属实验 | 说明 |
|--------|------|----------|------|
| `reachable_ratio(k)` | [0,1] | A1 | 在 `merged_k` 上可达 goal 比例（GT 全可达任务集上统计） |
| `metric_ratio(k)` | 无量纲 | A2 | `merged_k` 规划栅格度量路径长 / `gt_geodesic`，≥1.0，越接近 1 越优 |
| `topological_ratio(k)` | 无量纲 | A2 | 合并 topometric 图最短路物理长 / `gt_geodesic` |
| `node_count(k)` | 整数 | A2 | 合并 topometric 图节点数 |
| `path_length` | 米 | A2/B4 辅助 | 绝对路径长（度量栅格最短路物理长） |
| `path_length_before` / `path_length_after` | 米 | B4 | 通道关闭/打开前后的路径长 |
| `new_reachable_goals` | 整数 | B4 | 通道打开后相对单会话新增可达 goal 数 |

**一致性约束**：
1. `k=1` 始终为单会话基线，是所有曲线的参照起点。
2. `reachable_ratio` 随 `k` 单调不减；`metric_ratio`、`topological_ratio` 随 `k` 单调不增。
3. A2 在 A1 的"k=10 全可达任务子集"上统计，保证跨 k 可比。
4. 所有 ratio 分母统一为 GT 测地距离 `gt_geodesic`，分子均为各自地图上的物理路径长（米）。

---

## 实验 1 — 空间覆盖扩展（对应 A1 + A2）

**论断**：可导航面积与 goal 到达率随会话数单调增长，路径逐步逼近 GT 最优。

### 逐会话评测

在每个累计会话 k = 1, 2, …, 10 之后依次执行：
1. 构建规划栅格：`pg = (merged_k != -1).astype(uint8)`（unknown 视为障碍），膨胀一次后复用。
2. 在膨胀后的 `pg` 上对全部 **20 对** goal 调用 `astar`（A1：记录 `reachable_ratio(k)`）。
3. 对"k=10 全可达任务子集"调用 `astar` 得路径长，除以 `gt_geodesic` 得 `metric_ratio(k)`（A2）。
4. 在合并 topometric 图上用 `networkx.shortest_path_length(weight='weight')` 计算 `topological_ratio(k)` 与 `node_count(k)`（A2）。
5. 记录地图覆盖统计（已知自由面积 m²、总覆盖 m²、自由率）。

**预期结果**：`reachable_ratio` 随 k=1→10 单调上升趋于饱和；`metric_ratio`/`topological_ratio` 单调下降逼近 1.0；`node_count` 单调上升。

---

## 实验 2 — 时序适应性（对应 B4）

**论断**：过时的单会话地图导致导航失败；多会话更新恢复安全导航。

### 固定 start/goal

- `start` = Zone 1 种子（S1 锚点）
- `goal` = Zone 2 种子（S2 锚点）

这是一条 S1 部分观测为自由的跨区路径——但放置在 S1 观测路径上的动态障碍会在真值世界中阻断该路线。

### 评测

- **过时地图（stale，单会话 S1）**：仅 S1 观测 → **A\*** 规划路径 → 对真值世界（含动态障碍）做碰撞检测，记录 `path_length_before`
- **更新地图（updated，k=10 全会话合并）**：所有会话合并 → A* 规划路径 → 碰撞检测，记录 `path_length_after`
- 计算 `new_reachable_goals`

**预期结果**：stale 地图路径与动态障碍碰撞（`COLLISION`）；updated 地图安全绕行（`SAFE`），且 `path_length_after < path_length_before`、`new_reachable_goals > 0`。

---

## 输出图表

所有图统一风格：
- 背景：`#111827`
- 会话配色（10 色，Matplotlib `tab10` 调色板按序）：S1–S10 依次取 `tab10` 的 10 种颜色，确保相邻会话颜色可区分
- 已知自由：`[243,244,246]`，已知障碍：`[31,41,55]`，未知：`[107,114,128]`
- 所有文字：白色，DPI：150

### fig0_osm_base_map.png
完整 OSM 占据栅格（1000×1000），展示原始栅格化真实地图。标题：`"Real-World OSM Occupancy Map — [Location Name], [coords]"`。标注：总面积、障碍率、自由率。

### fig1_session_routes.png
2×5 面板（共 10 个子图）。每个面板在 base 占据图上叠加该会话的随机轨迹（会话色，长度各异）并标注区域种子。标题：`"Session N — Zone N Route (len=XXXX cells)"`，帮助直观展示各会话轨迹长度差异。

### fig2_cumulative_maps.png
选取 k=1, 3, 5, 7, 10 共 **5 个关键节点**（2×3 布局，最后格留空或展示 ground truth）。每个面板标注覆盖（m²）、自由面积（m²）、`reachable_ratio`（%）。按已知自由 / 已知障碍 / 未知着色；FOV 扇形覆盖痕迹清晰可见。

### fig3_nav_success.png
1×2 面板：
- 左：10×20 热力图（会话 × goal），绿/红配 ✓/✗ 符号
- 右：每会话 `reachable_ratio` 折线图（k=1→10），k=1 单会话基线用水平虚线标注

### fig4_temporal_adaptability.png
1×2 面板——stale vs updated 地图路径对比：
- 显示 base 地图 + 路径 + 动态障碍叠加（红色半透明）
- 标注：`"⚠ COLLISION"` vs `"✓ SAFE"`，并标 `path_length_before` / `path_length_after`
- 标注起点 ▲ 与终点 ★

### fig5_growth_charts.png
1×3 折线图，横轴均为会话数 k=1→10：
- 左：已知自由面积增长（m²）
- 中：`reachable_ratio` 增长曲线
- 右：`metric_ratio` 与 `topological_ratio` 随 k 下降（同图两条曲线，带标注）

### map_growth.gif
10 帧（每个累计会话一帧），FPS=3。每帧：合并地图（FOV 扇形覆盖着色）+ 当前会话轨迹。帧标题：`"After Session N | Coverage: XXXX m² | Reachable: XX% | MetricRatio: X.XX"`。

---

## 量化汇总（打印 + 保存）

打印到终端并保存为 `fig6_summary_table.png`：

```
========================================================================
MAP SOURCE: OpenStreetMap — [Location], [lat, lon]
GRID: 1000×1000 cells, 0.5 m/cell → 500×500 m real world
SESSIONS: 10  |  GOALS: 20 pairs  |  PLANNER: A*  |  FOV: 60°/7m
GT: geodesic A* on full base_map (metric, metres)
========================================================================
k   Config          Cov(m²)  Free(m²) FreeRatio Reach  MetricR TopoR  Nodes
------------------------------------------------------------------------
 1  S1 [single]      XXXX     XXXX     XX.X%     XX%    X.XX    X.XX   XXX
 2  +S2               XXXX     XXXX     XX.X%     XX%    X.XX    X.XX   XXX
 3  +S3               XXXX     XXXX     XX.X%     XX%    X.XX    X.XX   XXX
 4  +S4               XXXX     XXXX     XX.X%     XX%    X.XX    X.XX   XXX
 5  +S5               XXXX     XXXX     XX.X%     XX%    X.XX    X.XX   XXX
 6  +S6               XXXX     XXXX     XX.X%     XX%    X.XX    X.XX   XXX
 7  +S7               XXXX     XXXX     XX.X%     XX%    X.XX    X.XX   XXX
 8  +S8               XXXX     XXXX     XX.X%     XX%    X.XX    X.XX   XXX
 9  +S9               XXXX     XXXX     XX.X%     XX%    X.XX    X.XX   XXX
10  +S10 [full]       XXXX     XXXX     XX.X%     XX%    X.XX    X.XX   XXX
------------------------------------------------------------------------
B4 (temporal): path_before=XX.X m, path_after=XX.X m,
               new_reachable_goals=N,  stale→COLLISION, updated→SAFE
========================================================================
```

---

## 文件结构

```
multisession_sim_osm.py         ← 单一自包含脚本
output/
  base_map.npy                  ← 保存的栅格（可复现）
  fig0_osm_base_map.png
  fig1_session_routes.png
  fig2_cumulative_maps.png
  fig3_nav_success.png
  fig4_temporal_adaptability.png
  fig5_growth_charts.png
  fig6_summary_table.png
  map_growth.gif
```

---

## 关键实现注意事项

### 1. OSM 下载鲁棒性

将下载包在 try/except 中并自动回退：

```python
def download_osm_map(lat, lon, dist):
    try:
        gdf = ox.features_from_point((lat, lon), tags={"building": True}, dist=dist)
        if len(gdf) < 5:
            raise ValueError("Too few buildings")
        return gdf, f"({lat:.4f}, {lon:.4f})"
    except Exception as e:
        print(f"Primary location failed: {e}. Trying fallback...")
        for (flat, flon, fname) in FALLBACKS:
            try:
                gdf = ox.features_from_point((flat, flon), tags={"building": True}, dist=dist)
                if len(gdf) >= 5:
                    return gdf, fname
            except: continue
        raise RuntimeError("All OSM locations failed. Check internet connection.")
```

### 2. 栅格化性能

逐 cell 的 `shapely.contains` 循环正确但对大多边形慢。加速：

```python
from shapely.vectorized import contains  # 若可用（shapely ≥ 1.8）

# 或：bbox 预过滤 + 仅在 bbox 内 contains
```

若栅格化 >120 s，用 `tqdm` 加进度条或每 50 个建筑打印一次进度。

### 3. 栅格化后校验

```python
obstacle_ratio = grid.mean()
assert 0.03 < obstacle_ratio < 0.60, \
    f"Grid obstacle ratio {obstacle_ratio:.2f} out of expected range. Try a different location."
```

断言失败则增大 `AREA_M`（区域过稀）或减小（过密）。

### 4. 区域自动检测回退

若 `find_zones()` 返回少于 10 个种子（地图过于杂乱），回退到将栅格分为 ≈√10 的网格（如 4×3 = 12 格），在每个子格中采样距障碍最远的自由 cell，凑足 10 个种子。

### 5. 动态障碍放置校验

放置动态障碍后校验：
- 它不会切断整个地图（在加障碍后的栅格上对所有 10 个区域种子两两用 `astar` 检验连通性）
- 若切断，沿 S1 路径换一个 offset 重试

### 6. matplotlib 后端与 GIF

```python
import matplotlib
matplotlib.use("Agg")  # 必须在 import pyplot 之前

# GIF 帧捕获（兼容 matplotlib ≥ 3.8）：
fig.canvas.draw()
w, h = fig.canvas.get_width_height()
frame = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3]
```

### 7. 相机 FOV 覆盖仿真（60°、7 m、前进方向）

用**相机 FOV 扇形**代替 360° 圆形覆盖：每个机器人位置以轨迹前进方向为中心轴，向前方 ±30°（总计 60° 视场角）、最远 7 m（14 cells）范围内的 cell 标记为已观测。

```python
FOV_HALF_RAD = np.radians(30)   # ±30° → 60° 总视场角
FOV_RANGE_M  = 7.0              # 最远观测距离（米）

def observe_batch_fov(poses, grid, res=0.5):
    """
    基于相机 FOV 扇形的覆盖仿真。
    poses: [(r, c, yaw), ...]，yaw = 前进方向（弧度）
    返回 obs 数组：-1=已知自由, 1=已知障碍, 0=未知
    """
    obs  = np.zeros_like(grid, dtype=np.int8)
    mask = np.zeros_like(grid, dtype=bool)
    R    = int(np.ceil(FOV_RANGE_M / res))       # 最大半径（cells）
    H, W = grid.shape
    dr_arr = np.arange(-R, R+1)
    dc_arr = np.arange(-R, R+1)
    dR, dC = np.meshgrid(dr_arr, dc_arr, indexing='ij')
    dist2  = dR**2 + dC**2
    range_mask = dist2 <= R**2                   # 圆形距离过滤

    for r, c, yaw in poses:
        # 扇形角度过滤：cell 方向与 yaw 夹角 ≤ FOV_HALF_RAD
        angle_to_cell = np.arctan2(dR, dC)       # 相对方向
        angle_diff    = np.abs(np.arctan2(
            np.sin(angle_to_cell - yaw),
            np.cos(angle_to_cell - yaw)))        # [-π, π] → [0, π]
        fov_mask = (angle_diff <= FOV_HALF_RAD) & range_mask

        rr = np.clip(r + dR[fov_mask], 0, H-1)
        cc = np.clip(c + dC[fov_mask], 0, W-1)
        mask[rr, cc] = True

    obs[mask & (grid==0)] = -1   # 已知自由
    obs[mask & (grid==1)] =  1   # 已知障碍
    return obs
```

> **覆盖特性**：60° 前向视场使得每条轨迹的覆盖区域呈"走廊状"——沿前进方向可见，侧后方不可见。这比 360° 圆形更真实，且使多会话（不同方向探索）的覆盖互补效果更明显，更有力地支撑多会话收益的论断。

### 8. 度量路径长与 GT 测地距离（A2 一致性）

`metric_ratio`、`topological_ratio` 的分母必须用同一 `gt_geodesic`，且与导航规划使用相同的 **A\*** 算法：

```python
# GT：在膨胀后的 base_map 上
_, gt_geodesic = astar(inflate(base_map), start, goal, RES)

# 估计：在 merged_k 规划栅格上（unknown 视为障碍）
pg_k = (merged_k != -1).astype(np.uint8)
_, est_len = astar(inflate(pg_k), start, goal, RES)
metric_ratio_k = est_len / gt_geodesic   # ≥ 1.0，越接近 1 越优
```

### 9. 大栅格 A* 规划性能（500×500 m 场景）

1000×1000 栅格下，Python A* 单次规划约 0.5–2 s；A1 需对 20 对 goal × 10 个 k 共 200 次规划，总计约 2–6 分钟（优化后可压到 1 分钟内）。建议：
- `inflate` 后的栅格预计算一次，跨 goal 和跨 k 复用（合并图不同 k 仅增量更新）。
- GT 测地距离 `gt_geodesic` 对固定 (start, goal) 只算一次并缓存，不随 k 重算。
- 如性能仍不足，可用 `scipy.sparse.csgraph.dijkstra` 对预构建的稀疏邻接矩阵做多源批量规划，代替逐对 A*。

---

## 交付物

单一可运行脚本 `multisession_sim_osm.py`。运行：

```bash
python multisession_sim_osm.py
```

应当：
1. 下载真实 OSM 建筑轮廓（需联网，~10–20 s）
2. 栅格化为 1000×1000 占据栅格（~30–90 s，视区域密度）
3. 自动检测 10 个区域，生成 10 条随机不等长会话轨迹，构建 topometric 子图
4. 运行实验 1（A1+A2，k=1→10）与实验 2（B4）
5. 将全部输出文件保存到 `./output/`
6. 将量化汇总打印到终端
7. 在现代笔记本上端到端 **8 分钟内**完成

每个主要步骤打印进度，便于用户确认运行中。
