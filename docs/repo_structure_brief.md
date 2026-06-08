# OpenNavMap / LiteVLoc 仓库结构简述

## 1. 仓库定位

当前仓库是 **OpenNavMap** 系统仓库：一个面向 **multi-session mapping** 的 topometric map 系统，目标是从多次采集的数据中构建、对齐、合并并维护可用于导航的轻量级拓扑度量地图。

其中，**LiteVLoc** 是这个系统中的视觉定位子模块，作为 pinned git submodule 位于 `third_party/litevloc_code`。它的职责是：

- 基于最终构建出的 **multi-session topometric map** 做全局视觉定位
- 为 image-goal navigation 提供目标图像到地图节点的视觉锚定能力
- 在在线导航时提供视觉位置估计，并与规划/融合模块协同

因此，理解本仓库时应采用如下主从关系：

- **OpenNavMap**：主系统，强调多会话建图、地图管理、地图合并、图结构维护与导航支持
- **LiteVLoc**：子模块，强调基于已有 topometric map 的视觉定位

## 2. 代码主线

从代码职责上看，仓库可以分成三条主线：

1. **多会话地图构建与合并主线**
2. **LiteVLoc 视觉定位主线**（`third_party/litevloc_code`）
3. **导航与系统集成主线**（`third_party/litevloc_code`）

其中第一条是系统层面的主目标，第二条是建立在第一条结果之上的核心能力。

## 3. 顶层目录说明

```text
opennavmap/
├── python/            # OpenNavMap 核心建图、合并与 map-level benchmark
├── launch/            # OpenNavMap 启动入口（LiteVLoc launch 位于 submodule）
├── scripts/           # 批处理脚本与实验入口
├── docs/              # 使用说明与流程文档
├── rviz_cfg/          # RViz 配置
├── third_party/       # 仓库内第三方子模块/依赖
│   └── litevloc_code/ # LiteVLoc visual localization submodule
├── app/               # 应用侧代码/接口
├── paper_writing/     # 论文材料
├── requirements.txt   # Python 依赖
├── package.xml        # ROS 包定义
└── CMakeLists.txt     # ROS/catkin 构建配置
```

## 4. `python/` 目录的系统分层

### 4.1 多会话地图构建与合并

这一部分是 OpenNavMap 的主干。

- `python/map_merge_pipeline.py`
  - 多子地图读取、跨图匹配、回环建立、GTSAM 优化、地图合并的主入口
  - 是“multi-session topometric map”离线构建/对齐/融合的核心文件

- `python/map_manager.py`
  - 统一管理一个子地图中的多种图结构
  - 当前主要管理：`odom`、`trav`、`covis`

- `third_party/litevloc_code/python/image_graph.py` / `third_party/litevloc_code/python/image_node.py`
  - 维护带图像、descriptor、相机内参、深度等信息的共视图
  - 用于地图关键帧表示，也是 LiteVLoc 使用的地图观测层

- `third_party/litevloc_code/python/point_graph.py` / `third_party/litevloc_code/python/point_node.py`
  - 维护 odometry graph 与 traversability graph
  - 面向位姿链、通行性建图和最短路径规划

- `third_party/litevloc_code/python/utils/base_graph.py` / `third_party/litevloc_code/python/utils/base_node.py`
  - 图与节点的基础抽象层

- `python/utils/gtsam_pose_graph.py`
  - GTSAM 后端封装，用于地图合并与位姿图优化

从系统角度看，OpenNavMap 的地图不是单一结构，而是至少包含三类互补图：

- **covis graph**：图像关键帧及其视觉关联
- **odom graph**：顺序位姿链
- **trav graph**：导航可达关系图

这三类图共同构成 topometric map。

### 4.2 LiteVLoc 视觉定位子模块

这部分建立在上面的地图之上。

- `third_party/litevloc_code/python/loc_pipeline.py`
  - LiteVLoc 的核心流程
  - 从 topometric map 中读取 `covis graph`
  - 执行“VPR 粗定位 -> 图像匹配 -> 位姿求解”

- `third_party/litevloc_code/python/ros_loc_pipeline.py`
  - LiteVLoc 在线 ROS 封装
  - 订阅图像、深度、相机参数与融合位姿，输出 `/vloc/odometry`

- `third_party/litevloc_code/python/utils/utils_vpr_method.py`
  - VPR 模型初始化与检索/时序匹配封装

- `third_party/litevloc_code/python/utils/utils_image_matching_method.py`
  - 局部图像匹配封装

- `third_party/litevloc_code/python/utils/pose_solver.py`
  - PnP / Essential Matrix / Procrustes 等位姿求解器

LiteVLoc 的职责不是建图，而是：

- 使用 **已经生成好的 multi-session map**
- 对 query image 做全局 place recognition
- 在候选关键帧上做局部几何精化
- 输出观测相机在地图坐标系下的位姿

### 4.3 导航与系统集成

- `third_party/litevloc_code/python/global_planner.py`
  - 结合 LiteVLoc 的全局匹配能力与 `trav graph` 的最短路径规划
  - 将目标图像映射到地图节点后生成 waypoint

- `third_party/litevloc_code/python/ros_global_planner.py`
  - 全局规划 ROS 封装

- `third_party/litevloc_code/python/pose_fusion.py`
  - 用 GTSAM 将局部里程计与视觉定位结果做融合

- `third_party/litevloc_code/python/ros_pose_fusion.py`
  - 在线位姿融合 ROS 封装

- `third_party/litevloc_code/python/depth_registration.py`
  - 额外的局部几何/深度配准支持模块

这一层说明 OpenNavMap 不是“只做地图文件生成”，而是继续向在线导航系统延伸。

## 5. `third_party/` 与 OpenNavMap 的其他组成模块

当前仓库中的 `third_party/` 目录承载了 OpenNavMap 所依赖的重要外部模型组件和 LiteVLoc submodule。

### 5.1 仓库内已有的 `third_party/` 子模块

- `third_party/litevloc_code`
  - LiteVLoc 视觉定位、导航 runtime、pose fusion、map-free/RPE benchmarks 的 source-of-truth
  - OpenNavMap 通过 pinned commit 引用它

- `third_party/vismatch`
  - 提供局部图像匹配能力
  - 被 `python/utils/utils_image_matching_method.py` 所依赖
  - 是 LiteVLoc 局部精定位阶段的重要底层组件

### 5.2 仓库内的外部模型组件

- `third_party/VPR-methods-evaluation`
  - 为 VPR 模型、描述子提取和检索流程提供支持
  - 当前仓库中的 `utils_pipeline.py` 和 `utils_vpr_method.py` 会显式把它加入 `sys.path`
  - 它支撑 LiteVLoc 的全局视觉检索能力

因此，如果从系统组成的角度描述 OpenNavMap，可以理解为：

- **当前仓库**：OpenNavMap 的核心地图构建、地图合并与 map-level benchmark 逻辑
- **`third_party/litevloc_code`**：LiteVLoc 视觉定位、规划 runtime、位姿融合与定位侧 benchmark
- **`third_party/vismatch`**：局部图像匹配子模块
- **`third_party/VPR-methods-evaluation`**：全局视觉检索依赖的外部模型模块

## 6. 推荐理解顺序

如果要快速建立正确的系统认知，建议按下面顺序读代码：

1. `python/map_merge_pipeline.py`
2. `python/map_manager.py`
3. `third_party/litevloc_code/python/image_graph.py`
4. `third_party/litevloc_code/python/point_graph.py`
5. `third_party/litevloc_code/python/loc_pipeline.py`
6. `third_party/litevloc_code/python/global_planner.py`
7. `third_party/litevloc_code/python/pose_fusion.py`

这样的顺序更符合 OpenNavMap 的真实系统层次：

- 先理解地图是怎么组织与合并的
- 再理解 LiteVLoc 如何使用这张地图做视觉定位
- 最后理解定位如何服务导航与在线系统

## 7. 一句话总结

**OpenNavMap 是主系统仓库，目标是构建和维护 multi-session topometric map；LiteVLoc 是位于 `third_party/litevloc_code` 的视觉定位 submodule，负责基于该地图执行 global visual localization，并进一步支撑导航与位姿融合。**
