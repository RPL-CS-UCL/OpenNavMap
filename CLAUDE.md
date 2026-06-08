# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

**OpenNavMap** 是当前仓库承载的主系统：一个面向多次采集数据的 **multi-session topometric mapping** 系统，目标是构建、对齐、合并并维护可用于导航的轻量级拓扑度量地图，并进一步支持大规模、可扩展的 Image-goal Visual Navigation。

**LiteVLoc** 是 OpenNavMap 中的视觉定位子模块，负责基于最终构建出的 **multi-session topometric map** 执行 global visual localization，并在在线系统中与路径规划、位姿融合等模块协同。ROS 包名为 `litevloc`，主要语言为 Python 3.8。

## 常用命令

```bash
# 安装环境（Python 3.8 + CUDA 11.8）
conda create --name opennavmap python=3.8
conda activate opennavmap
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 numpy=1.24.3 -c pytorch -c nvidia
pip install -r requirements.txt

# 验证 torch 安装
python test_torch_install.py

# 构建 ROS 包（可选）
catkin build litevloc -DPYTHON_EXECUTABLE=$(which python)

# 运行单个测试
python -m pytest python/test/test_pose_solver.py
python -m pytest python/test/test_shortest_path.py

# 离线定位 pipeline
python python/loc_pipeline.py \
    --map_path <map_dir> \
    --query_data_path <query_dir> \
    --image_size 512 288 --device=cuda \
    --vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=256 \
    --img_matcher master \
    --pose_solver pnp --config_pose_solver python/config/dataset/matterport3d.yaml

# ROS 在线定位（仿真/实机）
roslaunch litevloc run_vloc_online_simuenv.launch
roslaunch litevloc run_vloc_online_anymal.launch

# 地图合并
bash scripts/run_map_merging.sh
```

## 代码架构

### 三条主线

1. **多会话地图构建与合并**
   - 主入口：`python/map_merge_pipeline.py`
   - 负责多子地图读取、跨图匹配、回环建立、GTSAM 优化与地图融合

2. **LiteVLoc 视觉定位**
   - 主入口：`python/loc_pipeline.py`、`python/ros_loc_pipeline.py`
   - 基于已构建的 topometric map 执行全局检索与局部位姿精化

3. **导航与系统集成**
   - 主入口：`python/global_planner.py`、`python/pose_fusion.py`
   - 负责目标图像导航、位姿融合与 ROS 在线系统对接

### LiteVLoc 三层层次化定位

1. **全局定位（VPR）** — 粗定位，基于视觉描述符的场景识别
   - 模型：CosPlace、NetVLAD、EigenPlaces、AnyLoc-DINOv2
   - 匹配策略：单次匹配 / 拓扑滤波 / 序列匹配 / 图搜索

2. **局部精化（图像匹配）** — 细粒度位姿估计
   - 特征方法：SIFT、DISK、SuperPoint、LightGlue、Mast3r、ROMA
   - 位姿求解：PnP、Essential Matrix、Procrustes

3. **地图表示** — 轻量级拓扑度量地图（三种图结构）
   - 共视图 `ImageGraph`：图像关键帧观测
   - 里程计图 `PointGraph`：顺序位姿链
   - 通行性图：连通性规划

### 核心模块

| 模块 | 职责 |
|------|------|
| `python/map_merge_pipeline.py` | OpenNavMap 主干，多子地图对齐、回环建立与合并 |
| `python/map_manager.py` | 多图协调管理，统一组织 `odom` / `trav` / `covis` 图 |
| `python/image_graph.py` / `point_graph.py` | topometric map 的核心图结构 |
| `python/loc_pipeline.py` | LiteVLoc 主定位流程（VPR → 图像匹配 → 位姿求解） |
| `python/ros_loc_pipeline.py` | LiteVLoc 在线 ROS 封装 |
| `python/global_planner.py` | 基于目标图像的最短路径规划 |
| `python/pose_fusion.py` | VIO + 视觉定位融合（GTSAM 后端） |

### 目录结构

```
python/
├── *.py                    # 核心算法模块（建图 / 定位 / 规划 / 融合）
├── ros_*.py                # ROS 节点封装
├── utils/                  # 工具函数（45+ 文件）
│   ├── utils_vpr_method.py     # VPR 模型初始化与推理
│   ├── utils_image_matching_method.py  # 特征匹配封装
│   ├── utils_geom.py           # 位姿转换与误差计算
│   ├── pose_solver.py          # 抽象位姿求解器接口
│   ├── gtsam_pose_graph.py     # GTSAM 后端
│   └── utils_ros/              # ROS 消息转换
├── map_merge_pipeline.py   # 多会话地图构建与合并主入口
├── loc_pipeline.py         # LiteVLoc 离线定位主入口
├── global_planner.py       # 基于 trav graph 的全局规划
├── pose_fusion.py          # 里程计与视觉定位融合
├── benchmark_vpr/          # VPR 评估
├── benchmark_map_free/     # 无地图特征匹配评估
├── benchmark_rpe/          # 相对位姿估计评估
├── benchmark_kf_selection/ # 关键帧选择评估
├── config/dataset/         # 各数据集的 YACS 配置（.yaml）
├── segment_change/         # 场景变化检测
├── ltl_task_planner/       # 时态逻辑任务规划
└── test/                   # 单元测试
```

## 地图数据格式

地图目录通常至少包含以下文件：

```
map_root/
├── seq/                        # 图像帧目录
│   ├── 000.color.jpg
│   └── 000000.depth.png
├── timestamps.txt              # img_name timestamp
├── intrinsics.txt              # fx fy cx cy width height
├── poses.txt                   # img_name qw qx qy qz tx ty tz
├── poses_abs_gt.txt            # 可选，绝对位姿 GT
├── gps_data.txt                # 可选，GPS 信息
├── iqa_data.txt                # 可选，图像质量评估
├── edges_covis.txt             # [node_a, node_b, weight]
├── edges_odom.txt
├── edges_trav.txt
└── database_descriptors.txt    # VPR 特征
```

## 配置系统

使用 YACS 进行层次化配置，数据集配置位于 `python/config/dataset/`（matterport3d、ucl_campus、hkust 等）。`--config_pose_solver` 参数指定 yaml 文件路径。

## 已知问题

- `cannot import name 'cache' from 'functools'`：将 `functools.cache` 替换为 `functools.lru_cache(maxsize=None)`
- `libffi/libtiff` 符号链接问题（ARM 架构）：手动重建 conda 环境中的 `.so` 符号链接
- `cannot allocate memory in static TLS block`：在启动脚本中加入 `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1`

## `third_party` 相关依赖

- `third_party/vismatch`：局部图像匹配模型依赖，对应 `python/utils/utils_image_matching_method.py`
- `third_party/VPR-methods-evaluation`：全局视觉检索模型依赖，对应 `python/utils/utils_vpr_method.py` 和 `python/utils/utils_pipeline.py`
