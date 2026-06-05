# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

### OpenNavMap（顶层系统）

**OpenNavMap** 是一个面向众包建图的多会话建图系统，目标是支持大规模、可扩展的 Image-goal Visual Navigation。整个系统由多个子项目组成，位于 `/Titan/code/robohike_ws/src/`，各子项目协作完成从地图构建、视觉定位到路径规划的完整导航流程。

**子项目及其职责：**

| 子项目 | 功能 |
|--------|------|
| `litevloc_private` | 层次化视觉定位（VPR + 图像匹配 + 位姿求解），ICRA 2025 |
| `MSG` | 多视图场景图构建，用于 3D 场景理解与拓扑定位，NeurIPS 2024 |
| `openintmap` | 关节化对象的最小化运动学场景图映射管道（MoMa-SG） |
| `VPR-methods-evaluation` | VPR 方法统一评估框架（12+ 模型） |
| `VPR-datasets-downloader` | VPR 数据集标准化下载与处理 |
| `f3loc` | 基于平面图的融合过滤定位，CVPR 2024 highlight |

### litevloc（本子项目）

**LiteVLoc** 是 OpenNavMap 的核心定位模块，实现轻量级拓扑度量地图上的高效相机位姿估计。ROS 包名为 `litevloc`，主要语言为 Python 3.8。

## 常用命令

```bash
# 安装环境（Python 3.8 + CUDA 11.8）
conda create --name litevloc python=3.8
conda activate litevloc
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

### 三层层次化定位

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
| `python/loc_pipeline.py` | 主定位流程（VPR → 图像匹配 → 位姿求解），ROS 节点 |
| `python/map_merge_pipeline.py` | 多子地图对齐与合并（SLAM 后端） |
| `python/global_planner.py` | 基于目标图像的最短路径规划 |
| `python/image_graph.py` / `point_graph.py` | 地图图结构（节点 + 边） |
| `python/map_manager.py` | 多图协调管理 |
| `python/pose_fusion.py` | VIO + 视觉定位融合（GTSAM 后端） |

### 目录结构

```
python/
├── *.py                    # 核心算法模块（见上表）
├── ros_*.py                # ROS 节点封装
├── utils/                  # 工具函数（45+ 文件）
│   ├── utils_vpr_method.py     # VPR 模型初始化与推理
│   ├── utils_image_matching_method.py  # 特征匹配封装
│   ├── utils_geom.py           # 位姿转换与误差计算
│   ├── pose_solver.py          # 抽象位姿求解器接口
│   ├── gtsam_pose_graph.py     # GTSAM 后端
│   └── utils_ros/              # ROS 消息转换
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

地图目录必须包含以下文件：

```
map_root/
├── seq/                        # 图像帧目录
│   ├── 000.color.jpg
│   └── 000000.depth.png
├── intrinsics.txt              # fx fy cx cy width height
├── poses.txt                   # img_name qw qx qy qz tx ty tz
├── edge_covis.txt              # [node_a, node_b, weight]
├── edge_odom.txt
├── edge_trav.txt
└── database_descriptors.txt    # VPR 特征（CosPlace 256-D）
```

## 配置系统

使用 YACS 进行层次化配置，数据集配置位于 `python/config/dataset/`（matterport3d、ucl_campus、hkust 等）。`--config_pose_solver` 参数指定 yaml 文件路径。

## 已知问题

- `cannot import name 'cache' from 'functools'`：将 `functools.cache` 替换为 `functools.lru_cache(maxsize=None)`
- `libffi/libtiff` 符号链接问题（ARM 架构）：手动重建 conda 环境中的 `.so` 符号链接
- `cannot allocate memory in static TLS block`：在启动脚本中加入 `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1`

## 外部依赖（需单独克隆）

- `image-matching-models`：`git clone git@github.com:gogojjh/image-matching-models.git --recursive && pip install -e .`
- `VPR-methods-evaluation`：`git clone git@github.com:gogojjh/VPR-methods-evaluation.git`
