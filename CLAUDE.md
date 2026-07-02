# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

This repository is **OpenNavMap**: a multi-session topometric mapping and scalable image-goal navigation system.

**OpenNavMap** 是当前仓库承载的主系统：一个面向多次采集数据的 **multi-session topometric mapping** 系统，目标是构建、对齐、合并并维护可用于导航的轻量级拓扑度量地图，并进一步支持大规模、可扩展的 Image-goal Visual Navigation。

**LiteVLoc** 是 OpenNavMap 的视觉定位子模块，位于 `third_party/litevloc_code`，并以 pinned git submodule 形式维护。LiteVLoc 负责基于最终构建出的 **multi-session topometric map** 执行 global visual localization，并在在线系统中与路径规划、位姿融合等模块协同。LiteVLoc 的 ROS 包名仍为 `litevloc`，主要语言为 Python 3.8。

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
catkin build opennavmap -DPYTHON_EXECUTABLE=$(which python)

# OpenNavMap 核心 import 验证（不依赖 LiteVLoc submodule）
PYTHONPATH=$(pwd)/python python python/map_merge_pipeline.py --help

# LiteVLoc 离线定位 pipeline（submodule）
PYTHONPATH=$(pwd)/third_party/litevloc_code/python python third_party/litevloc_code/python/loc_pipeline.py \
    --map_path <map_dir> \
    --query_data_path <query_dir> \
    --image_size 512 288 --device=cuda \
    --vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=256 \
    --img_matcher master \
    --pose_solver pnp --config_pose_solver third_party/litevloc_code/python/config/dataset/matterport3d.yaml

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
   - 主入口：`third_party/litevloc_code/python/loc_pipeline.py`、`third_party/litevloc_code/python/ros_loc_pipeline.py`
   - 基于已构建的 topometric map 执行全局检索与局部位姿精化

3. **导航与系统集成**
   - 主入口：`third_party/litevloc_code/python/global_planner.py`、`third_party/litevloc_code/python/pose_fusion.py`
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
| `third_party/litevloc_code/python/image_graph.py` / `point_graph.py` | topometric map 的图结构，LiteVLoc 维护 source-of-truth |
| `third_party/litevloc_code/python/loc_pipeline.py` | LiteVLoc 主定位流程（VPR → 图像匹配 → 位姿求解） |
| `third_party/litevloc_code/python/ros_loc_pipeline.py` | LiteVLoc 在线 ROS 封装 |
| `third_party/litevloc_code/python/global_planner.py` | 基于目标图像的最短路径规划 |
| `third_party/litevloc_code/python/pose_fusion.py` | VIO + 视觉定位融合（GTSAM 后端） |

### 目录结构

```
python/
├── map_merge_pipeline.py   # 多会话地图构建与合并主入口
├── map_manager.py          # 多图协调管理
├── benchmark_mms/          # 多会话建图 benchmark
├── benchmark_vpr/          # VPR 评估
├── benchmark_kf_selection/ # 关键帧选择评估
├── utils/                  # OpenNavMap 核心工具函数
│   ├── utils_map_merging.py    # 地图合并工具
│   ├── gen_covis_trav_edges.py # covis/trav edge 生成
│   ├── utils_geom.py           # 位姿转换与误差计算
│   ├── utils_image.py          # 图像工具（OpenNavMap 本地副本）
│   ├── gtsam_pose_graph.py     # GTSAM 后端

third_party/litevloc_code/
├── python/loc_pipeline.py      # LiteVLoc 离线定位主入口
├── python/ros_loc_pipeline.py  # LiteVLoc 在线定位 ROS 封装
├── python/global_planner.py    # 基于 trav graph 的全局规划
├── python/pose_fusion.py       # 里程计与视觉定位融合
├── python/benchmark_map_free/  # 无地图特征匹配评估
└── python/benchmark_rpe/       # 相对位姿估计评估
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

**intrinsics.txt**
Encodes per frame intrinsics with format
```bash
frame_path fx fy cx cy frame_width frame_height
```

**poses.txt**
Encodes per frame extrinsics with format called mapfree format
```bash
frame_path qw qx qy qz tx ty tz
```
where $q$ is the quaternion encoding rotation and $t$ is the **metric** translation vector. 

Note:
- The pose is given in world-to-camera format, i.e. $R(q), t$ transform a world point $p$ in seq0 to the camera coordinate system in seq1 as $Rp + t$.
- The reference frame (`seq0/frame_00000.jpg`) always has identity pose and the pose of query frames (`seq1/frame_*.jpg`) are given relative to the reference frame. 

## benchmark_map_merge

- **目录命名规则**：
  - 数据目录：`s00000_aria_data_000`（全量数据，无采样过滤）
  - SfM 建图结果：`s00000_sfm_netvlad_splg_{dist}`（`dist = f"{int(sfm_sample_dist*100):03d}"`，例如 `sfm_sample_dist=0.25` → `_025`）
  - Merge 结果：`s00000_results_{order_tag}_{method}_{dist}`（无 `_sba{n}` 后缀）
  - 示例（`sfm_sample_dist=0.25`）：`s00000_results_in_hloc_sfm_netvlad_splg_025`
  - `dist=0` 时不追加后缀

- 评估统一使用 `/Titan/code/robohike_ws/src/slam_trajectory_evaluation`，不使用 `evo`。流程：(1) `run_baseline.py --submap-merge` 执行合并，pipeline 结尾自动调用 `export_to_eval_structure()` 将 TUM 轨迹写入默认路径 `/Titan/dataset/data_opennavmap/traj_eval_data/map_merge_eval_data`；(2) 运行 `scripts/run_evaluation.sh` 计算 ATE（se3 对齐，`map_merge.yaml`）。

- 脚本入口（`python/benchmark_map_merge/scripts/`）：
  ```bash
  # Step 1: 为所有 submap 建 SFM（仅建图，不合并）
  bash run_baseline.sh --mode sfm

  # Step 1: 只建前 2 个 submap 的 SFM，覆写
  bash run_baseline.sh --mode sfm --max-submaps 2 --overwrite

  # Step 2: 合并 sub0+sub1，覆写，并自动运行轨迹评估
  bash run_baseline.sh --mode merge --max-submaps 2 --overwrite

  # Step 2: 合并，指定评估 yaml（默认 map_merge.yaml）
  bash run_baseline.sh --mode merge --max-submaps 2 --eval-config map_merge.yaml

  # 单独运行轨迹评估（不跑合并）
  bash run_evaluation.sh --config map_merge.yaml
  ```

## 配置系统

OpenNavMap root 中的配置仅用于 map-level workflows。LiteVLoc 定位相关 YACS 数据集配置位于 `third_party/litevloc_code/python/config/dataset/`（matterport3d、ucl_campus、hkust 等）。运行 LiteVLoc pipeline 时，`--config_pose_solver` 应指向 submodule 内的 yaml，例如 `third_party/litevloc_code/python/config/dataset/matterport3d.yaml`。

## 已知问题

- `cannot import name 'cache' from 'functools'`：将 `functools.cache` 替换为 `functools.lru_cache(maxsize=None)`
- `libffi/libtiff` 符号链接问题（ARM 架构）：手动重建 conda 环境中的 `.so` 符号链接
- `cannot allocate memory in static TLS block`：在启动脚本中加入 `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1`

## `third_party` 相关依赖

- `third_party/litevloc_code`：LiteVLoc 视觉定位 submodule，对应定位、规划、位姿融合、map-free/RPE benchmarks
- `third_party/vismatch`：局部图像匹配模型依赖，最终由 LiteVLoc 使用
- `third_party/VPR-methods-evaluation`：全局视觉检索模型依赖，最终由 LiteVLoc 使用

