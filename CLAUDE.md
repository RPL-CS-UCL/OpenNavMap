# CLAUDE.md

## 项目简介

**OpenNavMap**：multi-session topometric mapping + image-goal navigation 系统。**LiteVLoc**（`third_party/litevloc_code`）是必须初始化的 submodule，提供视觉定位、图结构（`image_graph.py`、`point_graph.py` 等）及 `utils/` 共享工具函数。

运行任何 OpenNavMap 脚本时，PYTHONPATH 必须同时包含两个路径：
```bash
export PYTHONPATH=$(pwd)/python:$(pwd)/third_party/litevloc_code/python
```

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

# OpenNavMap 核心 import 验证
PYTHONPATH=$(pwd)/python:$(pwd)/third_party/litevloc_code/python python python/map_merge_pipeline.py --help

# LiteVLoc 离线定位 pipeline
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

## 目录结构

```
python/
├── map_merge_pipeline.py   # 多会话地图构建与合并主入口
├── map_manager.py          # 多图协调管理
├── utils_map_merging.py    # 地图合并工具（opennavmap 独有）
├── gen_covis_trav_edges.py # covis/trav edge 生成脚本（opennavmap 独有）
├── benchmark_mms/          # 多会话建图 benchmark
├── benchmark_vpr/          # VPR 评估
├── benchmark_kf_selection/ # 关键帧选择评估
└── benchmark_map_merge/    # 地图合并评估

third_party/litevloc_code/python/
├── loc_pipeline.py         # LiteVLoc 离线定位主入口
├── ros_loc_pipeline.py     # LiteVLoc 在线定位 ROS 封装
├── global_planner.py       # 基于 trav graph 的全局规划
├── pose_fusion.py          # 里程计与视觉定位融合
├── image_graph.py          # ImageGraph 图结构（opennavmap 共用）
├── point_graph.py          # PointGraph 图结构（opennavmap 共用）
├── utils/                  # 共享工具函数（opennavmap 与 litevloc 共用）
└── config/dataset/         # YACS 数据集配置（单一来源）
```

## 地图数据格式

```
map_root/
├── seq/                        # 图像帧目录
├── timestamps.txt              # img_name timestamp
├── intrinsics.txt              # per-frame: frame_path fx fy cx cy width height
├── poses.txt                   # per-frame: frame_path qw qx qy qz tx ty tz
├── poses_abs_gt.txt            # 可选，绝对位姿 GT
├── gps_data.txt                # 可选
├── iqa_data.txt                # 可选，图像质量评估
├── edges_covis.txt             # [node_a, node_b, weight]
├── edges_odom.txt
├── edges_trav.txt
└── database_descriptors.txt    # VPR 特征
```

**poses.txt 格式（mapfree format）：**`frame_path qw qx qy qz tx ty tz`
- world-to-camera：$R(q), t$ 将世界坐标系点变换到相机坐标系，即 $Rp + t$
- `seq0/frame_00000.jpg` 恒为 identity pose；query 帧位姿相对于参考帧给出

## benchmark_map_merge

- **目录命名规则**：
  - 数据目录：`s00000_aria_data_000`
  - SfM 结果：`s00000_sfm_netvlad_splg_{dist}`（`dist = f"{int(sfm_sample_dist*100):03d}"`，如 `0.25` → `_025`）
  - Merge 结果：`s00000_results_{order_tag}_{method}_{dist}`（无 `_sba{n}` 后缀）
  - `dist=0` 时不追加后缀

- 评估使用 `/Titan/code/robohike_ws/src/slam_trajectory_evaluation`（不用 `evo`）。合并结束后自动调用 `export_to_eval_structure()` 写 TUM 轨迹到 `/Titan/dataset/data_opennavmap/traj_eval_data/map_merge_eval_data`。

- 脚本入口（`python/benchmark_map_merge/scripts/`）：
  ```bash
  bash run_baseline.sh --mode sfm                                    # 为所有 submap 建 SfM
  bash run_baseline.sh --mode sfm --max-submaps 2 --overwrite        # 只建前 2 个
  bash run_baseline.sh --mode merge --max-submaps 2 --overwrite      # 合并并评估
  bash run_evaluation.sh --config map_merge.yaml                     # 单独跑评估
  ```

## 已知问题

- `cannot import name 'cache' from 'functools'`：替换为 `functools.lru_cache(maxsize=None)`
- `libffi/libtiff` 符号链接问题（ARM 架构）：手动重建 conda 环境中的 `.so` 符号链接
- `cannot allocate memory in static TLS block`：启动脚本加 `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1`

## third_party 依赖

- `third_party/litevloc_code`：**必须初始化**（`git submodule update --init --recursive`）。缺少时 opennavmap 主流程无法运行。
- `third_party/vismatch`：图像匹配依赖，由 `litevloc_code/utils/utils_image_matching_method.py` 使用。
- `third_party/VPR-methods-evaluation`：VPR 检索依赖，由 `python/utils_map_merging.py` 使用。
