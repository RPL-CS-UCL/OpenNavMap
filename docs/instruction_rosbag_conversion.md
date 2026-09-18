# 从 rosbag 到多会话建图：数据转换说明

这条链把一个 ROS1 rosbag（一次整段录制）变成 OpenNavMap 的多会话建图输入，
再直接跑合并与评估。它**不依赖 ROS**（用纯 Python 的 `rosbags` 库读 bag），
所以在没有 `roscore` 的机器上也能跑。

流程分四步，一条命令跑完：

```bash
export PYTHONPATH=$(pwd)/python:$(pwd)/third_party/litevloc_code/python
bash scripts/run_rosbag_to_multisession.sh python/rosbag_convert/config/cmt_szs_odin1.yaml
```

| 步骤 | 做什么 | 用到的脚本 |
|---|---|---|
| A | 读 bag → 图像和位姿按时间戳对齐 → 按路程切成 N 段模拟"多次采集" → 每段按距离/角度挑关键帧 → 写成子图目录 | `python/rosbag_convert/convert_rosbag_to_multisession.py` |
| B | 给每张关键帧算 VPR 描述子（用来找"我好像来过这里"的候选）和图像质量分 | `litevloc_code/python/utils/extract_vpr_descriptors.py`、`extract_iqa.py` |
| C | 在每个子图内部加"共视边"（两张图看到同一片区域）和"可通行边"（两点之间能走） | `python/gen_covis_trav_edges.py` |
| D | 按顺序把子图合并成一张地图，并和真值轨迹比 ATE | `scripts/run_map_merging.sh` |

`--skip-convert` 跳过 A–C（数据已生成，只想重跑合并），`--skip-merge` 只做 A–C。

## 给一个新 bag 写 YAML

每个 bag 一个 YAML，放在 `python/rosbag_convert/config/`。换 bag 只改 YAML，不改代码。
字段（省略的用默认值）：

| 字段 | 含义 | 怎么找到值 |
|---|---|---|
| `bag_path` | bag 文件路径（必填） | |
| `output_root` | 数据集根目录（必填），子图目录和 `<scene>_orders.txt` 都写在这下面 | 例：`/Titan/dataset/data_opennavmap/map_multisession_eval/<name>` |
| `scene` / `sensor_tag` | 拼目录名：`<scene>_<sensor_tag>_data_<关键帧间距×100>`，如 `s00000_odin_data_390` | 与 ucl 数据 `s00000_aria_data_000` 同一套规则 |
| `image_topic` | `sensor_msgs/Image` 话题，支持 `bgr8`/`rgb8`/`mono8` | `rosbag info <bag>` |
| `camera_info_topic` | `sensor_msgs/CameraInfo`，只取第一条当全程内参 | 同上 |
| `odometry_topic` | `nav_msgs/Odometry`，机体（IMU/base）在里程计系下的位姿 | 同上 |
| `tf_topic` | `tf2_msgs/TFMessage` 话题 | 通常 `/tf` |
| `odom_frame` / `body_frame` / `camera_frame` | 里程计系、机体系、相机光学系的 frame id | 看 `/tf` 里的 `frame_id` / `child_frame_id`，或 Odometry 的 `header.frame_id` / `child_frame_id` |
| `camera_pose_source` | `tf`：直接读 `/tf` 里 `odom_frame -> camera_frame`（bag 里有相机位姿时用这个）；`odometry_static_extrinsic`：用里程计 × 固定外参（bag 里只有机体位姿时用） | 先 `grep` 一下 `/tf` 有没有 `camera_frame` |
| `T_body_camera_translation` / `T_body_camera_quat_xyzw` | 机体→相机的固定外参（平移 m，四元数 x y z w）；`odometry_static_extrinsic` 模式必填，`tf` 模式只是备用 | 标定文件，或从 `/tf` 里 `body->camera` 反推 |
| `sync_tolerance_s` | 图像与位姿时间戳最多差多少秒还算同一时刻 | 图像周期的一半以内 |
| `kf_trans_thresh_m` / `kf_rot_thresh_deg` | 关键帧规则：离上一张关键帧走了这么远**或**转了这么多度就留一张 | 论文口径 3.9 m / 60° |
| `num_sessions` | 切成几段模拟多会话 | |
| `split_mode` | `equal_distance`：按走过的路程等分；`time_boundaries`：按 `split_time_boundaries_s` 给的相对秒数切 | |
| `anchor_to_first_body` | `true` 时每段 `poses.txt` 以该段第一张关键帧的**机体**位姿为原点（z 轴朝上，与 ucl 数据一致）；`poses_abs_gt.txt` 始终是全局里程计系 | 一般保持 `true` |
| `jpeg_quality` | 导出图片的 JPEG 质量 | |

## 输出目录

```
<output_root>/
├── <scene>_orders.txt                 # 首行 "0 1 2 ..."：合并顺序
└── <scene>_<sensor>_data_<dist>/
    └── <session_id>/                  # 0, 1, 2, ...
        ├── seq/000000.color.jpg ...   # 关键帧，编号 = poses.txt 行号
        ├── timestamps.txt             # name 秒
        ├── intrinsics.txt             # name fx fy cx cy w h
        ├── poses.txt                  # name qw qx qy qz tx ty tz（world-to-camera，会话本地系）
        ├── poses_abs_gt.txt           # 同格式，全局里程计系（当伪真值）
        ├── gps_data.txt               # name + 5 个 nan（合并流程要求存在）
        ├── edges_odom.txt             # 相邻关键帧链 "i j 距离"
        ├── edges_covis.txt / edges_trav.txt   # 步骤 C 之后被加密
        ├── database_descriptors.txt   # 步骤 B 产出
        └── iqa_data.txt               # 步骤 B 产出
```

合并结果在 `<output_root>/<scene>_results_in_<method>_iqaigtd/`，最终地图是
`merge_finalmap/submap_disc_0/`；评估报告在
`/Titan/dataset/data_opennavmap/traj_eval_data/<output_root 目录名>_eval_data/report/`。

## 关于真值

bag 里如果没有独立的真值（动捕/RTK），`poses_abs_gt.txt` 就是里程计本身。此时 ATE
衡量的是"合并结果与机载里程计的一致性"，不是绝对精度。

## 常见报错

| 报错 | 原因 / 处理 |
|---|---|
| `BagReadError: Topic '...' not found in bag` | YAML 里话题名写错，用 `rosbag info` 核对 |
| `BagReadError: ... has non-zero distortion` | 图像没去畸变；地图格式只支持针孔无畸变图，先在录制端去畸变 |
| `BagReadError: No 'odom' -> 'camera_0' transform on '/tf'` | `/tf` 里没有相机位姿，改 `camera_pose_source: odometry_static_extrinsic` 并填外参 |
| `ValueError: A submap needs at least 3 keyframes` | 某段太短，减小 `kf_trans_thresh_m` 或减少 `num_sessions` |
| `ValueError: Split produced an empty session` | 机器人几乎没动（路程为 0），改用 `time_boundaries` |
| `WARNING: sessions did not all connect` | 合并后地图分成了多块（回环没找到），ATE 只覆盖第一块；先看 `preds/loop_registry.txt` |

## 单元测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=python:third_party/litevloc_code/python \
  python -m pytest python/rosbag_convert/tests -v
```
`test_bag_smoke.py` 需要真实 bag，文件不存在时自动跳过。
