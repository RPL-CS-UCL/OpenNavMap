# 多 Session 地图拼接 Baseline — 开发规格文档

**流水线：** HLoc + COLMAP `point_triangulator` + 全局 BA
**策略：** submap 内部 VIO pose 完全固定；跨 submap 合并时做联合 BA 优化 pose
**用途：** 可复现的 baseline，用于与所提方法进行公平对比
**开发策略：** 先用 3 个 submap 跑通并验证，再推广到完整数据集

---

## 1. 问题定义

给定 $N$ 个 submap，每个 submap 包含：
- RGB 关键帧图像序列 $\{I_k\}$
- 对应的 VIO pose $\{T_k^{W_i}\}$，定义在各自的局部坐标系 $W_i$ 下

**目标：** 在统一世界坐标系 $W$ 下生成全局一致的稀疏 3D 地图，其中：
- 每个 submap 的 3D 点在固定 VIO pose 下完成三角化
- 跨 submap 对齐通过视觉检索 + 特征匹配建立
- 最终全局 BA 联合优化所有 pose 和 3D 点

---

## 2. 前提假设与约束

| 项目 | 说明 |
|---|---|
| 输入模态 | 单目 RGB + VIO 6DOF pose（无深度） |
| 关键帧选取 | 由上游基于 VIO pose 运动量预先完成；与所提方法使用完全相同的关键帧集合 |
| VIO pose 质量 | 精度足以支撑 submap 内部三角化，无需 submap 内部 pose 精化 |
| 尺度 | 由 VIO 提供公制尺度 |
| 相机内参 | 已知且全程固定 |
| ROS 依赖 | 无，完全 standalone |

---

## 3. 目录结构

```
project/
├── data/
│   ├── submap_00/
│   │   ├── images/             # 关键帧 RGB 图像
│   │   └── poses.txt           # VIO pose，TUM 格式：timestamp tx ty tz qx qy qz qw
│   ├── submap_01/
│   │   ├── images/
│   │   └── poses.txt
│   ├── submap_02/
│   │   ├── images/
│   │   └── poses.txt
│   └── ...                     # full data 时继续添加
├── workspace/
│   ├── submap_00/
│   │   ├── database.db
│   │   ├── features/           # HLoc 提取的局部特征（.h5）
│   │   ├── retrieval/          # NetVLAD/CosPlace 全局描述子（.h5）
│   │   ├── pairs_within.txt    # submap 内部匹配图像对
│   │   ├── matches_within/     # submap 内部匹配结果（.h5）
│   │   └── sparse/
│   │       ├── prior_model/    # 仅含 VIO pose 的空模型（无 3D 点）
│   │       └── triangulated/   # 三角化后的稀疏模型
│   ├── submap_01/
│   │   └── ...
│   ├── submap_02/
│   │   └── ...
│   └── merged/
│       ├── images_symlink/     # 所有 submap 图像的平铺符号链接目录
│       ├── database.db         # 合并后的总 database
│       ├── retrieval/
│       │   ├── global_features.h5      # 所有图像的全局描述子
│       │   └── cross_submap_pairs.txt  # 跨 submap 检索图像对
│       ├── matches_cross/      # 跨 submap 特征匹配结果（.h5）
│       └── sparse/
│           ├── input/          # 合并前的初始模型
│           └── output/         # 全局 BA 后的最终模型
├── results/
│   ├── poses_before_ba.txt     # 合并前各 submap pose（TUM 格式，用于对比）
│   ├── poses_after_ba.txt      # 全局 BA 后的最终 pose（TUM 格式）
│   └── eval/                   # evo 评估输出
├── config/
│   └── camera.yaml             # 相机内参（fx, fy, cx, cy, W, H）
└── scripts/
    ├── stage1_per_submap.py    # 【阶段一】单 submap 全流程
    ├── stage2_merge.py         # 【阶段二】跨 submap 合并 + 全局 BA + pose 导出
    ├── stage3_eval.py          # 【阶段三】轨迹评估
    └── utils/
        ├── pose_io.py          # VIO pose 与 COLMAP 格式互转
        ├── db_utils.py         # 将 pose/内参写入 COLMAP database
        └── visualize.py        # 中间结果可视化工具
```

---

## 4. 流水线总览

```
【阶段一：单 submap 处理，对每个 submap 独立执行，可并行】
─────────────────────────────────────────────────────────────
Step 1-A  特征提取
  [RGB 关键帧] ──► SuperPoint 特征提取 ──► features/feats-superpoint.h5

Step 1-B  submap 内部匹配
  VIO pose ──► pairs_from_poses ──► pairs_within.txt
  pairs_within.txt + feats ──► LightGlue 匹配 ──► matches_within/*.h5

Step 1-C  写入 COLMAP database + 建立 prior model
  VIO pose + 内参 ──► database.db
                   ──► sparse/prior_model/  （仅含 pose，无 3D 点）

Step 1-D  三角化（VIO pose 固定）
  database.db + prior_model ──► point_triangulator
                             ──► sparse/triangulated/
  ✓ 验证点：检查 3D 点数量、平均重投影误差、COLMAP GUI 可视化

【阶段二：跨 submap 合并，所有 submap 完成阶段一后执行】
─────────────────────────────────────────────────────────────
Step 2-A  全局图像检索
  所有 submap 图像 ──► NetVLAD/CosPlace ──► global_features.h5
                  ──► pairs_from_retrieval（过滤同 submap 内对）
                  ──► cross_submap_pairs.txt
  ✓ 验证点：统计跨 submap 检索到的图像对数量

Step 2-B  跨 submap 特征匹配
  cross_submap_pairs.txt + feats ──► LightGlue ──► matches_cross/*.h5
  ✓ 验证点：统计有效匹配对数（inlier 数 > 30 的对）

Step 2-C  合并 database + 模型合并
  所有 submap database + 跨 submap 匹配 ──► merged/database.db
  所有 triangulated 模型 ──► merge_reconstructions / image_registrator
                         ──► merged/sparse/input/
  ✓ 验证点：确认所有图像均已注册，COLMAP GUI 查看合并前模型

Step 2-D  全局 BA（pose 在此处被优化）
  merged/sparse/input/ ──► bundle_adjuster（refine_extrinsics=True）
                       ──► merged/sparse/output/
  ✓ 验证点：BA 前后重投影误差对比，pose 变化量

Step 2-E  导出 pose 文件
  merged/sparse/output/ ──► results/poses_after_ba.txt（TUM 格式）
  merged/sparse/input/  ──► results/poses_before_ba.txt（TUM 格式，用于对比）

【阶段三：评估】
─────────────────────────────────────────────────────────────
Step 3    ATE 评估
  poses_after_ba.txt + ground_truth.txt ──► evo_ape ──► results/eval/
```

---

## 5. 开发推进策略

### 5.1 第一阶段：3 个 submap 跑通验证

在推广到完整数据集之前，**先用 submap_00、submap_01、submap_02 的数据完整跑通并逐步验证**。

```
开发顺序：
1. 单跑 submap_00 的阶段一，验证三角化结果
2. 单跑 submap_01 的阶段一，验证三角化结果
3. 单跑 submap_02 的阶段一，验证三角化结果
4. 3 个 submap 的阶段二（合并 + BA）
5. 阶段三评估
6. ✓ 所有步骤验证通过 → 进入 5.2
```

脚本调用方式（3 submap 模式）：

```bash
# 阶段一：逐个处理（先跑单个验证，再批量）
python scripts/stage1_per_submap.py --submap_id submap_00
python scripts/stage1_per_submap.py --submap_id submap_01
python scripts/stage1_per_submap.py --submap_id submap_02

# 阶段二：合并
python scripts/stage2_merge.py --submap_ids submap_00 submap_01 submap_02

# 阶段三：评估
python scripts/stage3_eval.py --gt data/ground_truth.txt
```

### 5.2 第二阶段：推广到完整数据集

3 submap 验证通过后，仅需修改 submap 列表即可：

```bash
# 阶段一：批量处理所有 submap
for i in $(seq -w 0 9); do
    python scripts/stage1_per_submap.py --submap_id submap_${i}
done

# 阶段二：合并全部 submap
python scripts/stage2_merge.py --submap_ids $(ls data/ | grep submap)

# 阶段三：评估
python scripts/stage3_eval.py --gt data/ground_truth.txt
```

---

## 6. 分阶段实现细节

---

### 【阶段一】单 submap 处理

**脚本：** `scripts/stage1_per_submap.py --submap_id <name>`

#### Step 1-A：特征提取

```python
from hloc import extract_features

feature_conf = extract_features.confs['superpoint_aachen']
# 大场景可改用 'superpoint_max'

extract_features.main(
    conf=feature_conf,
    image_dir=f'data/{submap_id}/images/',
    export_dir=f'workspace/{submap_id}/features/',
)
```

#### Step 1-B：submap 内部匹配

基于 VIO pose 距离生成图像对（序列匹配，非穷举）：

```python
from hloc import match_features, pairs_from_poses

# 先构建仅含 VIO pose 的临时 COLMAP 模型用于生成图像对
build_prior_model(
    poses_file=f'data/{submap_id}/poses.txt',
    image_dir=f'data/{submap_id}/images/',
    output_path=f'workspace/{submap_id}/sparse/prior_model/',
    camera_yaml='config/camera.yaml',
)

# 生成 submap 内部图像对（top-10 空间近邻）
pairs_from_poses.main(
    model=f'workspace/{submap_id}/sparse/prior_model/',
    output=f'workspace/{submap_id}/pairs_within.txt',
    num_matched=10,
)

# LightGlue 匹配
match_features.main(
    conf=match_features.confs['lightglue_superpoint'],
    pairs=f'workspace/{submap_id}/pairs_within.txt',
    features=f'workspace/{submap_id}/features/feats-superpoint.h5',
    export_dir=f'workspace/{submap_id}/matches_within/',
)
```

#### Step 1-C：写入 COLMAP database

```python
import pycolmap
from utils.pose_io import load_vio_poses_tum
from utils.db_utils import write_cameras_and_images

camera = pycolmap.Camera(
    model='PINHOLE',
    width=W, height=H,
    params=[fx, fy, cx, cy],
)

vio_poses = load_vio_poses_tum(f'data/{submap_id}/poses.txt')

# 注意：COLMAP 使用 T_CW，VIO 通常输出 T_WC，需转换
# cam_from_world = world_from_cam.inverse()

write_cameras_and_images(
    database_path=f'workspace/{submap_id}/database.db',
    camera=camera,
    images=sorted(os.listdir(f'data/{submap_id}/images/')),
    poses=vio_poses,
)

# 同时将匹配结果写入 database
import_matches_to_database(
    database_path=f'workspace/{submap_id}/database.db',
    pairs_file=f'workspace/{submap_id}/pairs_within.txt',
    matches_dir=f'workspace/{submap_id}/matches_within/',
)
```

#### Step 1-D：三角化（VIO pose 固定）

```bash
colmap point_triangulator \
    --database_path workspace/${SUBMAP_ID}/database.db \
    --image_path data/${SUBMAP_ID}/images/ \
    --input_path workspace/${SUBMAP_ID}/sparse/prior_model/ \
    --output_path workspace/${SUBMAP_ID}/sparse/triangulated/ \
    --Triangulation.min_angle 1.5 \
    --Triangulation.max_reproj_error 4.0 \
    --clear_points 1
```

或通过 pycolmap：

```python
pycolmap.triangulate_points(
    database_path=f'workspace/{submap_id}/database.db',
    image_path=f'data/{submap_id}/images/',
    input_path=f'workspace/{submap_id}/sparse/prior_model/',
    output_path=f'workspace/{submap_id}/sparse/triangulated/',
    options={'min_angle': 1.5, 'max_reproj_error': 4.0, 'clear_points': True},
)
```

#### ✅ 阶段一验证检查点

每个 submap 跑完后，执行以下验证再继续：

```python
# utils/visualize.py 提供以下验证函数

# 1. 统计三角化结果
check_triangulation(
    model_path=f'workspace/{submap_id}/sparse/triangulated/',
)
# 期望输出：
# → 注册图像数：N（应等于关键帧总数）
# → 3D 点数：>500（视场景而定）
# → 平均重投影误差：< 2.0 px
# → 平均 track 长度：> 3

# 2. 可视化匹配图像对（抽查）
visualize_matches(
    image_dir=f'data/{submap_id}/images/',
    pairs_file=f'workspace/{submap_id}/pairs_within.txt',
    matches_dir=f'workspace/{submap_id}/matches_within/',
    num_pairs=5,   # 随机抽查 5 对
    output_dir=f'workspace/{submap_id}/viz_matches/',
)

# 3. 用 COLMAP GUI 查看三角化结果（手动）
# colmap gui --database_path workspace/${SUBMAP_ID}/database.db \
#            --image_path data/${SUBMAP_ID}/images/ \
#            --import_path workspace/${SUBMAP_ID}/sparse/triangulated/
```

---

### 【阶段二】跨 submap 合并 + 全局 BA + pose 导出

**脚本：** `scripts/stage2_merge.py --submap_ids submap_00 submap_01 submap_02`

#### Step 2-A：建立图像符号链接目录 + 全局检索

```python
import os
from hloc import extract_features, pairs_from_retrieval

# 建立所有图像的平铺目录（用 symlink，避免复制）
os.makedirs('workspace/merged/images_symlink/', exist_ok=True)
for submap_id in submap_ids:
    for img in os.listdir(f'data/{submap_id}/images/'):
        src = os.path.abspath(f'data/{submap_id}/images/{img}')
        # 用 submap_id 做前缀避免文件名冲突
        dst = f'workspace/merged/images_symlink/{submap_id}__{img}'
        os.symlink(src, dst)

# 提取全局描述子（NetVLAD 或 CosPlace）
retrieval_conf = extract_features.confs['netvlad']
extract_features.main(
    conf=retrieval_conf,
    image_dir='workspace/merged/images_symlink/',
    export_dir='workspace/merged/retrieval/',
)

# 检索跨 submap 图像对，并过滤掉同 submap 内的对
pairs_from_retrieval.main(
    descriptors='workspace/merged/retrieval/global_features.h5',
    output='workspace/merged/retrieval/cross_submap_pairs_raw.txt',
    num_matched=15,
)

# 后处理过滤：去掉 img_name 前缀相同的对（同一 submap）
filter_cross_submap_pairs(
    input_pairs='workspace/merged/retrieval/cross_submap_pairs_raw.txt',
    output_pairs='workspace/merged/retrieval/cross_submap_pairs.txt',
)
```

#### ✅ Step 2-A 验证检查点

```python
# 统计跨 submap 检索结果
count_cross_submap_pairs(
    pairs_file='workspace/merged/retrieval/cross_submap_pairs.txt',
    submap_ids=submap_ids,
)
# 期望输出（3 submap 情况）：
# → submap_00 ↔ submap_01 对数：> 20
# → submap_00 ↔ submap_02 对数：> 20
# → submap_01 ↔ submap_02 对数：> 20
# 若某对 submap 之间对数 < 5，说明视觉重叠不足，需增大 num_matched 或换检索模型
```

#### Step 2-B：跨 submap 特征匹配

```python
from hloc import match_features

match_features.main(
    conf=match_features.confs['lightglue_superpoint'],
    pairs='workspace/merged/retrieval/cross_submap_pairs.txt',
    features='workspace/merged/retrieval/global_features.h5',  # 复用已提取特征
    export_dir='workspace/merged/matches_cross/',
)
```

#### ✅ Step 2-B 验证检查点

```python
# 统计有效匹配对（inlier 数量足够的对）
check_cross_matches(
    pairs_file='workspace/merged/retrieval/cross_submap_pairs.txt',
    matches_dir='workspace/merged/matches_cross/',
    min_inliers=30,   # 少于 30 个 inlier 的对视为无效
)
# 期望输出：
# → 总检索对数：M
# → 有效匹配对数（inlier > 30）：应 > M * 0.3
# → 各 submap 间有效对数分布（不应出现某两个 submap 之间完全无有效对）

# 可视化跨 submap 匹配（抽查）
visualize_matches(
    image_dir='workspace/merged/images_symlink/',
    pairs_file='workspace/merged/retrieval/cross_submap_pairs.txt',
    matches_dir='workspace/merged/matches_cross/',
    num_pairs=10,
    output_dir='workspace/merged/viz_cross_matches/',
)
```

#### Step 2-C：合并 database + 模型合并

```python
# 1. 合并所有 submap 的 database（含跨 submap 匹配）
merge_all_databases(
    submap_databases={sid: f'workspace/{sid}/database.db' for sid in submap_ids},
    cross_pairs='workspace/merged/retrieval/cross_submap_pairs.txt',
    cross_matches_dir='workspace/merged/matches_cross/',
    output_database='workspace/merged/database.db',
    image_name_prefix=True,   # 图像名加 submap 前缀，与 symlink 保持一致
)

# 2. 合并稀疏模型
#    方案 A：pycolmap.merge_reconstructions（要求有共视 3D 点，首选）
try:
    merged = pycolmap.merge_reconstructions(
        [f'workspace/{sid}/sparse/triangulated/' for sid in submap_ids],
        min_common_observations=10,
    )
    merged.write('workspace/merged/sparse/input/')
except Exception:
    # 方案 B：image_registrator（无共视 3D 点时的 fallback）
    # 以 submap_00 为参考，逐步注册其他 submap 的图像
    register_submaps_sequentially(
        reference_model=f'workspace/{submap_ids[0]}/sparse/triangulated/',
        remaining_submaps=submap_ids[1:],
        database='workspace/merged/database.db',
        image_dir='workspace/merged/images_symlink/',
        output_path='workspace/merged/sparse/input/',
    )
```

#### ✅ Step 2-C 验证检查点

```python
check_merged_model(model_path='workspace/merged/sparse/input/')
# 期望输出：
# → 注册图像总数：应等于所有 submap 关键帧之和
# → 未注册图像数：应为 0（或极少）
# → 3D 点总数：应约等于各 submap 3D 点数之和（+跨 submap 新三角化点）

# 用 COLMAP GUI 查看合并前的模型（手动确认各 submap 相对位置合理）
# colmap gui --import_path workspace/merged/sparse/input/
#            --image_path workspace/merged/images_symlink/
```

#### Step 2-D：全局 BA（pose 在此处被优化）

```python
import pycolmap

pycolmap.bundle_adjustment(
    input_path='workspace/merged/sparse/input/',
    output_path='workspace/merged/sparse/output/',
    options=pycolmap.BundleAdjustmentOptions(
        refine_extrinsics=True,         # ← pose 在此处优化（唯一一次）
        refine_focal_length=False,      # 内参固定
        refine_principal_point=False,
        refine_extra_params=False,
    )
)
```

#### ✅ Step 2-D 验证检查点

```python
compare_ba_results(
    before_path='workspace/merged/sparse/input/',
    after_path='workspace/merged/sparse/output/',
)
# 期望输出：
# → BA 前平均重投影误差：X px
# → BA 后平均重投影误差：Y px（应 < X，且 < 1.5 px）
# → 各图像 pose 变化量（平移）：中位数应 < VIO 漂移量级
# → BA 是否收敛（迭代次数、梯度范数）
```

#### Step 2-E：导出 pose 文件

```python
from utils.pose_io import export_poses_tum

# 导出 BA 后的最终 pose
export_poses_tum(
    model_path='workspace/merged/sparse/output/',
    output_file='results/poses_after_ba.txt',
    # 图像名去掉 submap 前缀，恢复原始 timestamp
    strip_prefix=True,
)

# 同时导出合并前（即 VIO pose）用于对比
export_poses_tum(
    model_path='workspace/merged/sparse/input/',
    output_file='results/poses_before_ba.txt',
    strip_prefix=True,
)

print("pose 文件已导出：")
print("  合并前（VIO pose 原始值）：results/poses_before_ba.txt")
print("  合并后（全局 BA 优化值）：results/poses_after_ba.txt")
```

输出格式（TUM）：

```
# timestamp tx ty tz qx qy qz qw
1000.000000 0.123 0.456 0.789 0.000 0.000 0.000 1.000
1000.100000 0.134 0.467 0.791 0.001 0.002 0.001 1.000
...
```

---

### 【阶段三】评估

**脚本：** `scripts/stage3_eval.py --gt data/ground_truth.txt`

#### Step 3：ATE 评估

```bash
# BA 后轨迹评估（主要指标）
evo_ape tum data/ground_truth.txt results/poses_after_ba.txt \
    --align --correct_scale \
    -p --save_results results/eval/ape_after_ba.zip \
    --plot_mode xyz

# BA 前轨迹评估（反映 VIO 漂移，作为参考基线）
evo_ape tum data/ground_truth.txt results/poses_before_ba.txt \
    --align --correct_scale \
    -p --save_results results/eval/ape_before_ba.zip \
    --plot_mode xyz

# 对比 BA 前后（可视化改善量）
evo_res results/eval/ape_before_ba.zip results/eval/ape_after_ba.zip \
    -p --save_table results/eval/comparison_table.csv
```

#### 需上报的指标

| 指标 | 来源 | 说明 |
|---|---|---|
| ATE RMSE（合并前） | `evo_ape` on `poses_before_ba.txt` | VIO 漂移基线 |
| ATE RMSE（合并后） | `evo_ape` on `poses_after_ba.txt` | 主要对比指标 |
| ATE 改善率 | `(before - after) / before` | 衡量 baseline 的有效性 |
| 合并后 3D 点总数 | COLMAP 模型统计 | 地图完整性 |
| 平均重投影误差（BA 后） | COLMAP 模型统计 | 三角化质量 |
| 跨 submap 有效匹配对数 | Step 2-B 验证 | 检索质量 |

---

## 7. 依赖安装

```bash
# Python 依赖
pip install hloc            # https://github.com/cvg/Hierarchical-Localization
pip install pycolmap        # COLMAP Python 绑定
pip install evo             # 轨迹评估
pip install numpy scipy h5py open3d

# COLMAP 二进制（Ubuntu）
sudo apt install colmap
# 或从源码编译以获得最新功能
```

---

## 8. 关键设计决策与依据

| 设计决策 | 依据 |
|---|---|
| submap 内部固定 VIO pose | 将问题聚焦在跨 submap 对齐上，避免混淆 VIO 精度与视觉质量 |
| 合并阶段才优化 pose | 跨 submap 漂移修正由视觉约束驱动，VIO pose 仅作初值 |
| SuperPoint + LightGlue | 低纹理/大视角变化场景下远优于 SIFT，对跨 submap 检索至关重要 |
| submap 内序列匹配 | 利用时序结构，避免 $O(N^2)$ 穷举匹配 |
| 与所提方法使用相同关键帧集合 | 消除关键帧选取作为干扰变量，保证公平对比 |
| 内参全程固定 | VIO 提供已标定内参，无需自标定 |
| 先 3 submap 后 full data | 快速验证流水线正确性，降低调试成本 |
| 图像名加 submap 前缀 | 避免不同 submap 图像文件名冲突，同时保持可追溯性 |

---

## 9. 已知故障模式与处理

| 故障模式 | 触发条件 | 处理方案 |
|---|---|---|
| `merge_reconstructions` 失败 | submap 间无共视 3D 点 | 改用 `image_registrator`，以 submap_00 为参考逐步注册 |
| 跨 submap 有效匹配对数过少 | session 间外观变化大 | 换用 CosPlace，或增大 `num_matched` 至 20~30 |
| 某些图像未被注册 | 关键帧基线过小导致三角化失败 | 调高 `THRESH_TRANS`，或降低 `min_angle` |
| 全局 BA 后误差反而升高 | 跨 submap 错误匹配污染 BA | 在写入 database 前加 RANSAC 几何验证；提高 `min_inliers` 阈值 |
| 图像名冲突 | 不同 submap 中存在同名图像文件 | 确认 symlink 创建时已加 `{submap_id}__` 前缀 |