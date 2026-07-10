# CLAUDE.md — benchmark_map_merge

## 概述

HLoc-based multi-session map merging benchmark。基于 hloc + COLMAP SfM 的子图合并评估，
支持 SuperPoint+LightGlue (splg) 和 DISK+LightGlue (disk_dilg) 两种特征方法。

## 环境

```bash
conda activate opennavmap  # Python 3.11, pycolmap, projectaria_tools
```

PYTHONPATH 需包含 opennavmap/python 和 pose_estimation_models/estimator（run_baseline.sh 自动设置）。

## 两步工作流

```bash
cd python/benchmark_map_merge

# Step 1: 为每个 submap 构建 SfM（--sfm-ba-iter 可选，指定 BA 迭代次数）
bash scripts/run_baseline.sh --mode sfm --env ucl_campus_aria --sfm-sample-dist 0.25 --overwrite
bash scripts/run_baseline.sh --mode sfm --env ucl_campus_aria --sfm-sample-dist 0.25 --sfm-ba-iter 10 --overwrite

# Step 2: 合并 submap（基于已有 SfM，不重建）
bash scripts/run_baseline.sh --mode merge --env ucl_campus_aria --sfm-sample-dist 0.25 \
  --prebuilt-sfm-root /path/to/s00000_sfm_netvlad_splg_025 --overwrite

# Step 2 带阈值调优 + 跳过评估（批量 merge 时用，最后统一评估）
bash scripts/run_baseline.sh --mode merge --env ucl_campus_aria \
  --method hloc_sfm_netvlad_disk_dilg --sfm-sample-dist 0.25 \
  --geo-verify-min-matches 300 --pnp-min-inliers 70 \
  --max-submaps 55 --prebuilt-sfm-root /path/to/sfm_root \
  --result-suffix ba10 --clean-work --skip-eval --overwrite
```

## 数据集

| 数据集 | submap 数 | data-dir | sfm-sample-dist |
|--------|----------|----------|-----------------|
| ucl_campus_aria | 55 | s00000_aria_data_000 | 0.25 |
| hkust_campus | 8 | 默认 | 0.25 |
| vineyard | 5 | 默认 | 0.25 |

## 方法与阈值

| 方法 | 特征 | value0 (gvm/pmi) | value1 | value2 |
|------|------|-------------------|--------|--------|
| hloc_sfm_netvlad_splg | SuperPoint+LightGlue | 100/25 | 120/35 | 150/50 |
| hloc_sfm_netvlad_disk_dilg | DISK+LightGlue | 300/70 | 400/110 | 500/150 |

## 评估

```bash
# 合并后自动评估（默认行为）
bash scripts/run_baseline.sh --mode merge ...

# 跳过评估（--skip-eval），最后统一评估
bash scripts/run_evaluation.sh --config OpenNavMap_map_merge.yaml
```

评估使用 `third_party/slam_trajectory_evaluation`，输出 ATE/RPE 到
`/Titan/dataset/data_opennavmap/traj_eval_data/map_merge_eval_data/report/`。

## 命名规则

- SfM：`s00000_sfm_netvlad_{method}_{dist}`（dist=`025`/`390`，dist=0 无后缀）
- SfM 带 BA：`s00000_sfm_netvlad_{method}_{dist}_ba10`（`--result-suffix ba10`）
- Merge：`s00000_results_in[_Nsub]_hloc_sfm_netvlad_{method}_{dist}_value{n}[_ba10]`

## 运行测试

```bash
cd python/benchmark_map_merge
conda activate opennavmap
pytest tests/ -v
```
