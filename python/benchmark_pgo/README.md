# benchmark_pgo

Robust 位姿图优化基准，用于在接入 map merging 管线前验证方法与参数。

## 背景

`ucl_campus_aria_s00000_in` 的 map merging 在 step 26 因单条 outlier loop 边
（conf=0.539、误差 57.8 m）产生 7.7 m 漂移。原有的 Huber 核不是 redescending 的，
无法把这类边的影响压到零。本基准用于验证 GNC-TLS 能否胜任。

## 数据集

来自 SE-Sync 仓库的经典 3D pose-graph benchmark，首次使用时自动下载并缓存到
`/Titan/dataset/data_opennavmap/g2o_benchmark/`。GTSAM 4.2 自带的 Data 目录
只有 toy 级文件，不适用。

| 名称 | 位姿数 | 用途 |
|---|---|---|
| `sphere2500` | 2500 | 主实验（规模接近 step 39 的 ~2900 节点） |
| `smallGrid3D` | 125 | 冒烟测试 |
| `parking-garage` / `torus3D` / `cubicle` | — | 可选扩展 |

## 运行

```bash
cd /Titan/code/robohike_ws/src/opennavmap
export PYTHONPATH=python:third_party/litevloc_code/python

# 主实验：4 方法 × 4 outlier 比例
python python/benchmark_pgo/run_benchmark.py \
  --dataset sphere2500 --output /tmp/pgo_benchmark

# barc² 敏感性扫描
python python/benchmark_pgo/run_benchmark.py \
  --dataset sphere2500 --methods gnc_tls \
  --ratios 0.1 0.2 0.5 --barc-probs 0.9 0.999 \
  --output /tmp/pgo_benchmark_barc
```

## 指标

- **参考轨迹**：sphere2500 无 GT，以零-outlier 时 vanilla LM 的解为参考。
- **ATE**：SE(3) 对齐后的平移 RMSE [m] 与旋转 RMSE [deg]。
- **outlier 识别**：precision / recall，仅在 between 因子上统计（prior 不参与）。

## 通过标准

1. GNC-TLS 在三个 outlier 比例下，trans/rot ATE 相对零-outlier 基线增幅 < 5%。
2. outlier 识别 precision 与 recall 均 ≥ 0.95。
3. GNC-TLS 的 ATE 在每个比例下严格优于 `huber`。

## 注意

本环境安装的 `dash` 自带一个坏掉的 pytest 插件（`typing_extensions has no
attribute 'Generic'`），跑测试时需加 `-p no:dash`：

```bash
python -m pytest python/benchmark_pgo/tests/ -v -p no:dash
```

`tests/` 里涉及 `map_merge_pipeline` 的用例还需要 `pose_estimation_models`
（`utils_map_merging` 会 import `estimator`）：

```bash
PYTHONPATH=python:third_party/litevloc_code/python:third_party/pose_estimation_models \
  python -m pytest tests/ python/benchmark_pgo/tests/ -q -p no:dash
```
