# benchmark_pgo

Robust 位姿图优化基准，用于在接入 map merging 管线前验证方法与参数。

> **更正（结论已被真实数据推翻）**
>
> 本基准在 sphere2500 合成图上得出的两条结论——「TLS 优于 GM」「GM 在低 outlier
> 比例下失效」——**没有迁移到真实数据**。真实序列上 TLS 的硬截断会把子图之间的
> 约束整片切掉、导致地图拆散，GM 的软降权反而稳定；管线最终默认是 `gnc_gm`。
>
> 合成图的边噪声是标定过的，真实管线的 loop 边 sigma 只是启发式，两者的残差归一化
> 尺度不可比，方法排序也就不可比。**方法与参数选择一律以全序列真实实验为准**，
> 本基准只作为接入前的冒烟验证：确认求解器能跑通、能在有 outlier 时收敛。
> 下方全部结论都应放在这个限定里读。

## 背景

`ucl_campus_aria_s00000_in` 的 map merging 会因个别 outlier loop 边产生米级漂移。
原有的 Huber 核是饱和型 M-estimator，超过阈值后权重不再增长但也不归零，坏边照样
把解拽走。本基准用于在接入管线前确认 redescending 核（GNC）能把这类边的权重压到零。

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
