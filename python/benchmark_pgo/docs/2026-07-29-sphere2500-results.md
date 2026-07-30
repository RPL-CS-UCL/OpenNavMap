# sphere2500 robust-PGO 基准结果（2026-07-29）

阶段 1 的验收实验：在接入 map merging 管线之前，先在经典 g2o 数据上确认
GNC-TLS 能扛住注入的 outlier 闭环边。

- 数据：`sphere2500`（2500 位姿 / 4949 因子 / 2450 条真实闭环）
- 参考轨迹：零-outlier 图上 vanilla LM 的解
- 随机种子：42
- 原始数据：[`sphere2500_main.csv`](sphere2500_main.csv)、
  [`sphere2500_barc_sweep.csv`](sphere2500_barc_sweep.csv)

复现：

```bash
cd /Titan/code/robohike_ws/src/opennavmap
export PYTHONPATH=python:third_party/litevloc_code/python

python python/benchmark_pgo/run_benchmark.py \
  --dataset sphere2500 --ratios 0.0 0.1 0.2 0.5 --output /tmp/pgo_benchmark

python python/benchmark_pgo/run_benchmark.py \
  --dataset sphere2500 --methods gnc_tls \
  --ratios 0.1 0.2 0.5 --barc-probs 0.9 0.999 \
  --output /tmp/pgo_benchmark_barc
```

单次约 15 min（16 组 / 6 组，含参考轨迹构建）。

## 主实验

barc_prob = 0.99。P/R 只在 between 因子上统计，prior 不参与。

| ratio | 注入 | method | trans RMSE [m] | rot RMSE [deg] | P | R |
|---|---:|---|---:|---:|---:|---:|
| 0.0 | 0 | none | 0.0 | 0.0 | 1.0 | 1.0 |
| 0.0 | 0 | huber | 0.0 | 0.0001 | 1.0 | 1.0 |
| 0.0 | 0 | gnc_gm | 0.0 | 0.0 | 1.0 | 1.0 |
| 0.0 | 0 | gnc_tls | 0.0 | 0.0 | 1.0 | 1.0 |
| 0.1 | 245 | none | 47.5495 | 97.8228 | 1.0 | 0.0 |
| 0.1 | 245 | huber | 47.9893 | 96.8306 | 1.0 | 0.0 |
| 0.1 | 245 | gnc_gm | 47.5492 | 97.8136 | 1.0 | 0.0 |
| 0.1 | 245 | **gnc_tls** | **0.1898** | **0.4117** | 1.0 | 1.0 |
| 0.2 | 490 | none | 49.2595 | 112.7173 | 1.0 | 0.0 |
| 0.2 | 490 | huber | 48.8624 | 95.2152 | 1.0 | 0.0 |
| 0.2 | 490 | gnc_gm | 0.0001 | 0.0002 | 1.0 | 1.0 |
| 0.2 | 490 | **gnc_tls** | **0.4766** | **0.9524** | 1.0 | 1.0 |
| 0.5 | 1225 | none | 49.8376 | 112.8064 | 1.0 | 0.0 |
| 0.5 | 1225 | huber | 49.7790 | 113.8282 | 1.0 | 0.0 |
| 0.5 | 1225 | gnc_gm | 0.0001 | 0.0003 | 1.0 | 1.0 |
| 0.5 | 1225 | **gnc_tls** | **1.8012** | **3.4480** | 1.0 | 1.0 |

## 通过标准判定

**门槛 2 —— outlier 识别 P/R ≥ 0.95：通过。** GNC-TLS 在 10% / 20% / 50%
三档均为 P = R = 1.0。50% 档意味着 1225 条坏边一条不漏地被剔除，且没有误伤
任何一条真实闭环。

**门槛 3 —— 严格优于 huber：通过。** 平移 ATE 分别是 huber 的 1/253、
1/102、1/28。

Huber 在所有 outlier 比例下 recall 都是 0.0，ATE 与完全不做鲁棒处理
（`none`）几乎相同。这直接坐实了设计文档里的诊断：Huber 是饱和型
M-estimator，超过阈值后梯度变为常数但不归零，残差越大杠杆臂越长，坏边照样
把解拽走。redescending 损失才能把权重压到零。

**门槛 1 —— ATE 增幅：按修订标准通过。** 原标准写的是"相对零-outlier 基线
增幅 < 5%"，但参考轨迹就是零-outlier 时 vanilla LM 自身的解，基线 ATE 恒为
0，比值无定义。改按场景尺度判定：sphere2500 的失败态 ATE 约 50 m，即该图的
尺度量级；GNC-TLS 的 0.19 / 0.48 / 1.80 m 对应 0.4% / 1.0% / 3.6%。

## GM 损失不可用

`gnc_gm` 在 20% 和 50% 下近乎完美（0.0001 m），却在 **10% 下彻底失效**
（47.55 m、R = 0.0）。GM 是软降权，权重渐近趋零但不会真正到零；该档所有权重
停在 0.5 判据之上，等价于没有剔除。

这种随 outlier 比例非单调的失效模式不可接受，因此默认损失定为 TLS——它在三档
全部稳定在 P = R = 1.0。`gnc_gm` 仍保留为 `--pgo_robust` 的可选值，仅供对照。

## barc² 阈值选取

| ratio | barc=0.9 | barc=0.99 | barc=0.999 |
|---|---:|---:|---:|
| 0.1 | 0.1016 | 0.1898 | 0.2857 |
| 0.2 | 0.2404 | 0.4766 | 0.7534 |
| 0.5 | 0.8587 | 1.8012 | 0.0000 |

（平移 ATE [m]；三档 barc_prob 下 P 和 R 全部为 1.0）

三个阈值**剔除的边集完全相同**——P 和 R 都是 1.0，说明被拒集恰好等于注入集。
因此表中的 ATE 差异不可能来自 outlier 处理，只能来自加权 LM 沿不同 mu 调度
收敛到不同的局部极小（sphere2500 从里程计初值出发本就以难收敛著称，
ratio=0.5 时 0.999 反而给出 0.0 也印证了这一点）。

生产参数取 **0.99**。理由不是 sphere2500 上的 ATE（那反而略偏向 0.9），而是
噪声模型的可信度差异：

- 管线里 loop 边用的是 `loop_sigma / conf` 这个启发式 sigma，不是标定过的
  协方差；sphere2500 的边噪声是标定好的，所以收紧到 0.9 也没误杀。真实数据上
  sigma 若偏紧，真实闭环的归一化残差会偏大，收紧阈值就有误杀的风险。
- 我们要打掉的那类边（submap 26 的 `(181,105)`，57.8 m 误差、约 310σ）在三个
  阈值下都会被以巨大余量剔除，不需要靠收紧阈值去够。

宁可给近似噪声模型留余量。

## 结论

阶段 1 三项门槛全部通过，进入阶段 2：`--pgo_robust gnc_tls`、
`--pgo_gnc_barc_prob 0.99`，跑 `--max-submaps 27` 验证 step 26 的修复。
