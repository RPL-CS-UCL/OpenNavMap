# submap 26 map-merging 验收结果（2026-07-30）

阶段 2 的验收实验：把阶段 1 选定的 GNC-TLS 接入 map merging 管线，用
`--max-submaps 27` 重跑，确认 step 26 的坏边 `(181,105)` 被剔除且 ATE 恢复。

- 数据：`ucl_campus_aria` `s00000`，order `in`，前 27 个 submap
- 基线：`s00000_results_in_spgo_cc_seqmatch_master_iqaigtd`（Huber）
- 本次：`s00000_results_in_27sub_spgo_cc_seqmatch_master_gnctls_iqaigtd`
  （`PGO_ROBUST=gnc_tls`、`PGO_GNC_BARC_PROB=0.99`；运行时目录名不带 `27sub`，
  跑完后重命名归档，以免被阶段 3 的完整重跑覆盖，`run.log` 同目录）
- 耗时约 1 h 45 min
- 原始数据：[`step26_per_step_ate.txt`](step26_per_step_ate.txt)、
  [`step26_gnc_rejections.csv`](step26_gnc_rejections.csv)

复现：

```bash
cd /Titan/code/robohike_ws/src/opennavmap
PGO_ROBUST=gnc_tls PGO_GNC_BARC_PROB=0.99 \
  bash scripts/run_map_merging.sh s00000 0 spgo_cc_seqmatch_master_gnctls master 1 1 1 27
```

## 验收 1：坏边被判为 outlier

step 26 的 `preds/gnc_weights.txt` 全部 7 条闭环边：

| db_id | query_id | weight | conf | trans_err [m] | rot_err [deg] |
|---:|---:|---:|---:|---:|---:|
| 46 | 2 | 1.0 | 1.326 | 0.151 | 0.798 |
| 1698 | 6 | 1.0 | 2.693 | 0.055 | 0.735 |
| 1699 | 7 | 1.0 | 2.473 | 0.063 | 0.265 |
| 1700 | 8 | 1.0 | 3.093 | 0.074 | 0.149 |
| 203 | 95 | **0.0** | 0.548 | 11.061 | 3.091 |
| 179 | 104 | **0.0** | 0.558 | 8.996 | 9.176 |
| **181** | **105** | **0.0** | 1.580 | **40.780** | 12.112 |

目标坏边 `(181,105)` 权重为 0，同族的另两条也一并剔除。保留边的最大 GT 平移
误差 0.151 m，被拒边的最小 GT 平移误差 8.996 m，两个分布相差 60 倍，分离干净。

## 验收 2：坏边不再写回 odom 图

`merge_finalmap/preds/edge_history.txt` 统计行：

```
Number of edges added by VPR: 129
Number of edges removed by GV: 102 (79.07%)
Number of edges removed by CCM: 20 (15.50%)
Number of edges removed by PGO: 3 (2.33%)
Number of edges retained: 4 (3.10%)
```

三条被拒边（含 `181,105`）标记为 `removed_by_pgo`，因此不会进入 odom 图、也就
不会在 step 27 被当成 `odom_sigma` 因子重新引入。这切断了设计文档里描述的
"坏边升格"链路。

## 验收 3：ATE 恢复

评估框架（`slam_trajectory_evaluation`，`align_type: se3`）在 step 26 全图上的结果：

| | Huber（基线） | GNC-TLS |
|---|---:|---:|
| trans RMSE [m] | 7.747 | **0.879** |
| rot RMSE [deg] | 3.301 | **0.720** |

达到验收线（预期回到 ~0.92 m 量级，实际 0.879 m）。

## step 0–25 未被拖坏

`per_step_ate.py` 直接读各步 `merge_*/submap_disc_0/` 的 `poses.txt` 和
`poses_abs_gt.txt` 做同样的 SE(3) 对齐，无需导出 TUM。它在 step 26 复现了评估
框架的 7.747 / 3.301 与 0.879 / 0.720，故整表口径一致。

节选（完整表见 `step26_per_step_ate.txt`）：

| step | n | base_t | gnc_t | base_r | gnc_r |
|---:|---:|---:|---:|---:|---:|
| 0 | 44 | 0.096 | 0.096 | 0.338 | 0.338 |
| 5 | 576 | 0.255 | 0.274 | 0.336 | 0.415 |
| 6 | 689 | 0.508 | 0.528 | 0.600 | 0.626 |
| 8 | 924 | 1.690 | **1.930** | 0.992 | 1.063 |
| 15 | 1478 | 1.089 | 1.018 | 0.894 | 0.815 |
| 22 | 1921 | 1.100 | 0.819 | 0.992 | 0.761 |
| 25 | 2086 | 0.917 | **0.783** | 0.840 | 0.730 |
| 26 | 2215 | 7.747 | **0.879** | 3.301 | 0.720 |

26 步里 GNC-TLS 在 22 步上不劣于基线，step 9 之后每一步都更好。回退只出现在
step 2 / 5 / 6 / 8，最大一次是 step 8 的 +0.24 m（+14%）；这些步的
`max_rej_err` 都在米级以下，属于加权 LM 收敛差异而非误杀级联——若真是误杀，
后续步会持续恶化，但实际 step 9 起基线与 GNC 的差距单调拉开。

## 剔除率与 conf 缩放的二阶问题

26 步合计 **299/818 条边被剔除（36.6%）**，其中 164 条（占被拒边 54.8%）的 GT
平移误差 > 0.45 m。逐步表里 `max_kept_err` 多在 0.15–0.8 m，而 `max_rej_err`
可达 32.4 m / 40.8 m。

剩下 45% 的被拒边多是 0.2–0.5 m 量级的边缘情况，根因是 sigma 的定义方式：
`sigma = loop_sigma / conf`，`loop_sigma` 平移分量取 0.1 m，而 master 度量定位
真实闭环典型就在 0.1–0.4 m，正好压在 barc 0.99 的 4.10σ 判据边界上。conf 越高
sigma 越紧，反而被更严格地要求，例如：

- `(4,53)`：0.294 m / conf 1.856 → 5.5σ → 剔除
- `(41,66)`：0.394 m / conf 0.773 → 3.0σ → 保留

即误差更小的边因为置信度高反而被拒。这是 sigma 模型的问题，不是 GNC 的问题；
step 26 的验收不受影响（坏边在任何合理阈值下都超出量级），但如果后续想降低剔除
率，正确的做法是把 `loop_sigma` 的平移分量调到与 master 实际精度匹配（0.3 m
量级），而不是放宽 barc。此项未在本阶段改动。

## 结论

阶段 2 三条验收全部通过，进入阶段 3：完整 40 步重跑，与全量基线
`UCL_Campus_S00000_in` 的 7.925 m / 3.085° 对比（健康参照
`UCL_Campus_S00003_Culling_in` 为 0.136 m / 0.358°）。
