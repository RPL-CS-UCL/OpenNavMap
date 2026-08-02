# 持久 Loop 因子 PGO 验收结果

**注意（2026-07-31）**：本文档的数值产出于 `optimize_pose_graph_with_GNC` 修复 `relativeCostTol` 提前终止 bug 之前（旧默认 1e-5，增量合并场景下 GNC 可能 0 次迭代直接返回全 1 权重）。修复后（`relative_cost_tol=0.0`）重跑会得到不同数值，本文数据仅作历史记录。

**注意（2026-08-01）**：本文验收的开关已经删除——持久 loop 因子现在是无条件行为，`--pgo_no_persistent_loops` / `PGO_PERSISTENT_LOOPS` 都不再存在，下方 OFF 一列对应的命令无法再复现。关掉它等于把历史闭环焊成里程计因子，那是本文一开始就要修的故障形态，不是可切换的消融维度。

持久 loop 因子让每一步 merge 的 PGO 用**原始测量** `T_AB` 重建历史
inter-submap 闭环边，而不是从当前位姿把它们重新测量成 odometry 因子。后者的残差
恒为 0，任何 robust kernel 都看不出它错了——这就是坏边一旦进图就被永久焊死的机制。

## 冒烟验收（3 步，开关 OFF / ON 各一次）

```bash
# OFF
PGO_ROBUST=gnc_tls PGO_GNC_BARC_PROB=0.99 \
  OUTPUT_ROOT=/tmp/plp_smoke_off bash scripts/run_map_merging.sh \
  s00000 0 spgo_cc_seqmatch_master_gnctls master 1 1 1 3
# ON
PGO_ROBUST=gnc_tls PGO_GNC_BARC_PROB=0.99 PGO_PERSISTENT_LOOPS=1 \
  OUTPUT_ROOT=/tmp/plp_smoke_on bash scripts/run_map_merging.sh \
  s00000 0 spgo_cc_seqmatch_master_gnctls master 1 1 1 3
```

两次都 `exit=0`，各约 6 min。

| 检查项 | OFF | ON |
|---|---|---|
| `preds/loop_registry.txt` | 不生成 | 生成，step 1 → 16 条，step 2 → 26 条 |
| `gnc_weights.txt` 的 `origin` 列 | 全部 `new` | step 1 全 `new`；step 2 为 23 `new` + 16 `hist` |
| step 1 的 25 条边权重 | — | 与 OFF **逐位一致**（构图时注册表为空） |
| `edge_history.txt` 行尾 | `gv_inlier: N` | `gv_inlier: N` |
| `merge_finalmap` 节点数 | 255 | 255 |
| 最终位姿差 | — | max \|Δt\| = 0.037 m，mean 0.021 m |

节点数相同说明拓扑图未被改动，`submap_disc_0` 的 ATE 口径与 Huber baseline 仍可比。

**因子图无重复边**：注册表中的边被跳过 odom 构建后只以 loop 因子出现一次，
`initial_pose_graph.g2o` 中 `(4,48)`、`(5,49)`、`(42,112)` 各只有 1 条 `EDGE_SE3:QUAT`。

**step 2 的 16 条历史边全部维持 weight 1.0**（`hist_overturned = 0`）——这一步没有
坏边需要翻案，符合预期；机制本身的有效性由单元测试
`tests/test_persistent_loop_pgo.py::test_gnc_overturns_a_registered_bad_edge_once_good_edges_disagree`
证明：一条 180° 翻转的历史边在三条一致的新边出现后被压到 weight 0。

### 顺带修掉的回归

`lloc_history` 用 **submap-local** query id 作键，而 `loop_factor_keys` 现在存的是
**合并后的全局 id**，导致 `gnc_weights.txt` 的 `conf/trans_err/rot_err` 全写成 `nan`
（见上表 OFF 那次运行）。新边查表时减回 `id_offset` 即可；历史边仍是 `nan`，因为
`lloc_history` 只覆盖当前步，它们的 conf 记在 `loop_registry.txt` 里。

`summarize_gnc_weights.py` 同时兼容 6 列（旧）与 7 列（新）格式，并把历史边的重判
结果与本步新边分开统计。冒烟运行的剔除情况：

```
step  total   rej   rej%  justified  max_rej_err  max_kept_err
   1     25     9    36%          6        1.726         0.394
   2     23    13    57%         11       32.405         0.798
```

## 42 submap 定向验收

待运行结束后补充。目标是同时覆盖两个已知故障点：

- **step 26**：坏边 `(181,105)`，Huber baseline 下 ATE 从 0.6 m 跳到 7.7 m。
- **step 40**：VPR 出 110 条候选、GV 只放行 1 条，且这唯一一条是 180° 翻转边
  （`3277,96,weight=1.000000,conf=0.519,Δt=23.537 m,Δrot=179.294°`）。单条闭环边时
  GNC 没有杠杆（PGO 可以整体刚性移动 submap 把残差压到 ~0），所以它在 step 40
  当步仍会被接受；本方案指望的是 step 41–54 出现矛盾证据后把它翻案。

  > **更正**：「GNC 当步无法把这条边判成 outlier」这个观察出自当时的 loose-sigma
  > 配置（`σ_t=0.5` + `conf_scaling=none`，入场半径约为参考配置的 14 倍）。定稿的
  > 参考配置（`σ_t=0.1` + `inverse`）下，同一条边在 step 40 当步就被 GNC 以远低于
  > 0.5 判据的权重拒掉，不需要等后续步的矛盾证据。也就是说「单条闭环边 GNC 必然
  > 无从判定」的推断不成立：残差按收紧后的 sigma 归一化，量级足以让退火过程把权重
  > 压到接近零。这一条同时是删除低连通度守卫的依据——守卫拦的正是这条边。

判据：`loop_registry.txt` 中 `3277,96` 的 `reject_count` 是否随步数增长、
`last_weight` 是否降到 0.5 以下，以及逐步 ATE 是否在 step 40 之后回落。
