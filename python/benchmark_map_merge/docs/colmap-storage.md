- feats-ref.h5 存储的是参考地图（reference map）的 SuperPoint 局部特征，格式为 hloc h5 feature 文件：每张图像一个 group，包含 keypoints（N×2 像素坐标）、descriptors（256×N）、scores（N）。
关键特性：它是一个持续增长的累积文件。初始内容来自 sub0 的 feats-sp.h5（复制过来），之后每合并一个新 submap，该 submap 的局部特征就会通过 _append_features 追加进来。到合并第 40 个 submap 时，它已经包含了前 0..37 个成功合并子图的所有帧的 SuperPoint 特征，因此大小达到了 23G。

- global-feats-netvlad.h5 存储的是所有已合并帧的 NetVLAD 全局描述符，每张图像一个 group，包含一个全图级别的向量（shape (D,)，通常 4096 维）。
同样是累积文件：初始来自 sub0，每合并一个新 submap 后通过 _append_features 追加该 submap 的 NetVLAD 描述符。它在每轮 merge 中作为检索数据库（db_descriptors）传入 pairs_from_retrieval.main()，让新进来的 incoming submap 查找最相似的 top-k 参考帧。大小约 210M。

| | feats-inc.h5 | feats-merged.h5 |
|---|---|
| 内容 | 只有本轮 inc submap 的特征 | ref 地图 + 本轮 inc 的合并 |
| 大小 | 约 1-2G（单个 submap） | 约等于当前 feats-ref.h5 大小（随轮次增长） |
| 用途 | 提取后用于 retrieval、匹配，最终 append 进 ref | 匹配时的统一特征索引 |
| 生命周期 | 本轮生成，本轮用完，留在 merge_subN/ 未清理 | 每轮开头删除重建，本轮匹配结束即可丢弃 |