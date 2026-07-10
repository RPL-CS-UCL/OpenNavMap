我想将map-merge的结果使用rerun进行可视化

输入：
1. 可以是已完成map-merging的结果，例如：
/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria/s00000_results_in_kf_spgo_cc_seqmatch
/Titan/dataset/data_opennavmap/map_multisession_eval/ucl_campus_aria/s00000_results_r4_kf_spgo_cc_seqmatch

2. 可以是在线通过run-map-merging.sh在线生成的结果，因此你可以在代码中添加rerun的代码进行内容的存储。用户可以设置跟rerun相关的flag决定是否做可视化。

输出：
一个以keyframe id 为时间轴的rerun文件，可以之后用于结果回播

我想你帮我设计一个可以用rerun进行可视化的方式，可以逐步看到这个地图逐步拼接的过程，能展示出这个一个一个子图出现，然后经过下面的步骤进行逐步拼接的过程。并且要按照时间顺序把中间结果进行记录和展示。

run-map-merging会经过这几步：
1. 加载submap，包括其里程计和图像，以odometry连边+坐标轴+相机和图像的方式可视化这个子图
2. 计算submap的descriptor
3. 以参考图像图像作为reference，另一个submap作为query图像序列，计算其difference matrix

可以参考/Titan/code/robohike_ws/src/opennavmap/paper_writing/python/viz_vpr_data.py的格式对topological localizaiton的结果进行绘制（如import对应的函数）
4. Topological localization - Visual Place Recognition and Sequence Matching：在diffusion matrix上进行基于vpr-dp的搜索，然后得到搜索路径。搜索路径可以以绿色点的方式绘制在difference matrix上. 对dmatrix进行绘制。此外，你绘制匹配的Query（带有ID）和Reference（带有ID）
5. Topological localization - Geometric Verification：根据具体特征点匹配的inlier的数量然后拒绝false positive。要绘制inlier匹配的结果和去除后在difference matrix的结果。

Metric Localization
1. 根据metric localizaiton的结果添加intersubmap匹配节点的连边


node culling
1. 记录keyframe如何被删除和被新的keyframe取代。

Pose Graph Optimization
1. 符合pose graph的节点会在rerun上连上边（绿色），然后紧接着用优化后的posegraph取代当前的pose graph。就完成了submap拼接。
开始下一个拼接。

---

## Archived Prototype: 2026-07-03 First Rerun Implementation

### Status

Archived as a backup and reference only. This prototype is not the target implementation for the next version.

Backup commits on branch `feat/map-merge-rerun-viz`:

- `10cb0b2 backup: save first map merge rerun prototype`
- `a50e5bd chore: remove visualization bytecode from backup`

### Implemented Scope

The first prototype created a `python/visualization/` package with:

- `map_merge_viz_events.py`: event dataclasses and simple geometry helpers.
- `map_merge_result_replay.py`: read-only parser for completed `merge_*` result directories.
- `map_merge_rerun_writer.py`: Rerun `.rrd` writer and blueprint.
- `map_merge_result_to_rerun.py`: CLI for converting existing result directories to `.rrd`.
- `map_merge_viz_recorder.py`: runtime recorder facade that reused the read-only parser.
- `record_rerun_replay_video.py`: `.rrd` post-process video recorder.

It also added optional Rerun flags to:

- `python/map_merge_pipeline.py`
- `python/utils_map_merging.py`
- `scripts/run_map_merging.sh`

### Validation Outputs From Prototype

- `/tmp/opencode/vineyard_map_merge_process_incremental.rrd`
- `/tmp/opencode/vineyard_map_merge_process_incremental_web.mp4`

### Problems Found

The output did not satisfy the intended professional map-merging demonstration.

Main problems:

1. The read-only parser inferred a process from saved final/intermediate result files instead of recording the actual algorithm process.
2. Incremental node and edge appearance was reconstructed from `poses.txt` and `edges_*.txt`, so it did not faithfully represent when the algorithm observed or accepted information.
3. D-matrix, VPR sequence matching path, GV accepted/rejected candidates, metric localization edges, node culling, and PGO transitions were not all tied to exact runtime code points.
4. The `.rrd` was hard to debug because there was no human-readable intermediate event log.
5. Native Rerun video recording through Xvfb failed on this machine due GPU capability limitations; the prototype used Rerun web viewer + Playwright + ffmpeg as a workaround.

### Decision For Next Version

Restart the implementation with a runtime-first design.

V2 should not use completed result directories as the primary source. Instead, while `run_map_merging.sh` / `map_merge_pipeline.py` is running, it should record true runtime events into a visualization folder attached to the merge output:

```text
<output_map_path>/rerun_viz/
  metadata.json
  demo_events.jsonl
  artifacts/
    step_000/
    step_001/
```

The `.rrd` renderer should read this intermediate folder, not directly infer the process from final map files. Existing completed result conversion is postponed until after runtime recording is validated.

The first V2 target is only submap0 + submap1 merging, so the user can inspect `demo_events.jsonl` before expanding to longer sequences.
