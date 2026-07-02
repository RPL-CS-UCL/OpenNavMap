# viz_mapfree_data 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建 `viz_mapfree_data.py`，为 map-free 格式数据集可视化一张双子图：上方展示 1 张 query 图像 + 4 张随机 reference 图像，下方展示俯视 pose 轨迹（query 红色，reference 绿色）。

**Architecture:** 单脚本，`argparse` 接收 `--dataset_root`、`--dataset_name`、`--split`、`--scene`，复用 `viz_rpe_pose_results.py` 中相同的 `draw_orientation_arrow` 工具函数和 `PALLETE` 颜色方案。所有数据集的 scene 路径统一为 `<dataset_root>/<dataset_name>/map_free_eval/<split>/<scene>/`。Pose 解析直接读 `poses.txt`（world-to-camera，qw qx qy qz tx ty tz），转换为 4×4 camera-to-world 矩阵后取 x/z 坐标做俯视图。

**Tech Stack:** Python 3, matplotlib, numpy, scipy (Rotation), PIL (Image)

**支持的数据集（路径示例）:**
- `mapfree/map_free_eval/val/s00460/`
- `ucl_campus_aria/map_free_eval/test/s00000/`
- `360loc_aria/map_free_eval/test/s00000/`

---

### Task 1: 数据加载函数 + 单元测试

**Files:**
- Create: `paper_writing/python/viz_mapfree_data.py`
- Test: `paper_writing/python/test/test_viz_mapfree_data.py`

- [ ] **Step 1: 写失败测试 — poses.txt 解析**

```python
# paper_writing/python/test/test_viz_mapfree_data.py
import textwrap
from pathlib import Path
import numpy as np
from paper_writing.python import viz_mapfree_data


def test_load_poses_parses_query_and_refs(tmp_path: Path) -> None:
    poses_file = tmp_path / "poses.txt"
    poses_file.write_text(textwrap.dedent("""\
        #seq0/frame_00000.jpg 0.9525 0.0620 -0.2925 -0.0564 0.6753 0.0501 0.0500
        seq0/frame_00000.jpg 1.0 0.0 0.0 0.0 0.0 0.0 0.0
        seq1/frame_00000.jpg 0.9682 0.0956 0.2145 0.0851 -0.9034 -0.1792 0.3971
        seq1/frame_00001.jpg 0.9683 0.0988 0.2128 0.0849 -0.8941 -0.1803 0.3898
    """), encoding="utf-8")

    poses = viz_mapfree_data.load_mapfree_poses(poses_file)

    assert "seq0/frame_00000.jpg" in poses
    assert "seq1/frame_00000.jpg" in poses
    assert "seq1/frame_00001.jpg" in poses
    assert sum(1 for k in poses if k.startswith("#")) == 0
    T_c2w = poses["seq0/frame_00000.jpg"]
    np.testing.assert_allclose(T_c2w[:3, 3], [0.0, 0.0, 0.0], atol=1e-6)
    assert T_c2w.shape == (4, 4)


def test_load_poses_w2c_inverted_correctly(tmp_path: Path) -> None:
    poses_file = tmp_path / "poses.txt"
    # qw=1 (identity rotation), tx=1 ty=0 tz=0 → T_c2w translation = [-1,0,0]
    poses_file.write_text(
        "seq1/frame_00000.jpg 1.0 0.0 0.0 0.0 1.0 0.0 0.0\n",
        encoding="utf-8",
    )
    poses = viz_mapfree_data.load_mapfree_poses(poses_file)
    T_c2w = poses["seq1/frame_00000.jpg"]
    np.testing.assert_allclose(T_c2w[:3, 3], [-1.0, 0.0, 0.0], atol=1e-6)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /Titan/code/robohike_ws/src/opennavmap
pytest paper_writing/python/test/test_viz_mapfree_data.py -v
```
Expected: `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3: 实现 `load_mapfree_poses`**

- [ ] **Step 4: 运行测试，确认通过**

```bash
cd /Titan/code/robohike_ws/src/opennavmap
pytest paper_writing/python/test/test_viz_mapfree_data.py -v
```
Expected: `2 passed`

- [ ] **Step 5: commit**

```bash
git add paper_writing/python/viz_mapfree_data.py paper_writing/python/test/test_viz_mapfree_data.py
git commit -m "feat: add viz_mapfree_data with load_mapfree_poses"
```

---

### Task 2: `draw_orientation_arrow` + 图像随机采样

**Files:**
- Modify: `paper_writing/python/viz_mapfree_data.py`
- Test: `paper_writing/python/test/test_viz_mapfree_data.py`

- [ ] **Step 1: 写失败测试并实现**

- [ ] **Step 2: 运行测试，确认 4 passed**

```bash
cd /Titan/code/robohike_ws/src/opennavmap
pytest paper_writing/python/test/test_viz_mapfree_data.py -v
```

- [ ] **Step 3: commit**

```bash
git add paper_writing/python/viz_mapfree_data.py paper_writing/python/test/test_viz_mapfree_data.py
git commit -m "feat: add sample_ref_images and draw_orientation_arrow"
```

---

### Task 3: 双子图可视化函数 `visualize_scene`

**Files:**
- Modify: `paper_writing/python/viz_mapfree_data.py`
- Test: `paper_writing/python/test/test_viz_mapfree_data.py`

- [ ] **Step 1: 写失败测试并实现 `visualize_scene`**

- [ ] **Step 2: 运行测试，确认 5 passed**

```bash
cd /Titan/code/robohike_ws/src/opennavmap
pytest paper_writing/python/test/test_viz_mapfree_data.py -v
```

- [ ] **Step 3: commit**

```bash
git add paper_writing/python/viz_mapfree_data.py paper_writing/python/test/test_viz_mapfree_data.py
git commit -m "feat: add visualize_scene two-row figure"
```

---

### Task 4: `main()` CLI + 路径解析 + 真实数据验证

**Files:**
- Modify: `paper_writing/python/viz_mapfree_data.py`

- [ ] **Step 1: 实现 `resolve_scene_dir` 和 `main()`**

- [ ] **Step 2: 测试三个数据集**

```bash
cd /Titan/code/robohike_ws/src/opennavmap
python paper_writing/python/viz_mapfree_data.py --dataset_name mapfree --split val --scene s00460
python paper_writing/python/viz_mapfree_data.py --dataset_name ucl_campus_aria --split test --scene s00000
python paper_writing/python/viz_mapfree_data.py --dataset_name 360loc_aria --split test --scene s00000
```

- [ ] **Step 3: 确认输出**

```bash
ls -lh /Titan/dataset/data_opennavmap/map_free_eval/viz_mapfree_rpe_pose/mapfree/val/s00460.png \
        /Titan/dataset/data_opennavmap/map_free_eval/viz_mapfree_rpe_pose/ucl_campus_aria/test/s00000.png \
        /Titan/dataset/data_opennavmap/map_free_eval/viz_mapfree_rpe_pose/360loc_aria/test/s00000.png
```

- [ ] **Step 4: commit**

```bash
git add paper_writing/python/viz_mapfree_data.py
git commit -m "feat: add main() CLI with unified map_free_eval path resolution"
```
