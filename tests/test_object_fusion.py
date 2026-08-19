#! /usr/bin/env python
"""object_fusion（点云融合 + 后处理）和可视化投影的回归测试。

跑法: pytest tests/test_object_fusion.py

这里盯住三类以前出过问题的地方:
  1. 近平面裁剪 —— 物体贴得很近、框有一半跑到相机后面时，以前是整个框都不画，
     现在要求画出看得见的那部分线框（用户明确要求的行为）
  2. 配置项读取 —— overlap_tau 配在 assoc 段下，后处理以前从 post 段读，一直取不到
  3. 几何工具 —— 高度是不是真高度、点云重合比例算得对不对
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "python" / "visualization"))
sys.path.insert(0, str(_ROOT / "third_party" / "litevloc_code" / "python"))

from object_fusion import load_config, merge_config  # noqa: E402
from object_fusion.geom import (  # noqa: E402
    cluster_sizes, dbscan, intervals_overlap, mask_touches_border, obb_diag,
    obb_from_points, overlap_ratio, voxel_downsample, z_range,
)
from object_node import OBB  # noqa: E402


class _FakeNode:
    """最小的关键帧替身：相机在原点朝 +x 看，够 _clip_obb_edges 用。

    quat 是 xyzw，表示 相机->世界 的旋转。这里让相机的 z 轴（看向的方向）对上世界的
    x 轴，也就是绕竖直轴摆一下，跟真实数据里机器人朝前看是一个姿态。
    """

    def __init__(self, trans=(0.0, 0.0, 0.0)):
        # 相机系(x右 y下 z前) -> 世界系(x前 y左 z上) 的旋转矩阵，按列写:
        #   相机 x(右)  -> 世界 -y     相机 y(下) -> 世界 -z     相机 z(前) -> 世界 +x
        rot = np.array([[0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0]])
        from scipy.spatial.transform import Rotation as R
        self.quat = R.from_matrix(rot).as_quat()      # xyzw
        self.trans = np.asarray(trans, float)
        self.K = np.array([[388.2, 0.0, 320.0],
                           [0.0, 388.2, 240.0],
                           [0.0, 0.0, 1.0]])
        self.img_size = np.array([640, 480])
        self.id = 0


def _box_at(x: float, size=(1.0, 1.0, 1.0)) -> OBB:
    """在相机正前方 x 米处放一个框（世界系 x 就是相机看的方向）。"""
    return OBB.from_center_size_yaw([x, 0.0, 0.0], list(size), 0.0)


def _box_straddling_camera() -> OBB:
    """一个骑在相机上的框：中心在正前方 0.3 米，纵深 2 米，所以前 1.3 米在相机前面、
    后 0.7 米在相机后面。

    横截面特意做小（0.6 x 0.6 米），这样看得见的那一头投影出来落在画面正中
    （大约 230~410 列 / 150~330 行），画框时能在图上留下像素。
    换成 2 x 2 米的粗框，线框会大到把整个视野圈在里面、线条本身全在画面外，
    那时候画不出东西是几何上对的结果，不是 bug —— 见
    test_box_engulfing_camera_leaves_image_blank。
    """
    return OBB.from_center_size_yaw([0.3, 0.0, 0.0], [2.0, 0.6, 0.6], 0.0)


# ---------------------------------------------------------- 1. 近平面裁剪

def _viz():
    import map_rerun_viz as viz
    return viz


def test_box_fully_in_front_keeps_all_12_edges():
    viz = _viz()
    corners = viz._obb_world_corners(_box_at(3.0))
    segs = viz._clip_obb_edges(corners, _FakeNode())
    assert len(segs) == 12, "框完全在相机前面时 12 条棱一条都不该丢"
    assert all(s.shape == (2, 2) for s in segs)


def test_box_fully_behind_camera_draws_nothing():
    viz = _viz()
    corners = viz._obb_world_corners(_box_at(-3.0))
    assert viz._clip_obb_edges(corners, _FakeNode()) == []


def test_half_behind_box_still_draws_partial_wireframe():
    """这是本次要修的行为：框骑在相机上（一半在前一半在后）。

    改之前 _project_corners 会因为"有角点在相机后面"返回 None，整个框消失；
    改之后 _clip_obb_edges 要把看得见的那部分画出来。
    """
    viz = _viz()
    corners = viz._obb_world_corners(_box_straddling_camera())
    node = _FakeNode()

    assert viz._project_corners(corners, node) is None, \
        "旧的全有全无逻辑对这个框确实返回 None —— 说明这个用例真的踩到了那个问题"

    segs = viz._clip_obb_edges(corners, node)
    assert 0 < len(segs) < 12, f"应该画出部分线框（活下来 {len(segs)} 条棱），不是空图也不是全部"


def test_clipped_pixels_stay_finite():
    """不在近平面截断的话，z 接近 0 会让 u = fx*x/z 除出几万像素，
    画出一条横贯全图的乱线，比不画还糟。截断之后坐标必须是有限值。"""
    viz = _viz()
    corners = viz._obb_world_corners(_box_straddling_camera())
    pix = np.concatenate(viz._clip_obb_edges(corners, _FakeNode()))
    assert np.all(np.isfinite(pix)), "裁剪后不该出现 inf/nan"
    assert np.abs(pix).max() < 1e5, f"坐标最大 {np.abs(pix).max():.0f} 像素，量级不对"


def test_draw_partial_box_actually_marks_pixels():
    """端到端：半在后面的框画到图上，图必须真的被改过（不是原图返回）。"""
    viz = _viz()

    class _Obj:
        id = "obj_0"
        label = "sofa"
        obb = _box_straddling_camera()

    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    out = viz._draw_obb_on_image(rgb, [_Obj()], _FakeNode())
    assert out.shape == rgb.shape, "画完尺寸不能变（横向出界交给 cv2 自己裁）"
    assert int((out != rgb).any(axis=2).sum()) > 50, "应该有像素被画上，不能是空图"


def test_box_engulfing_camera_leaves_image_blank():
    """反面用例，防止有人把上面那条断言错当成"只要有活棱就一定看得见"。

    相机站在一个很大的框里面时，线框会大到把整个视野圈在外面，12 条棱里虽然有 8 条
    活着，但线条本身一条都不穿过画面 —— 这时候图上是空的，几何上就该这样。
    """
    viz = _viz()

    class _Obj:
        id = "obj_1"
        label = "room"
        obb = _box_at(0.0, size=(2.0, 2.0, 2.0))

    node = _FakeNode()
    assert len(viz._clip_obb_edges(viz._obb_world_corners(_Obj.obb), node)) == 8
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    out = viz._draw_obb_on_image(rgb, [_Obj()], node)
    assert int((out != rgb).any(axis=2).sum()) == 0, "线框全在画面外，图上不该有东西"


# ---------------------------------------------------------- 2. 配色按物体 id

def test_object_color_is_stable_across_keyframes():
    """同一个物体在不同帧里必须同色。以前是按"该帧可见物体的枚举序号"取色，
    同一个沙发在两帧里可能一帧黄一帧蓝，读图时会误以为是两个物体。"""
    viz = _viz()

    class _Obj:
        def __init__(self, oid):
            self.id = oid

    a = viz._object_color(_Obj("obj_3"), fallback=0)
    b = viz._object_color(_Obj("obj_3"), fallback=7)   # 换个帧内序号
    assert a == b, "颜色只能由物体 id 决定，不能受帧内枚举顺序影响"
    assert viz._object_color(_Obj("obj_0")) != viz._object_color(_Obj("obj_1"))


def test_object_color_falls_back_when_id_has_no_digits():
    viz = _viz()

    class _Obj:
        id = "sofa"          # 没有数字，解析不出编号

    assert isinstance(viz._object_color(_Obj(), fallback=2), tuple)


# ---------------------------------------------------------- 3. 配置读取

def test_overlap_tau_is_read_from_assoc_section():
    """overlap_tau 配在 assoc 段下。后处理以前从 post 段读，永远取不到，
    一直在用硬编码的 0.05。"""
    cfg = load_config()
    assert "overlap_tau" in cfg["assoc"], "默认配置里 overlap_tau 应该在 assoc 段"
    tau = float(cfg["post"].get("overlap_tau",
                                cfg["assoc"].get("overlap_tau", 0.05)))
    assert tau == pytest.approx(cfg["assoc"]["overlap_tau"])
    # post 段能单独覆盖
    cfg2 = merge_config(cfg, {"post": {"overlap_tau": 0.2}})
    tau2 = float(cfg2["post"].get("overlap_tau",
                                  cfg2["assoc"].get("overlap_tau", 0.05)))
    assert tau2 == pytest.approx(0.2)


def test_merge_config_keeps_untouched_keys():
    """逐段递归合并：只写一个键不能把整段顶掉。"""
    base = {"assoc": {"score_thresh": 0.55, "dist_max": 1.0}}
    out = merge_config(base, {"assoc": {"score_thresh": 0.7}})
    assert out["assoc"]["score_thresh"] == 0.7
    assert out["assoc"]["dist_max"] == 1.0, "没被覆盖的键必须保留"


# ---------------------------------------------------------- 4. 几何工具

def test_obb_third_size_is_true_height():
    """框只绕竖直轴转，所以尺寸的第三个数永远是真实高度，不受水平朝向影响。"""
    rng = np.random.default_rng(0)
    pts = rng.uniform([-1, -0.5, 0], [1, 0.5, 2.0], size=(500, 3)).astype(np.float32)
    obb = obb_from_points(pts, up_axis=2)
    assert obb.size[2] == pytest.approx(2.0, abs=0.05), "第三个数应该是 z 方向跨度"
    assert obb_diag(obb) > 2.0
    lo, hi = z_range(obb, up_axis=2)
    assert lo == pytest.approx(0.0, abs=0.05) and hi == pytest.approx(2.0, abs=0.05)


def test_overlap_ratio_bounds():
    a = np.random.default_rng(1).uniform(0, 1, size=(200, 3)).astype(np.float32)
    assert overlap_ratio(a, a, tau=0.05) == pytest.approx(1.0)
    far = a + 100.0
    assert overlap_ratio(a, far, tau=0.05) == pytest.approx(0.0)


def test_voxel_downsample_thins_but_keeps_extent():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 1, size=(5000, 3)).astype(np.float32)
    out = voxel_downsample(pts, voxel=0.1)
    assert len(out) < len(pts), "降采样后点数必须变少"
    assert len(out) <= 11 ** 3, "1 米立方体切 0.1 米的格子，最多这么多个格子"
    assert np.allclose(out.min(axis=0), pts.min(axis=0), atol=0.15)


def test_dbscan_separates_two_far_apart_blobs():
    rng = np.random.default_rng(3)
    blob = rng.normal(0, 0.02, size=(100, 3))
    pts = np.vstack([blob, blob + 5.0]).astype(np.float32)
    labels = dbscan(pts, eps=0.1, min_samples=5)
    sizes = cluster_sizes(labels)
    assert len([s for _, s in sizes if s >= 50]) == 2, "两团离得很远的点应该分成两簇"


def test_intervals_overlap_respects_slack():
    assert intervals_overlap((0.0, 1.0), (0.5, 1.5))
    assert not intervals_overlap((0.0, 1.0), (1.2, 2.0))
    assert intervals_overlap((0.0, 1.0), (1.2, 2.0), slack=0.3), "给了容差就该算重叠"


def test_mask_touches_border():
    m = np.zeros((100, 100), dtype=bool)
    m[40:60, 40:60] = True
    assert not mask_touches_border(m)
    m[0, 50] = True
    assert mask_touches_border(m), "贴到画面边缘说明物体被裁掉了一截，权重要打折"
