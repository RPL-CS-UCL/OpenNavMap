"""几何工具：点云 -> 3D 框、点云重合度、密度聚类。

IoU、yaw 提取、鲁棒权重、90 度对齐、余弦相似度这些本仓库的 utils_object_geom.py
已经有了，不重复；这里补它缺的四件事：

  * 把深度图上的一片 mask 反投影成世界系点云（unproject）
  * 点云体素降采样（voxel_downsample）—— 用网格取整 + np.unique，不需要 open3d
  * 从点云算带朝向的 3D 框（obb_from_points）—— 用 cv2.minAreaRect，不用主成分分析
  * 两团点云的重合比例（overlap_ratio）和聚类去噪（dbscan）—— 用 scipy 的 KD 树

为什么 OBB 不用主成分分析（PCA）/ open3d：open3d 的 OBB 按点云方差最大的方向排轴，
导出的"长宽高"三个数到底哪个是高度是不确定的（本仓库早先存出来的地毯"高 0.51 米"
就是这个毛病）。改成先把点云拍到水平面上求最小面积旋转矩形、竖直方向单独取上下界，
第三个数就永远是真实高度，而且不受点云密度分布影响。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from object_node import OBB


# ------------------------------------------------------------------ 反投影

def unproject(depth_m: np.ndarray, mask: Optional[np.ndarray], k: np.ndarray,
              t_ros_cam: np.ndarray, min_depth: float = 0.1,
              max_depth: float = 4.75, erode_px: int = 1,
              max_points: int = 0) -> np.ndarray:
    """把深度图上 mask 覆盖的像素反投影成 ROS 世界系点云，返回 (N, 3) float32。

    mask 传 None 就用整张图（检查地板平不平、墙直不直的时候这么用）。

    erode_px：先把 mask 往里缩 1 个像素。物体边缘那一圈深度值是"物体和背景之间的
    过渡值"，不缩的话会拉出一条飘在空中的假点云拖尾。这招是从
    basic_utils/object_point_cloud_utils/object_point_cloud.py 里搬过来的。

    max_depth 默认 4.75 米 = 传感器量程 5 米的 95%：贴着量程上限的那些点本身就不可信。
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    valid = (depth > min_depth) & (depth < max_depth)
    if mask is not None:
        m = np.asarray(mask)
        if m.dtype != np.uint8:
            m = (m.astype(np.uint8) * 255)
        if erode_px > 0:
            m = cv2.erode(m, None, iterations=int(erode_px))
        valid &= m > 0
    vs, us = np.nonzero(valid)
    if vs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if max_points and vs.size > max_points:
        sel = np.linspace(0, vs.size - 1, max_points).astype(np.int64)
        vs, us = vs[sel], us[sel]

    k = np.asarray(k, dtype=np.float64)
    z = depth[vs, us].astype(np.float64)
    # OpenCV 相机系：x 向右、y 向下、z 沿视线朝前
    x = (us - k[0, 2]) * z / k[0, 0]
    y = (vs - k[1, 2]) * z / k[1, 1]
    pts_cam = np.stack([x, y, z], axis=1)

    t = np.asarray(t_ros_cam, dtype=np.float64).reshape(4, 4)
    pts_world = pts_cam @ t[:3, :3].T + t[:3, 3]
    return pts_world.astype(np.float32)


# ------------------------------------------------------------------ 点云处理

def voxel_downsample(points: np.ndarray, voxel: float = 0.01) -> np.ndarray:
    """体素降采样：把空间切成边长 voxel 米的小格子，每格只留一个点。

    等价于 open3d 的 voxel_down_sample，但只用 numpy —— 把坐标除以格子边长取整，
    再对整数三元组去重就行。默认 1 厘米，室内物体这个精度足够。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0 or voxel <= 0:
        return pts
    keys = np.floor(pts / float(voxel)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def merge_clouds(cloud_a: np.ndarray, cloud_b: np.ndarray,
                 voxel: float = 0.01, max_points: int = 20000) -> np.ndarray:
    """两团点云求并集再降采样；点太多就均匀抽稀，免得内存和 KD 树都吃不消。"""
    pts = np.concatenate([np.asarray(cloud_a, dtype=np.float32).reshape(-1, 3),
                          np.asarray(cloud_b, dtype=np.float32).reshape(-1, 3)], axis=0)
    pts = voxel_downsample(pts, voxel)
    if max_points and pts.shape[0] > max_points:
        sel = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[sel]
    return pts


def overlap_ratio(query: np.ndarray, target: np.ndarray, tau: float = 0.05) -> float:
    """query 里有多大比例的点，能在 target 里找到 tau 米以内的邻居。

    这是判断"两次看到的是不是同一个东西"最管用的信号，尤其对地毯、门、画、电视这类
    又薄又扁的物体 —— 它们的 3D 框重叠率（IoU）天生接近 0，纯看 IoU 会把同一张地毯
    每帧都当成一个新物体，但两帧的点云明明是重合的，这个比例会接近 1。
    """
    from scipy.spatial import cKDTree

    q = np.asarray(query, dtype=np.float32).reshape(-1, 3)
    t = np.asarray(target, dtype=np.float32).reshape(-1, 3)
    if q.shape[0] == 0 or t.shape[0] == 0:
        return 0.0
    dist, _ = cKDTree(t).query(q, k=1, workers=-1)
    return float(np.mean(dist <= tau))


def dbscan(points: np.ndarray, eps: float = 0.1,
           min_samples: int = 10) -> np.ndarray:
    """DBSCAN 密度聚类，返回每个点的簇号（-1 表示噪声点）。

    自己实现（KD 树 + 并查集，就下面这几十行）而不是装 scikit-learn：
    这里只用到一个 DBSCAN，为它往 ROS 环境里塞一个大包不值得。

    干什么用：深度图物体边缘会飘出一些离群点，物体点云里也可能混进旁边的桌子；
    密度聚类能把"主体那一大坨"和"零散飘点"分开，同时还能发现
    "这个节点其实是两把椅子被错合成了一个"（分出两个都很大的簇）。
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    tree = cKDTree(pts)
    neighbors: List[np.ndarray] = tree.query_ball_point(pts, eps, workers=-1,
                                                        return_sorted=False)
    neighbors = [np.asarray(nb, dtype=np.int64) for nb in neighbors]
    is_core = np.array([nb.size >= min_samples for nb in neighbors])

    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # 核心点之间只要互为邻居就并到一起 —— 这就是 DBSCAN 的"密度相连"
    for i in np.nonzero(is_core)[0]:
        for j in neighbors[i]:
            if is_core[j]:
                union(int(i), int(j))

    labels = np.full(n, -1, dtype=np.int64)
    roots: dict = {}
    for i in np.nonzero(is_core)[0]:
        r = find(int(i))
        labels[i] = roots.setdefault(r, len(roots))
    # 边界点：自己不够密，但落在某个核心点的邻域里，就跟着那个簇
    for i in np.nonzero(~is_core)[0]:
        for j in neighbors[i]:
            if is_core[j]:
                labels[i] = labels[j]
                break
    return labels


def cluster_sizes(labels: np.ndarray) -> List[Tuple[int, int]]:
    """把聚类结果整理成 [(簇号, 点数)]，按点数从多到少排，噪声（-1）不算。"""
    out = [(int(c), int((labels == c).sum())) for c in np.unique(labels) if c >= 0]
    return sorted(out, key=lambda kv: -kv[1])


# ------------------------------------------------------------------ 3D 框

def obb_from_points(points: np.ndarray, up_axis: int = 2,
                    min_extent: float = 0.02) -> Optional[OBB]:
    """从点云算一个"只绕竖直轴旋转"的 3D 框（OBB），点太少就返回 None。

    做法：把点拍到水平面上 → 求最小面积旋转矩形（cv2.minAreaRect）拿到中心、长宽、朝向
    → 竖直方向单独取最低最高。这样得到的 size 三个数含义固定：(长, 宽, 高)，
    第三个永远是真实高度，不会像主成分分析那样三个轴顺序乱掉。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        return None
    a0, a1 = [i for i in range(3) if i != up_axis]
    flat = pts[:, [a0, a1]].astype(np.float32)
    try:
        hull = cv2.convexHull(flat.reshape(-1, 1, 2))
        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(hull)
    except cv2.error:
        return None

    lo, hi = float(pts[:, up_axis].min()), float(pts[:, up_axis].max())
    center = np.zeros(3)
    center[a0], center[a1], center[up_axis] = cx, cy, (lo + hi) / 2.0
    size = np.zeros(3)
    size[0] = max(float(w), min_extent)      # 框自己那套轴：第 0、1 个是水平的长和宽
    size[1] = max(float(h), min_extent)
    size[2] = max(hi - lo, min_extent)       # 第 2 个永远是竖直方向的高
    return OBB.from_center_size_yaw(center, size, np.radians(float(angle_deg)), up_axis)


def obb_diag(obb: OBB) -> float:
    """框的对角线长度，用来把"离多远算远"按物体大小缩放。"""
    return float(np.linalg.norm(np.asarray(obb.size, dtype=np.float64).reshape(3)))


def z_range(obb: OBB, up_axis: int = 2) -> Tuple[float, float]:
    """框在竖直方向上占的区间 (最低, 最高)。"""
    c = float(np.asarray(obb.center, dtype=np.float64).reshape(3)[up_axis])
    half = float(np.asarray(obb.size, dtype=np.float64).reshape(3)[2]) / 2.0
    return c - half, c + half


def intervals_overlap(a: Tuple[float, float], b: Tuple[float, float],
                      slack: float = 0.05) -> bool:
    """两个竖直区间有没有交集（留 slack 米余量）。

    用来挡掉"桌上的杯子被合进桌子"这类错误：两者水平位置完全重叠，但高度区间是分开的。
    """
    return (a[0] - slack) <= b[1] and (b[0] - slack) <= a[1]


# ------------------------------------------------------------------ mask 判断

def mask_touches_border(mask: np.ndarray, margin: int = 2) -> bool:
    """mask 是否贴到画面边缘 —— 贴边说明物体被画面裁掉了一部分，这次观测只能看到局部，
    不能拿它去缩小已经建好的框。"""
    m = np.asarray(mask).astype(bool)
    if m.size == 0:
        return False
    k = max(1, int(margin))
    return bool(m[:k, :].any() or m[-k:, :].any() or m[:, :k].any() or m[:, -k:].any())
