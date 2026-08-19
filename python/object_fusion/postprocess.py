"""全部帧跑完之后的一遍全局整理。

逐帧建图只能看到"到目前为止"的信息，有些错误必须等全部看完才判得出来：
哪块点云是飘出来的噪声、哪两个其实是同一个东西、哪一个其实是两把椅子、
哪一大片是地板不是物体。这一遍就干这些，是离线建图相比在线建图最大的好处。

顺序（有先后依赖，别调换）：
  1. 去噪     每个物体的点云做密度聚类，把零散飘点扔掉
  2. 剔幽灵   只被看到过一次、或点太少的，多半是误检，扔掉
  3. 合漏合   同类物体两两之间点云重合或框重叠够多，合成一个
  3.5 认错类  两坨点云几乎完全长在一起、但被检成了不同类别，说明是同一件东西
              被检测器在不同帧叫成了不同名字，按观测次数多的那个名字合成一个。
              默认关闭（post.merge_cross_label: false），因为它会削弱"椅子不许并进沙发"
              那道防线；开之前先看 §已知问题
  4. 拆错合   一个物体的点云明显分成相距很远的两坨，拆回两个
  5. 去地板墙 又薄又大的一片，是地板或墙面，不是物体
  6. 收尾     重算框、重算置信度、把 id 重新编号
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from object_node import ObjectNode
from utils_object_geom import iou_3d, normalize

from object_fusion.geom import (
    cluster_sizes, dbscan, merge_clouds, obb_from_points, overlap_ratio,
)


def postprocess(graph: Any, cfg: Dict[str, Any]) -> Dict[str, int]:
    """就地整理 graph，返回每一步动了多少个物体，方便写进 map_meta.json。"""
    p = cfg.get("post", {})
    # 判断"两个点算重合"的距离容差。它配在 assoc 段下（关联时也用同一个口径），
    # 但后处理这边也允许用 post 段单独覆盖，所以按 post -> assoc -> 默认值取。
    tau = float(p.get("overlap_tau", cfg.get("assoc", {}).get("overlap_tau", 0.05)))
    # 合并点云要用的体素边长和点数上限：PointFusionGraph 自己带这两个属性，
    # boxfusion 用的是原版 ObjectGraph 没有，这里按配置补上，两条路口径一致。
    f = cfg.get("fuse", {})
    if not hasattr(graph, "voxel"):
        graph.voxel = float(f.get("voxel", 0.01))
    if not hasattr(graph, "max_points"):
        graph.max_points = int(f.get("max_points_per_object", 20000))
    stats = {"denoised": 0, "dropped_ghost": 0, "merged": 0,
             "split": 0, "dropped_structure": 0}

    stats["denoised"] = _denoise(graph, p)
    stats["dropped_ghost"] = _drop_ghosts(graph, p)
    stats["merged"] = _merge_leftovers(graph, p, tau)
    if bool(p.get("merge_cross_label", False)):
        stats["merged_cross_label"] = _merge_cross_label(graph, p, tau)
        stats["merged"] += stats["merged_cross_label"]
    stats["split"] = _split_wrongly_merged(graph, p)
    if not bool(p.get("keep_structure", False)):
        stats["dropped_structure"] = _drop_structure(graph, p)
    _finalize(graph)
    print(f"[postprocess] 去噪 {stats['denoised']} 个 / 剔除误检 {stats['dropped_ghost']} 个 / "
          f"合并 {stats['merged']} 次 / 拆分 {stats['split']} 次 / "
          f"去掉地板墙面 {stats['dropped_structure']} 个 -> 最终 {len(graph.nodes)} 个物体",
          flush=True)
    return stats


# ------------------------------------------------------------------ 1. 去噪

def _denoise(graph: Any, p: Dict[str, Any]) -> int:
    eps = float(p.get("dbscan_eps", 0.1))
    min_samples = int(p.get("dbscan_min_samples", 10))
    changed = 0
    for node_id, pts in list(graph.clouds.items()):
        if pts is None or len(pts) < min_samples:
            continue
        labels = dbscan(pts, eps, min_samples)
        sizes = cluster_sizes(labels)
        if not sizes:
            continue
        # 保留够大的簇（占总点数 10% 以上）。留多个是故意的：万一真是两把椅子
        # 被合成了一个，第 4 步还要靠这些簇把它拆开。
        keep_min = max(min_samples, int(0.1 * len(pts)))
        keep = [c for c, n in sizes if n >= keep_min] or [sizes[0][0]]
        mask = np.isin(labels, keep)
        if mask.all():
            continue
        graph.clouds[node_id] = np.asarray(pts)[mask]
        changed += 1
    return changed


# ------------------------------------------------------------------ 2. 剔幽灵

def _drop_ghosts(graph: Any, p: Dict[str, Any]) -> int:
    min_obs = int(p.get("min_observations", 2))
    min_points = int(p.get("min_points", 40))
    doomed = [n.id for n in graph.nodes.values()
              if n.num_observations < min_obs
              or len(graph.clouds.get(n.id, ())) < min_points]
    for node_id in doomed:
        _remove(graph, node_id)
    return len(doomed)


# ------------------------------------------------------------------ 3. 合漏合

def _merge_leftovers(graph: Any, p: Dict[str, Any], tau: float) -> int:
    thr_overlap = float(p.get("merge_overlap", 0.5))
    thr_iou = float(p.get("merge_iou", 0.4))
    merged = 0
    changed = True
    while changed:                       # 合完可能又跟第三个够近了，所以迭代到不再变
        changed = False
        ids = list(graph.nodes)
        for i, id_a in enumerate(ids):
            if id_a not in graph.nodes:
                continue
            for id_b in ids[i + 1:]:
                if id_b not in graph.nodes or id_a not in graph.nodes:
                    continue
                a, b = graph.nodes[id_a], graph.nodes[id_b]
                if not _label_ok(graph, a.label, b.label):
                    continue
                pa, pb = graph.clouds.get(id_a), graph.clouds.get(id_b)
                ov = overlap_ratio(pb, pa, tau) if pa is not None and pb is not None else 0.0
                if ov < thr_overlap and float(iou_3d(a.obb, b.obb, graph.up_axis)) < thr_iou:
                    continue
                _absorb(graph, a, b)
                merged += 1
                changed = True
    return merged


# --------------------------------------------------------- 3.5 认错类（默认关）

def _merge_cross_label(graph: Any, p: Dict[str, Any], tau: float) -> int:
    """把"其实是同一件东西、却被检成不同类别"的节点并起来。

    为什么门槛比同类的那一步高得多（默认 0.7 而不是 0.5）：
    类别不同这件事本身就是"它们不是一个东西"的证据，要推翻它，几何证据必须非常硬。
    这里比的是点云重合度而不是框重叠——挨着放的两把椅子框会重叠，但各自的点
    不会有七成落在对方 5 厘米以内；而同一张沙发被检成 sofa 和 bed 时会。
    合并后留观测次数多的那个类别名，被丢掉的名字记进 quality.merged_labels 备查。
    """
    thr = float(p.get("cross_label_overlap", 0.7))
    merged = 0
    changed = True
    while changed:
        changed = False
        ids = list(graph.nodes)
        for i, id_a in enumerate(ids):
            if id_a not in graph.nodes:
                continue
            for id_b in ids[i + 1:]:
                if id_b not in graph.nodes or id_a not in graph.nodes:
                    continue
                a, b = graph.nodes[id_a], graph.nodes[id_b]
                if _label_ok(graph, a.label, b.label):
                    continue                      # 同类的上一步已经处理过了
                pa, pb = graph.clouds.get(id_a), graph.clouds.get(id_b)
                if pa is None or pb is None:
                    continue
                # 取双向重合度里大的那个：小物体整个长在大物体里时，
                # 小的那边比例接近 1，大的那边很低，用大的才判得出来
                ov = max(overlap_ratio(pb, pa, tau), overlap_ratio(pa, pb, tau))
                if ov < thr:
                    continue
                keep, drop = (a, b) if a.num_observations >= b.num_observations else (b, a)
                dropped_label = drop.label
                _absorb(graph, keep, drop)
                if hasattr(graph, "quality"):
                    graph.quality.setdefault(keep.id, {}) \
                        .setdefault("merged_labels", []).append(str(dropped_label))
                merged += 1
                changed = True
    return merged


# ------------------------------------------------------------------ 4. 拆错合

def _split_wrongly_merged(graph: Any, p: Dict[str, Any]) -> int:
    eps = float(p.get("dbscan_eps", 0.1))
    min_samples = int(p.get("dbscan_min_samples", 10))
    min_frac = float(p.get("split_min_frac", 0.3))
    min_dist = float(p.get("split_dist", 1.0))
    splits = 0
    for node_id in list(graph.nodes):
        pts = graph.clouds.get(node_id)
        if pts is None or len(pts) < 2 * min_samples:
            continue
        labels = dbscan(pts, eps, min_samples)
        sizes = cluster_sizes(labels)
        big = [c for c, n in sizes if n >= min_frac * len(pts)]
        if len(big) < 2:
            continue
        parts = [np.asarray(pts)[labels == c] for c in big]
        centers = [part.mean(axis=0) for part in parts]
        # 只有两坨点确实离得远才拆；离得近多半就是同一个物体中间有个缺口
        far = max(float(np.linalg.norm(centers[i] - centers[j]))
                  for i in range(len(centers)) for j in range(i + 1, len(centers)))
        if far < min_dist:
            continue
        _split(graph, node_id, parts)
        splits += 1
    return splits


# ------------------------------------------------------------------ 5. 去地板墙面

def _drop_structure(graph: Any, p: Dict[str, Any]) -> int:
    z_span = float(p.get("floor_z_span", 0.08))
    area_min = float(p.get("floor_area", 4.0))
    doomed = []
    for node in graph.nodes.values():
        size = np.asarray(node.obb.size, float).reshape(3)
        area = float(size[0] * size[1])
        if size[2] <= z_span and area >= area_min:          # 平躺的一大片 -> 地板
            doomed.append(node.id)
        elif min(size[0], size[1]) <= z_span and size[2] * max(size[0], size[1]) >= area_min:
            doomed.append(node.id)                          # 立着的一大片 -> 墙
    for node_id in doomed:
        _remove(graph, node_id)
    return len(doomed)


# ------------------------------------------------------------------ 6. 收尾

def _finalize(graph: Any) -> None:
    """重算框、把观测质量汇总进 quality、id 重新从 obj_0 开始编。"""
    if not hasattr(graph, "quality"):
        graph.quality = {}
    for node in graph.nodes.values():
        pts = graph.clouds.get(node.id)
        if pts is not None and len(pts):
            obb = obb_from_points(pts, graph.up_axis)
            if obb is not None:
                node.obb = obb
                node.trans = np.asarray(obb.center, float).reshape(3)
        q = graph.quality.setdefault(node.id, {})
        weights = q.pop("obs_weights", None)
        if weights:
            q["mean_obs_weight"] = round(float(np.mean(weights)), 4)
        q["n_points"] = int(len(pts)) if pts is not None else 0
        q["num_observations"] = int(node.num_observations)

    nodes, clouds, quality = {}, {}, {}
    for i, node in enumerate(graph.nodes.values()):
        old = node.id
        node.id = f"obj_{i}"
        nodes[node.id] = node
        if old in graph.clouds:
            clouds[node.id] = graph.clouds[old]
        quality[node.id] = graph.quality.get(old, {})
    graph.set_node(nodes)
    graph.clouds.clear()
    graph.clouds.update(clouds)
    graph.quality.clear()
    graph.quality.update(quality)


# ------------------------------------------------------------------ 小工具

def _label_ok(graph: Any, label_a: str, label_b: str) -> bool:
    vocab = getattr(graph, "vocab", None)
    if vocab is not None:
        return bool(vocab.compatible(label_a, label_b))
    return str(label_a).strip().lower() == str(label_b).strip().lower()


def _remove(graph: Any, node_id: str) -> None:
    graph.nodes.pop(node_id, None)
    graph.clouds.pop(node_id, None)
    if hasattr(graph, "quality"):
        graph.quality.pop(node_id, None)


def _absorb(graph: Any, keep: ObjectNode, drop: ObjectNode) -> None:
    """把 drop 并进 keep：点云求并、观测数相加、特征按置信度加权平均。"""
    voxel = float(getattr(graph, "voxel", 0.01))
    max_points = int(getattr(graph, "max_points", 20000))
    graph.clouds[keep.id] = merge_clouds(graph.clouds.get(keep.id),
                                         graph.clouds.get(drop.id), voxel, max_points)
    w_keep, w_drop = max(keep.confidence, 1e-6), max(drop.confidence, 1e-6)
    for key, vec in drop.embeddings.items():
        if key in keep.embeddings:
            keep.embeddings[key] = normalize(
                (w_keep * keep.embeddings[key] + w_drop * vec) / (w_keep + w_drop))
        else:
            keep.embeddings[key] = vec
    keep.num_observations += drop.num_observations
    keep.confidence = float((w_keep * keep.confidence + w_drop * drop.confidence)
                            / (w_keep + w_drop))
    keep.last_verified_step = max(keep.last_verified_step, drop.last_verified_step)
    keep.observed_keyframes = list(keep.observed_keyframes) + list(drop.observed_keyframes)
    if hasattr(graph, "quality"):
        qa = graph.quality.setdefault(keep.id, {})
        qb = graph.quality.get(drop.id, {})
        qa.setdefault("obs_weights", []).extend(qb.get("obs_weights", []))
        qa.setdefault("merged_from", []).append(drop.id)
    obb = obb_from_points(graph.clouds[keep.id], graph.up_axis)
    if obb is not None:
        keep.obb = obb
        keep.trans = np.asarray(obb.center, float).reshape(3)
    _remove(graph, drop.id)


def _split(graph: Any, node_id: str, parts: List[np.ndarray]) -> None:
    """把一个节点按点云的几坨拆成几个，观测数按点数比例分。"""
    src = graph.nodes[node_id]
    total = float(sum(len(p) for p in parts)) or 1.0
    src_quality = dict(graph.quality.get(node_id, {})) if hasattr(graph, "quality") else {}
    _remove(graph, node_id)
    for part in parts:
        obb = obb_from_points(part, graph.up_axis)
        if obb is None:
            continue
        new = ObjectNode(
            id=graph._new_id(),
            label=src.label,
            obb=obb,
            embeddings={k: np.array(v) for k, v in src.embeddings.items()},
            confidence=float(src.confidence),
            last_verified_step=int(src.last_verified_step),
            provider=src.provider,
        )
        new.num_observations = max(1, int(round(src.num_observations * len(part) / total)))
        new.observed_keyframes = list(src.observed_keyframes)
        new.trans = np.asarray(obb.center, float).reshape(3)
        graph.add_node(new)
        graph.clouds[new.id] = part
        if hasattr(graph, "quality"):
            q = dict(src_quality)
            q["split_from"] = node_id
            graph.quality[new.id] = q
