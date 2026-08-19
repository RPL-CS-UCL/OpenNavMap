"""pointfusion：以点云为准的物体关联与融合策略。

一句话说清跟本仓库原做法（ObjectGraph.integrate_observation，下称 boxfusion）的区别：
**物体的"真身"是一路累积下来的点云，包围盒每次从点云重新算，而不是把历次的框平均。**

boxfusion 的毛病和这里对应的解法：

  毛病                                          这里怎么解
  ─────────────────────────────────────────  ───────────────────────────────
  薄扁的东西（地毯、画、电视）两个框几乎没有   除了框重叠，还看两片点云重不重合。
  体积交集，同一张地毯每帧都被当成新物体      地毯的框交集≈0，但点云是实打实重合的
  只看到沙发一半时框小，看全了反被平均拉回去   框从点云并集重算，看得越全框只会越准
  不看类别，椅子和旁边的沙发会被合成一个       先过类别门：不是同一类直接否掉
  框的三个边长顺序会乱，"高 0.51 米的地毯"     用 cv2.minAreaRect 算，第三个数永远是真高度
  没有共同特征时相似度返回 1.0（等于没门）      改成返回 0，并打一次提醒
  5 米外和 1 米内的观测一样重                  按距离/是否被画面裁掉/像素数打折
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from object_graph import ObjectGraph
from object_node import ObjectNode, ObjectObservation
from utils_object_geom import iou_3d, yaw_from_R

from object_fusion.geom import (
    intervals_overlap, merge_clouds, obb_diag, obb_from_points, overlap_ratio, z_range,
)


class PointFusionGraph(ObjectGraph):
    """在 ObjectGraph 上换掉"怎么判断是同一个物体"和"合并后框怎么定"。

    其余（存盘格式、可见性边、best_crop、读盘）全部继承，不改一行。
    """

    def __init__(self, map_root: Path, edge_type: str = "object", up_axis: int = 2,
                 cfg: Optional[Dict[str, Any]] = None,
                 clouds: Optional[Dict[str, np.ndarray]] = None,
                 vocab: Any = None) -> None:
        super().__init__(map_root, edge_type, up_axis)
        cfg = cfg or {}
        self.cfg = cfg
        self.clouds = clouds if clouds is not None else {}
        self.vocab = vocab
        self.quality: Dict[str, Dict[str, Any]] = {}
        a = cfg.get("assoc", {})
        self.score_thresh = float(a.get("score_thresh", 0.55))
        self.dist_max = float(a.get("dist_max", 1.0))
        self.dist_scale = float(a.get("dist_scale", 0.5))
        self.z_slack = float(a.get("z_slack", 0.05))
        self.overlap_tau = float(a.get("overlap_tau", 0.05))
        self.w_iou = float(a.get("w_iou", 0.35))
        self.w_overlap = float(a.get("w_overlap", 0.35))
        self.w_emb = float(a.get("w_emb", 0.20))
        self.w_label = float(a.get("w_label", 0.10))
        f = cfg.get("fuse", {})
        self.voxel = float(f.get("voxel", 0.01))
        self.max_points = int(f.get("max_points_per_object", 20000))
        self._warned_no_emb = False

    # -------------------------------------------------------------- 关联

    def _has_shared_embedding(self, obs: ObjectObservation, node: ObjectNode) -> bool:
        if set(obs.embeddings) & set(node.embeddings):
            return True
        if not self._warned_no_emb:
            print("[pointfusion] 观测和地图里的物体没有共同的特征名，"
                  "这一项不计分，只靠几何和类别关联", flush=True)
            self._warned_no_emb = True
        return False

    def _embedding_similarity(self, obs: ObjectObservation, node: ObjectNode) -> float:
        """没有共同特征时返回 0，不是父类那样返回 1.0。

        父类返回 1.0 意味着"视觉这一关白送"，一旦特征名字对不上，整个视觉判据就
        静默失效、退化成只看几何。这里宁可把这一项记 0，让问题露出来。
        """
        if not self._has_shared_embedding(obs, node):
            return 0.0
        return super()._embedding_similarity(obs, node)

    def _label_ok(self, label_a: str, label_b: str) -> bool:
        if self.vocab is not None:
            return bool(self.vocab.compatible(label_a, label_b))
        return str(label_a).strip().lower() == str(label_b).strip().lower()

    def _gates(self, obs: ObjectObservation, node: ObjectNode) -> bool:
        """三道硬门，任何一道不过就直接判定"不是同一个物体"。"""
        if not self._label_ok(obs.label, node.label):
            return False
        d = float(np.linalg.norm(np.asarray(obs.obb.center, float).reshape(3)
                                 - np.asarray(node.obb.center, float).reshape(3)))
        if d > max(self.dist_max, self.dist_scale * obb_diag(node.obb)):
            return False
        # 高度区间不相交 -> 一个在桌上一个是桌子，不能合
        if not intervals_overlap(z_range(obs.obb, self.up_axis),
                                 z_range(node.obb, self.up_axis), self.z_slack):
            return False
        return True

    def _score(self, obs: ObjectObservation, node: ObjectNode,
               pts: Optional[np.ndarray]) -> float:
        """四项加权：框重叠、点云重合、视觉相似、标签完全一致。总分满分 1。

        没有共同视觉特征时（比如关掉了特征提取），这一项不是记 0 就完事——
        那样等于凭空扣掉 0.2 分，同一个阈值在"有特征"和"没特征"两种情况下松紧
        差一大截。这里把它那份权重按比例摊回给剩下三项，让阈值的含义保持一致。
        """
        iou = float(iou_3d(obs.obb, node.obb, self.up_axis))
        ov = 0.0
        if pts is not None and len(pts) and node.id in self.clouds:
            ov = overlap_ratio(pts, self.clouds[node.id], self.overlap_tau)
        same = 1.0 if str(obs.label).lower() == str(node.label).lower() else 0.0
        geom = self.w_iou * iou + self.w_overlap * ov + self.w_label * same
        if not self._has_shared_embedding(obs, node):
            total = self.w_iou + self.w_overlap + self.w_label
            return geom / total if total > 0 else 0.0
        sim = max(0.0, self._embedding_similarity(obs, node))
        return geom + self.w_emb * sim

    def integrate_observation(self, obs: ObjectObservation, step: int,
                              **kwargs: Any) -> Tuple[ObjectNode, bool]:
        # 观测质量直接折进置信度：远的、被画面裁掉的、像素少的观测从此在
        # 加权平均里说话声音就小（父类所有加权都用 confidence 这一个量）
        weight = float(getattr(obs, "weight", 1.0))
        obs.confidence = float(obs.confidence) * weight
        pts = getattr(obs, "points", None)
        obs_yaw = yaw_from_R(obs.obb.R, self.up_axis)

        best_node, best_score = None, -1.0
        for node in self.nodes.values():
            if not self._gates(obs, node):
                continue
            score = self._score(obs, node, pts)
            if score > best_score:
                best_node, best_score = node, score

        if best_node is None or best_score < self.score_thresh:
            node = ObjectNode.from_observation(self._new_id(), obs, step)
            node._yaws = [obs_yaw]
            self.add_node(node)
            if pts is not None:
                self.clouds[node.id] = pts
                self._refresh_obb(node)
            self.quality[node.id] = {"obs_weights": [weight]}
            return node, True

        self._merge_into(best_node, obs, obs_yaw, step)
        self.quality.setdefault(best_node.id, {"obs_weights": []})["obs_weights"].append(weight)
        return best_node, False

    # -------------------------------------------------------------- 融合

    def _merge_into(self, node: ObjectNode, obs: ObjectObservation,
                    obs_yaw: float, step: int) -> None:
        # 统计量、特征融合、可见帧记录照父类的口径来（一行都不重写），
        # 它顺手算的那个"框加权平均"随后会被点云重算的框覆盖掉。
        super()._merge_into(node, obs, obs_yaw, step)
        pts = getattr(obs, "points", None)
        if pts is None or len(pts) == 0:
            return
        self.clouds[node.id] = merge_clouds(self.clouds.get(node.id), pts,
                                            self.voxel, self.max_points)
        self._refresh_obb(node)

    def _refresh_obb(self, node: ObjectNode) -> None:
        """从这个物体当前的全部点云重算包围盒。

        点云只增不减，所以框只会长大不会缩小——"只看到一半时把框平均小了"
        这个毛病在这里从构造上就不存在，不需要额外的"禁止缩小"判断。
        """
        cloud = self.clouds.get(node.id)
        if cloud is None or len(cloud) == 0:
            return
        obb = obb_from_points(cloud, self.up_axis)
        if obb is None:
            return
        node.obb = obb
        node.trans = np.asarray(obb.center, float).reshape(3)
