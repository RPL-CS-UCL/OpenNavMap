"""物体关联与融合策略：拿逐帧的物体观测，融合成"这个房子里有哪些东西"。

放在这里而不是各个调用方仓库里，是因为它写的数据结构（ObjectNode / OBB /
ObjectGraph / 存盘格式 schema v2.1）全在本仓库。算法跟数据结构住在一起，
schema 一升级只要改一处。

两种策略：

  boxfusion    原版做法，就是 ObjectGraph.integrate_observation 本身：
               3D 框交并比 ≥ 0.3 **且** 视觉特征余弦 ≥ 0.7 才算同一个物体，
               合并后的框是历次观测框的加权平均。留着当对照基线。
  pointfusion  以点云为准：物体的"真身"是一路累积的点云，框每次从点云重算。
               关联时先过三道硬门（类别兼容 / 中心距够近 / 高度区间相交），
               再用四项加权打分（框重叠 + 点云重合 + 视觉相似 + 标签相同）。
               对地毯、画、电视这类薄物体和"只看到一半"的情况明显更准。

跑完所有帧之后还要调一次 postprocess()，做去噪、剔误检、补合并、拆错合、
去地板墙面这一整套全局整理 —— 这是离线建图相比在线建图最大的好处。

典型用法：

    from object_fusion import build_fusion_graph, load_config, postprocess

    cfg = load_config(my_overrides)          # 合并到本包的 default_config.yaml 上
    clouds = {}                              # 物体 id -> (N,3) 点云，跟着图一起长
    graph = build_fusion_graph("pointfusion", map_root, cfg, clouds=clouds)
    for step, obs in observations:
        graph.integrate_observation(obs, step)
    postprocess(graph, cfg)
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # opennavmap/python
_LITEVLOC = os.path.join(os.path.dirname(_PY_DIR), "third_party", "litevloc_code", "python")
for _p in (_PY_DIR, _LITEVLOC):
    if _p not in sys.path:
        sys.path.append(_p)

from object_graph import ObjectGraph  # noqa: E402

from object_fusion.pointfusion import PointFusionGraph  # noqa: E402
from object_fusion.postprocess import postprocess  # noqa: E402

FUSIONS = ("pointfusion", "boxfusion")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.yaml"

__all__ = [
    "FUSIONS", "DEFAULT_CONFIG_PATH",
    "build_fusion_graph", "load_config", "merge_config",
    "ObjectGraph", "PointFusionGraph", "postprocess",
]


def load_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读本包的 default_config.yaml，把调用方的配置盖上去，返回合并后的字典。"""
    import yaml

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return merge_config(cfg, overrides or {})


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """逐段递归合并：override 里没写的键保留 base 的值，不会整段被顶掉。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def build_fusion_graph(name: str, map_root: Any, cfg: Dict[str, Any],
                       up_axis: int = 2,
                       clouds: Optional[Dict[str, Any]] = None,
                       vocab: Any = None) -> ObjectGraph:
    """按名字造一张物体图。

    Args:
        name: "pointfusion" 或 "boxfusion"。
        map_root: 地图目录，图存盘时往这里写。
        cfg: 完整配置（至少含 assoc / fuse 两段），一般来自 load_config()。
        up_axis: 竖直方向是第几个坐标轴，z-up 的世界系填 2。
        clouds: 物体 id -> (N,3) 点云的字典，由调用方持有；pointfusion 会往里写。
            boxfusion 用不到，传了也不影响。
        vocab: 可选的类别词表对象，需要有 compatible(a, b) 方法判断两个类别名
            算不算同一类（比如 "couch" 和 "sofa"）。不传就按字符串小写相等判断。

    Returns:
        一张 ObjectGraph（或它的子类），可以直接 integrate_observation。

    Raises:
        ValueError: name 不在 FUSIONS 里。
    """
    key = str(name).strip().lower()
    if key == "pointfusion":
        return PointFusionGraph(Path(map_root), "object", up_axis,
                                cfg=cfg, clouds=clouds, vocab=vocab)
    if key == "boxfusion":
        return ObjectGraph(Path(map_root), "object", up_axis=up_axis)
    raise ValueError(f"不认识的融合策略 {name!r}，可选：{', '.join(FUSIONS)}")
