"""Inject synthetic outlier loop closures into a 3D pose graph."""
from typing import List, Set, Tuple

import gtsam
import numpy as np


def _existing_pairs(graph: gtsam.NonlinearFactorGraph) -> Set[Tuple[int, int]]:
    """Collect the unordered key pairs already constrained by a between factor."""
    pairs = set()
    for i in range(graph.size()):
        factor = graph.at(i)
        if isinstance(factor, gtsam.BetweenFactorPose3):
            a, b = (int(k) for k in factor.keys())
            pairs.add((min(a, b), max(a, b)))
    return pairs


def count_loop_closures(graph: gtsam.NonlinearFactorGraph) -> int:
    """Count between factors that are not sequential odometry edges."""
    count = 0
    for i in range(graph.size()):
        factor = graph.at(i)
        if isinstance(factor, gtsam.BetweenFactorPose3):
            a, b = (int(k) for k in factor.keys())
            if abs(a - b) != 1:
                count += 1
    return count


def inject(
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    num_outliers: int,
    seed: int = 0,
    translation_scale: float = 10.0,
) -> Tuple[gtsam.NonlinearFactorGraph, List[int]]:
    """Append `num_outliers` random loop closures to a copy of `graph`.

    Injected edges connect non-adjacent poses that are not already constrained,
    and carry a uniformly random relative pose. Returns the new graph and the
    factor indices of the injected edges. `graph` itself is not modified.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(int(k) for k in initial.keys())
    taken = _existing_pairs(graph)

    out = gtsam.NonlinearFactorGraph()
    for i in range(graph.size()):
        out.add(graph.at(i))

    noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3))
    injected: List[int] = []
    while len(injected) < num_outliers:
        a, b = (int(x) for x in rng.choice(keys, size=2, replace=False))
        pair = (min(a, b), max(a, b))
        if abs(a - b) <= 1 or pair in taken:
            continue
        taken.add(pair)
        wrong_pose = gtsam.Pose3(
            gtsam.Rot3.Expmap(rng.uniform(-np.pi, np.pi, size=3)),
            gtsam.Point3(*rng.uniform(-translation_scale, translation_scale, size=3)),
        )
        out.add(gtsam.BetweenFactorPose3(a, b, wrong_pose, noise))
        injected.append(out.size() - 1)
    return out, injected
