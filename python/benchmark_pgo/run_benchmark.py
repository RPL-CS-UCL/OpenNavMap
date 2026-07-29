"""Robust-PGO benchmark: classic g2o datasets with injected outlier closures.

Example:
    PYTHONPATH=python:third_party/litevloc_code/python \\
      python python/benchmark_pgo/run_benchmark.py \\
        --dataset sphere2500 --output /tmp/pgo_benchmark
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import gtsam
import numpy as np

from benchmark_pgo import dataset, metrics, outliers
from utils.gtsam_pose_graph import PoseGraph

PRIOR_SIGMA = np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3) / 1e2
METHODS = ("none", "huber", "gnc_gm", "gnc_tls")


def anchor_components(
    graph: gtsam.NonlinearFactorGraph, initial: gtsam.Values
) -> List[int]:
    """Add one prior per connected component; return the prior factor indices."""
    model = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_SIGMA)
    indices = []
    for component in PoseGraph.find_connected_components(graph):
        graph.add(gtsam.PriorFactorPose3(component[0],
                                         initial.atPose3(component[0]), model))
        indices.append(graph.size() - 1)
    return indices


def build_case(
    base_graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    num_outliers: int,
    seed: int,
) -> Tuple[gtsam.NonlinearFactorGraph, List[int], List[int]]:
    """Return (graph with priors, injected factor indices, known inlier indices)."""
    graph, injected = outliers.inject(base_graph, initial, num_outliers, seed=seed)
    anchor_components(graph, initial)
    injected_set = set(injected)
    known_inliers = [i for i in range(graph.size()) if i not in injected_set]
    return graph, injected, known_inliers


def between_factor_indices(graph: gtsam.NonlinearFactorGraph) -> List[int]:
    """Indices of every BetweenFactorPose3, i.e. the outlier-classifiable set."""
    return [i for i in range(graph.size())
            if isinstance(graph.at(i), gtsam.BetweenFactorPose3)]


def run_one(
    method: str,
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    known_inlier_indices: List[int],
    barc_prob: float,
) -> Tuple[gtsam.Values, np.ndarray]:
    """Optimize with one back-end. Returns (result, per-factor weights)."""
    if method in ("none", "huber"):
        result = PoseGraph.optimize_pose_graph_with_LM(
            graph, initial, verbose=False, robust_kernel=(method == "huber"))
        weights = np.ones(graph.size())
    elif method in ("gnc_tls", "gnc_gm"):
        result, weights = PoseGraph.optimize_pose_graph_with_GNC(
            graph, initial,
            known_inlier_indices=known_inlier_indices,
            loss="TLS" if method == "gnc_tls" else "GM",
            barc_prob=barc_prob)
    else:
        raise ValueError(f"Unknown method '{method}', expected one of {METHODS}")
    return result, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="sphere2500",
                        choices=sorted(dataset.DATASET_URLS))
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.5],
                        help="Outlier count as a fraction of the genuine loop closures")
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHODS))
    parser.add_argument("--barc-probs", type=float, nargs="+", default=[0.99],
                        help="GNC inlier-cost probabilities to sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_graph, initial = dataset.load(dataset.fetch(args.dataset))
    num_loops = outliers.count_loop_closures(base_graph)
    print(f"{args.dataset}: {len(base_graph.keyVector())} poses, "
          f"{base_graph.size()} factors, {num_loops} loop closures")

    # Reference trajectory: vanilla LM on the clean graph.
    clean_graph, _, _ = build_case(base_graph, initial, 0, args.seed)
    reference, _ = run_one("none", clean_graph, initial, [], 0.99)
    print("reference built (vanilla LM, 0 outliers)")

    rows: List[Dict[str, object]] = []
    for ratio in args.ratios:
        num_outliers = int(round(ratio * num_loops))
        graph, injected, known_inliers = build_case(
            base_graph, initial, num_outliers, args.seed)
        candidates = between_factor_indices(graph)
        for method in args.methods:
            probs = args.barc_probs if method.startswith("gnc") else [args.barc_probs[0]]
            for barc_prob in probs:
                result, weights = run_one(
                    method, graph, initial, known_inliers, barc_prob)
                traj = metrics.ate(reference, result)
                cls = metrics.outlier_classification(weights, injected, candidates)
                row = {
                    "dataset": args.dataset, "ratio": ratio,
                    "num_outliers": num_outliers, "method": method,
                    "barc_prob": barc_prob if method.startswith("gnc") else "",
                    "trans_rmse": round(traj["trans_rmse"], 4),
                    "rot_rmse": round(traj["rot_rmse"], 4),
                    "precision": round(cls["precision"], 4),
                    "recall": round(cls["recall"], 4),
                }
                rows.append(row)
                print(f"  ratio={ratio:<5} {method:<8} barc={str(row['barc_prob']):<6} "
                      f"trans={row['trans_rmse']:<10} rot={row['rot_rmse']:<10} "
                      f"P={row['precision']:<7} R={row['recall']}")

    csv_path = out_dir / f"{args.dataset}_results.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
