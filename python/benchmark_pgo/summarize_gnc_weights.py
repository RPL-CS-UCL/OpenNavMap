"""Summarize the per-step GNC weights written by map_merge_pipeline.

Cross-checks each rejection against the ground-truth localization error that
lloc_history recorded for the same edge, so over-rejection is visible.

Example:
    python python/benchmark_pgo/summarize_gnc_weights.py \\
      /Titan/dataset/.../s00000_results_in_spgo_cc_seqmatch_master_gnctls_iqaigtd
"""
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

WEIGHT_THRESHOLD = 0.5
#: A rejection is "justified" when the edge really was this far off the GT pose.
GT_TRANS_TOLERANCE = 0.45


def _parse(path: Path) -> List[Dict[str, float]]:
    """Read one gnc_weights.txt into a list of per-edge records."""
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            db, query, weight, conf, trans_err, rot_err = line.split(",")
            rows.append({
                "db_id": int(db), "query_id": int(query),
                "weight": float(weight), "conf": float(conf),
                "trans_err": float(trans_err), "rot_err": float(rot_err),
            })
    return rows


def _step_index(weights_path: Path) -> int:
    """merge_0_1_2/preds/gnc_weights.txt -> 2 (the last submap merged in)."""
    return int(weights_path.parent.parent.name.split("_")[-1])


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Rejection counts plus how many rejections the GT error backs up."""
    rejected = [r for r in rows if r["weight"] < WEIGHT_THRESHOLD]
    justified = [r for r in rejected
                 if not math.isnan(r["trans_err"])
                 and r["trans_err"] > GT_TRANS_TOLERANCE]
    kept_max = max((r["trans_err"] for r in rows
                    if r["weight"] >= WEIGHT_THRESHOLD), default=float("nan"))
    return {
        "total": len(rows),
        "rejected": len(rejected),
        "justified": len(justified),
        "max_rejected_trans_err": max((r["trans_err"] for r in rejected),
                                      default=float("nan")),
        "max_kept_trans_err": kept_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=str,
                        help="Merge result directory containing merge_*/preds/")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to write the per-step table")
    args = parser.parse_args()

    paths = sorted(Path(args.result_root).glob("merge_*/preds/gnc_weights.txt"),
                   key=_step_index)
    if not paths:
        raise SystemExit(f"no gnc_weights.txt under {args.result_root}")

    rows: List[Dict[str, object]] = []
    print(f"{'step':>4} {'total':>6} {'rej':>5} {'rej%':>6} "
          f"{'justified':>10} {'max_rej_err':>12} {'max_kept_err':>13}")
    for path in paths:
        stats = summarize(_parse(path))
        stats["step"] = _step_index(path)
        rows.append(stats)
        print(f"{stats['step']:>4} {stats['total']:>6} {stats['rejected']:>5} "
              f"{100 * stats['rejected'] / stats['total']:>5.0f}% "
              f"{stats['justified']:>10} {stats['max_rejected_trans_err']:>12.3f} "
              f"{stats['max_kept_trans_err']:>13.3f}")

    total = sum(r["total"] for r in rows)
    rejected = sum(r["rejected"] for r in rows)
    justified = sum(r["justified"] for r in rows)
    print(f"\n{len(rows)} steps: {rejected}/{total} edges rejected "
          f"({100 * rejected / total:.1f}%), {justified} of them with "
          f"GT translation error > {GT_TRANS_TOLERANCE} m "
          f"({100 * justified / rejected:.1f}% of rejections)"
          if rejected else f"\n{len(rows)} steps: no edge rejected")

    if args.csv:
        fields = ["step", "total", "rejected", "justified",
                  "max_rejected_trans_err", "max_kept_trans_err"]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: r[k] for k in fields} for r in rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
