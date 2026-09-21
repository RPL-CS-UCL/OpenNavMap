"""ATE evaluation job: strictly the traj_evaluation toolchain, nothing computed here.

CLI: python eval_job.py --step_dir DIR --eval_dir DIR --dataset NAME --alg NAME [--fake-report]

1. Exports DIR/{poses.txt,poses_abs_gt.txt,timestamps.txt} to TUM under
   <eval_dir>/traj_eval_data/ (groundtruth + algorithms/<alg>/laptop/traj). The GT write
   is commented out upstream in export_tum_files (NOTE(gogojjh)), so it is done here.
2. Writes a one-dataset/one-algorithm yaml (<eval_dir>/eval_config.yaml) and runs
   benchmark_map_merge/scripts/run_evaluation.sh with TRAJ_PATH=<eval_dir>/traj_eval_data,
   --config <abs yaml> and --output-dir <eval_dir>/report. The analysis script nests its
   files under report/report_benchmark_<yaml stem>/. --fake-report skips the toolchain and
   writes a canned report in the same layout (tests / fake pipeline never shell out).
3. Parses the rmse LaTeX tables (+ *_meas_stats.json for the frame count) into
   <eval_dir>/eval.json: {"status", "ate_trans", "ate_rot", "frames", "created_at", "report_dir", "error"?}.

Runs under the evaluation interpreter (NAVMAP_CONSOLE_EVAL_PYTHON): only numpy/scipy needed.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[5]  # app/navmap_console/backend/navmap_console/jobs/eval_job.py -> repo root
PY_DIR = REPO_ROOT / "python"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

FAKE_ATE_TRANS = 0.612
FAKE_ATE_ROT = 1.23
CONFIG_NAME = "eval_config.yaml"
TUM_FMT = "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def export_tum(step_dir: Path, traj_root: Path, dataset: str, alg: str) -> Tuple[Path, Path]:
    """TUM groundtruth + estimate under the traj_evaluation layout; returns (gt_path, est_path)."""
    import numpy as np
    from benchmark_map_merge.export_eval_data import _convert_mapfree_to_tum_data
    from benchmark_map_merge.merge_writer import read_poses, read_timestamps

    poses_est = read_poses(str(step_dir / "poses.txt"))
    poses_gt = read_poses(str(step_dir / "poses_abs_gt.txt"))
    timestamps = read_timestamps(str(step_dir / "timestamps.txt"))
    gt_path = traj_root / "groundtruth" / "traj" / f"{dataset}.txt"
    est_path = traj_root / "algorithms" / alg / "laptop" / "traj" / f"{dataset}.txt"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    est_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(gt_path, _convert_mapfree_to_tum_data(poses_gt, timestamps), fmt=TUM_FMT)
    np.savetxt(est_path, _convert_mapfree_to_tum_data(poses_est, timestamps), fmt=TUM_FMT)
    return gt_path, est_path


def write_config(eval_dir: Path, dataset: str, alg: str) -> Path:
    """Minimal analyze_trajectories yaml; labels equal the raw names so the report rows/columns parse back."""
    path = eval_dir / CONFIG_NAME
    path.write_text(
        "Datasets:\n"
        f"  {dataset}:\n    platform: mobile_robot\n    label: {dataset}\n    date: 00000000\n    title: {dataset}\n"
        "Algorithms:\n"
        f"  {alg}:\n    fn: {alg}\n    label: {alg}\n"
        "RelDistances: []\nRelDistancePercentages: []\n")
    return path


def write_fake_report(report_dir: Path, dataset: str, alg: str, frames: int) -> None:
    """Same nested layout and table format as a real report, with canned numbers."""
    nested = report_dir / f"report_benchmark_{Path(CONFIG_NAME).stem}"
    nested.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    header = f"               &  {alg}\n"
    (nested / f"laptop_translation_rmse_{alg}{stamp}.txt").write_text(header + f"{dataset}&  {FAKE_ATE_TRANS:.3f} \n")
    (nested / f"laptop_rotation_rmse_{alg}{stamp}.txt").write_text(header + f"{dataset}&  {FAKE_ATE_ROT:.3f} \n")
    stats_dir = nested / f"laptop_mobile_robot_{dataset}"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / f"{dataset}_meas_stats.json").write_text(
        json.dumps({alg: {"start": [1000.0], "end": [1000.0 + frames], "n_meas": [frames]}}))


def _table_value(path: Path, dataset: str) -> Optional[float]:
    """Cell of the row labelled `dataset` in a `&`-separated rmse table ('-' or absent -> None)."""
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) > 1 and cells[0].replace("\\_", "_") == dataset:
            return None if cells[1] in ("", "-") else float(cells[1])
    return None


def parse_report(report_dir: Path, dataset: str, alg: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """(ate_trans [m], ate_rot [deg], frames) from the newest rmse tables under report_dir; None when absent."""
    trans_files = sorted(report_dir.rglob(f"laptop_translation_rmse_{alg}*.txt"))
    rot_files = sorted(report_dir.rglob(f"laptop_rotation_rmse_{alg}*.txt"))
    ate_trans = _table_value(trans_files[-1], dataset) if trans_files else None
    ate_rot = _table_value(rot_files[-1], dataset) if rot_files else None
    frames = None
    for stats in sorted(report_dir.rglob(f"{dataset}_meas_stats.json")):
        meas = json.loads(stats.read_text())
        if meas.get(alg, {}).get("n_meas"):
            frames = int(meas[alg]["n_meas"][0])
    return ate_trans, ate_rot, frames


def run_toolchain(traj_root: Path, report_dir: Path, config: Path) -> int:
    script = REPO_ROOT / "python" / "benchmark_map_merge" / "scripts" / "run_evaluation.sh"
    env = dict(os.environ, TRAJ_PATH=str(traj_root), PYTHON=sys.executable)
    proc = subprocess.run(["bash", str(script), "--output-dir", str(report_dir), "--config", str(config)],
                          env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step_dir", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alg", default="opennavmap")
    ap.add_argument("--fake-report", action="store_true")
    args = ap.parse_args(argv)

    step_dir = Path(args.step_dir)
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    traj_root = eval_dir / "traj_eval_data"
    report_dir = eval_dir / "report"
    data: Dict[str, Any]
    try:
        if not (step_dir / "poses_abs_gt.txt").is_file():
            raise RuntimeError("poses_abs_gt.txt missing: this run/step has no GT")
        if not (step_dir / "timestamps.txt").is_file():
            raise RuntimeError("timestamps.txt missing: cannot associate poses with GT")
        gt_path, _ = export_tum(step_dir, traj_root, args.dataset, args.alg)
        frames = sum(1 for line in gt_path.read_text().splitlines() if line.strip())
        config = write_config(eval_dir, args.dataset, args.alg)
        if args.fake_report:
            print(f"fake report -> {report_dir}", flush=True)
            write_fake_report(report_dir, args.dataset, args.alg, frames)
        else:
            print(f"running traj_evaluation on {step_dir}", flush=True)
            rc = run_toolchain(traj_root, report_dir, config)
            if rc != 0:
                raise RuntimeError(f"run_evaluation.sh exited with {rc}")
        ate_trans, ate_rot, parsed_frames = parse_report(report_dir, args.dataset, args.alg)
        if ate_trans is None:
            raise RuntimeError(f"no translation rmse for {args.dataset}/{args.alg} under {report_dir}")
        data = {"status": "succeeded", "ate_trans": ate_trans, "ate_rot": ate_rot,
                "frames": parsed_frames if parsed_frames is not None else frames,
                "created_at": _now_iso(), "report_dir": str(report_dir)}
        print(f"eval.json: ate_trans={ate_trans} ate_rot={ate_rot} frames={data['frames']}", flush=True)
    except Exception as exc:  # noqa: BLE001 - a failed evaluation must still leave a status on disk
        data = {"status": "failed", "ate_trans": None, "ate_rot": None, "frames": None,
                "created_at": _now_iso(), "error": f"{type(exc).__name__}: {exc}", "report_dir": str(report_dir)}
        print(f"evaluation failed: {data['error']}", flush=True)
        (eval_dir / "eval.json").write_text(json.dumps(data))
        return 1
    (eval_dir / "eval.json").write_text(json.dumps(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
