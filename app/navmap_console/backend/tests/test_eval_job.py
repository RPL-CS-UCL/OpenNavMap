"""jobs/eval_job.py: TUM export + fake report parsing, exercised as a subprocess (the way the runner calls it)."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]  # app/navmap_console/backend/tests -> repo root
EVAL_JOB = REPO_ROOT / "app" / "navmap_console" / "backend" / "navmap_console" / "jobs" / "eval_job.py"


def _step(tmp_path: Path, with_gt: bool = True) -> Path:
    step = tmp_path / "merge_0"
    step.mkdir()
    lines = [f"seq/{i:06d}.color.jpg 1 0 0 0 {i}.0 0 0" for i in range(3)]
    (step / "poses.txt").write_text("\n".join(lines) + "\n")
    if with_gt:
        (step / "poses_abs_gt.txt").write_text("\n".join(lines) + "\n")
        (step / "timestamps.txt").write_text(
            "\n".join(f"seq/{i:06d}.color.jpg {1000 + i}.0" for i in range(3)) + "\n")
    return step


def _run(step: Path, out: Path, fake: bool = True) -> subprocess.CompletedProcess:
    args = [sys.executable, str(EVAL_JOB), "--step_dir", str(step), "--eval_dir", str(out),
            "--dataset", "step_00", "--alg", "opennavmap"]
    if fake:
        args.append("--fake-report")
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME")}
    return subprocess.run(args, capture_output=True, text=True, env=env)


def test_fake_report_writes_eval_json(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    proc = _run(_step(tmp_path), out)
    assert proc.returncode == 0, proc.stderr
    data = json.loads((out / "eval.json").read_text())
    assert data["status"] == "succeeded"
    assert data["ate_trans"] == 0.612
    assert data["ate_rot"] == 1.23
    assert data["frames"] == 3
    assert Path(data["report_dir"]).is_dir()
    # the fake report uses the real nested layout so parse_report exercises the production path
    assert list((out / "report").glob("report_benchmark_*/laptop_translation_rmse_*.txt"))
    assert (out / "eval_config.yaml").is_file()


def test_fake_report_writes_gt_tum(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    proc = _run(_step(tmp_path), out)
    assert proc.returncode == 0, proc.stderr
    gt = out / "traj_eval_data" / "groundtruth" / "traj" / "step_00.txt"
    est = out / "traj_eval_data" / "algorithms" / "opennavmap" / "laptop" / "traj" / "step_00.txt"
    assert gt.is_file() and est.is_file()  # GT write is commented out upstream; the job must fill it in
    assert gt.read_text().startswith("1000.000000")
    assert len(est.read_text().splitlines()) == 3


def test_missing_gt_fails_with_status(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    proc = _run(_step(tmp_path, with_gt=False), out)
    assert proc.returncode == 1
    data = json.loads((out / "eval.json").read_text())
    assert data["status"] == "failed"
    assert "poses_abs_gt" in data["error"]


def test_parse_report_reads_real_table_format(tmp_path: Path) -> None:
    """Row label carries LaTeX-escaped underscores; '-' marks a missing value."""
    sys.path.insert(0, str(EVAL_JOB.parent))
    import eval_job

    report = tmp_path / "report" / "report_benchmark_eval_config"
    report.mkdir(parents=True)
    (report / "laptop_translation_rmse_opennavmap202609211200.txt").write_text(
        "               &  opennavmap\nstep\\_07          &  1.940 \n")
    (report / "laptop_rotation_rmse_opennavmap202609211200.txt").write_text(
        "               &  opennavmap\nstep\\_07          &  - \n")
    t, r, f = eval_job.parse_report(tmp_path / "report", "step_07", "opennavmap")
    assert t == 1.94 and r is None and f is None
