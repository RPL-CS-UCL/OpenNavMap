"""jobs/export_job.py: the three bundles (map/report/preds) and the verify pass."""
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPORT_JOB = REPO_ROOT / "app" / "navmap_console" / "backend" / "navmap_console" / "jobs" / "export_job.py"


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(EXPORT_JOB), *argv], capture_output=True, text=True,
                          env={"PYTHONPATH": str(REPO_ROOT / "python")})


def _final_map(final: Path) -> None:
    (final / "seq").mkdir(parents=True)
    (final / "seq" / "000000.color.jpg").write_bytes(b"\xff\xd8fakejpg")
    for name, body in (("poses.txt", "seq/000000.color.jpg 1 0 0 0 0 0 0\n"),
                       ("intrinsics.txt", "seq/000000.color.jpg 1 1 1 1 1 1\n"),
                       ("database_descriptors.txt", ""), ("edges_covis.txt", ""), ("edges_trav.txt", "")):
        (final / name).write_text(body)


def _json_of(run_dir: Path, name: str) -> dict:
    return json.loads((run_dir / "exports" / f"{name}.json").read_text())


def test_map_bundle_from_final(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _final_map(run_dir / "final")
    rc = _run("--run_dir", str(run_dir), "--kind", "map", "--name", "m1",
              "--out_json", str(run_dir / "exports" / "m1.json"))
    assert rc.returncode == 0, rc.stderr
    meta = _json_of(run_dir, "m1")
    assert meta["status"] == "succeeded"
    tar = run_dir / "exports" / "m1.tar.gz"
    assert tar.is_file()
    with tarfile.open(tar) as tf:
        names = tf.getnames()
    assert "poses.txt" in names and "seq/000000.color.jpg" in names
    assert "preds" not in names  # preds are not part of the navigation map


def test_map_bundle_without_final_consolidates_last_step(tmp_path: Path) -> None:
    """Imported runs have no final/: the last step is consolidated with images from --sources."""
    run_dir = tmp_path / "run"
    step = run_dir / "output" / "merge_0_1"
    (step / "preds").mkdir(parents=True)
    (step / "preds" / "gnc_weights.txt").write_text("x\n")
    for name, body in (("poses.txt", "seq/000000.color.jpg 1 0 0 0 0 0 0\n"),
                       ("intrinsics.txt", "seq/000000.color.jpg 1 1 1 1 1 1\n"),
                       ("database_descriptors.txt", ""), ("edges_covis.txt", ""), ("edges_trav.txt", "")):
        (step / name).write_text(body)
    source = tmp_path / "images" / "seq"
    source.mkdir(parents=True)
    (source / "000000.color.jpg").write_bytes(b"\xff\xd8fakejpg")
    rc = _run("--run_dir", str(run_dir), "--kind", "map", "--name", "m2", "--sources", str(source.parent),
              "--out_json", str(run_dir / "exports" / "m2.json"))
    assert rc.returncode == 0, rc.stderr
    with tarfile.open(run_dir / "exports" / "m2.tar.gz") as tf:
        names = tf.getnames()
    assert "poses.txt" in names and "seq/000000.color.jpg" in names
    assert not any("gnc_weights" in n for n in names)
    assert not (run_dir / "exports" / ".stage_m2").exists()


def test_report_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    report = run_dir / "evaluations" / "final" / "report"
    report.mkdir(parents=True)
    (report / "plot.pdf").write_bytes(b"%PDF-fake")
    rc = _run("--run_dir", str(run_dir), "--kind", "report", "--name", "r1",
              "--eval_dir", str(report), "--out_json", str(run_dir / "exports" / "r1.json"))
    assert rc.returncode == 0, rc.stderr
    with tarfile.open(run_dir / "exports" / "r1.tar.gz") as tf:
        assert "plot.pdf" in tf.getnames()


def test_preds_bundle_steps_filter(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for name, i in (("merge_0", 0), ("merge_0_1", 1), ("merge_0_1_2", 2)):
        preds = run_dir / "output" / name / "preds"
        preds.mkdir(parents=True)
        (preds / "gnc_weights.txt").write_text(f"step {i}\n")
    rc = _run("--run_dir", str(run_dir), "--kind", "preds", "--name", "p1", "--steps", "0,2",
              "--out_json", str(run_dir / "exports" / "p1.json"))
    assert rc.returncode == 0, rc.stderr
    with tarfile.open(run_dir / "exports" / "p1.tar.gz") as tf:
        names = tf.getnames()
    assert "merge_0/preds/gnc_weights.txt" in names
    assert "merge_0_1_2/preds/gnc_weights.txt" in names
    assert not any("merge_0_1/" in n for n in names)
    assert _json_of(run_dir, "p1")["steps"] == [0, 2]


def test_verify_pass_marks_verified(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _final_map(run_dir / "final")
    rc = _run("--run_dir", str(run_dir), "--kind", "map", "--name", "m1",
              "--out_json", str(run_dir / "exports" / "m1.json"), "--verify")
    assert rc.returncode == 0, rc.stderr
    assert _json_of(run_dir, "m1")["verified"] is True
    assert not (run_dir / "exports" / ".verify_m1").exists()


def test_failure_writes_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rc = _run("--run_dir", str(run_dir), "--kind", "report", "--name", "r0",
              "--eval_dir", str(run_dir / "missing"), "--out_json", str(run_dir / "exports" / "r0.json"))
    assert rc.returncode == 1
    meta = _json_of(run_dir, "r0")
    assert meta["status"] == "failed" and "missing" in meta["error"]


def test_placeholder_metadata_is_kept(tmp_path: Path) -> None:
    """create() writes created_at/job_id before the job runs; the script must keep them, not restamp."""
    run_dir = tmp_path / "run"
    report = run_dir / "evaluations" / "final" / "report"
    report.mkdir(parents=True)
    (report / "plot.pdf").write_bytes(b"%PDF-fake")
    out_json = run_dir / "exports" / "r1.json"
    out_json.parent.mkdir(parents=True)
    out_json.write_text(json.dumps({"name": "r1", "kind": "report", "status": "queued", "job_id": "job_x",
                                    "created_at": "2026-09-20T01:02:03+00:00"}))
    rc = _run("--run_dir", str(run_dir), "--kind", "report", "--name", "r1",
              "--eval_dir", str(report), "--out_json", str(out_json))
    assert rc.returncode == 0, rc.stderr
    meta = _json_of(run_dir, "r1")
    assert meta["status"] == "succeeded" and meta["entries"] == 1
    assert meta["job_id"] == "job_x" and meta["created_at"] == "2026-09-20T01:02:03+00:00"
