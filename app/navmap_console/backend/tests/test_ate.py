"""readers/ate.py: per-step eval.json reader."""
import json
from pathlib import Path

from navmap_console.readers.ate import ate_fields, final_eval_dir, read_step_ate, step_eval_path, step_has_gt


def _write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def test_step_eval_path_layout(tmp_path: Path) -> None:
    assert step_eval_path(tmp_path, 0) == tmp_path / "evaluations" / "per_step" / "step_00" / "eval.json"
    assert step_eval_path(tmp_path, 41) == tmp_path / "evaluations" / "per_step" / "step_41" / "eval.json"
    assert final_eval_dir(tmp_path) == tmp_path / "evaluations" / "final"


def test_read_step_ate_missing_is_none(tmp_path: Path) -> None:
    assert read_step_ate(tmp_path, 3) is None


def test_read_step_ate_roundtrip(tmp_path: Path) -> None:
    data = {"status": "succeeded", "ate_trans": 0.612, "ate_rot": 1.23, "frames": 890}
    _write(step_eval_path(tmp_path, 3), data)
    assert read_step_ate(tmp_path, 3) == data


def test_ate_fields_missing_gives_pending() -> None:
    t, r, f, reason = ate_fields(None)
    assert (t, r, f) == (None, None, None)
    assert reason == "pending"


def test_ate_fields_succeeded() -> None:
    t, r, f, reason = ate_fields({"status": "succeeded", "ate_trans": 0.612, "ate_rot": 1.23, "frames": 890})
    assert (t, r, f, reason) == (0.612, 1.23, 890, None)


def test_ate_fields_failed_carries_reason() -> None:
    t, r, f, reason = ate_fields({"status": "failed", "error": "no shared frames"})
    assert (t, r, f) == (None, None, None)
    assert reason == "no shared frames"


def test_step_has_gt(tmp_path: Path) -> None:
    step = tmp_path / "merge_0"
    step.mkdir()
    assert step_has_gt(step) is False
    (step / "poses_abs_gt.txt").write_text("")
    assert step_has_gt(step) is False
    (step / "timestamps.txt").write_text("")
    assert step_has_gt(step) is True
