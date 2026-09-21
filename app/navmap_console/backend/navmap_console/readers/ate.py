"""Per-step / final evaluation results on disk: evaluations/per_step/step_XX/eval.json."""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PER_STEP_EVAL_DIR = "per_step"
EVAL_JSON_NAME = "eval.json"


def step_eval_path(run_dir: Path, step_index: int) -> Path:
    return run_dir / "evaluations" / PER_STEP_EVAL_DIR / f"step_{step_index:02d}" / EVAL_JSON_NAME


def final_eval_dir(run_dir: Path) -> Path:
    return run_dir / "evaluations" / "final"


def read_step_ate(run_dir: Path, step_index: int) -> Optional[Dict[str, Any]]:
    """The eval.json for one step, or None when the step has not been evaluated (yet)."""
    path = step_eval_path(run_dir, step_index)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if "status" not in data:
        return None
    return data


def ate_fields(data: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[str]]:
    """(ate_trans [m], ate_rot [deg], frames, reason) from an eval.json dict; reason is None when OK."""
    if data is None:
        return None, None, None, "pending"
    if data.get("status") != "succeeded":
        return None, None, None, str(data.get("error") or "evaluation failed")
    return data.get("ate_trans"), data.get("ate_rot"), data.get("frames"), None


def step_has_gt(step_dir: Path) -> bool:
    """GT-based ATE is possible only when the step carries both the GT poses and timestamps."""
    return (step_dir / "poses_abs_gt.txt").is_file() and (step_dir / "timestamps.txt").is_file()
