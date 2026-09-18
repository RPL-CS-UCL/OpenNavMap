"""Task specifications: parameter schema and command lines for every job kind."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Settings

FAKE_SCRIPT = Path(__file__).resolve().parent / "fake_pipeline.py"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str  # int | float | str | bool | choice
    default: Any
    help: str
    group: str  # localization | optimization | culling | advanced
    choices: Optional[Tuple[str, ...]] = None
    advanced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.type, "default": self.default, "help": self.help,
                "group": self.group, "choices": list(self.choices) if self.choices else None,
                "advanced": self.advanced}

    def coerce(self, value: Any) -> Any:
        try:
            if self.type == "bool":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
                    return value.lower() in ("true", "1")
                raise ValueError
            if self.type == "int":
                if isinstance(value, bool):
                    raise ValueError
                return int(value)
            if self.type == "float":
                if isinstance(value, bool):
                    raise ValueError
                return float(value)
            if self.type == "choice":
                if str(value) not in (self.choices or ()):
                    raise ValueError
                return str(value)
            return str(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name}: invalid value {value!r} for type {self.type}")


# Defaults = the reference configuration validated end-to-end (scripts/run_map_merging.sh).
MERGE_PARAMS: Tuple[ParamSpec, ...] = (
    ParamSpec("pose_estimation_method", "str", "master",
              "Metric pose estimator run on geometrically verified frame pairs (master = MASt3R)", "localization"),
    ParamSpec("vpr_match_seq_len", "int", 10,
              "Sequence length used by the VPR sequence matcher on the distance matrix", "localization"),
    ParamSpec("vpr_match_model", "str", "vpr_dp", "VPR sequence matching model", "localization", advanced=True),
    ParamSpec("pgo_robust", "choice", "gnc_gm",
              "Robust back-end of the pose graph optimization; GM lets a later step re-admit an edge that "
              "an earlier step down-weighted", "optimization", choices=("none", "huber", "gnc_tls", "gnc_gm")),
    ParamSpec("pgo_loop_sigma_trans", "float", 0.1,
              "Translation std [m] of an inter-submap loop factor; with conf scaling it sets the GNC intake radius",
              "optimization"),
    ParamSpec("pgo_loop_sigma_rot", "float", 1.0, "Rotation std [deg] of an inter-submap loop factor", "optimization"),
    ParamSpec("pgo_loop_conf_scaling", "choice", "inverse",
              "How matcher confidence scales a loop factor's sigma: inverse divides by conf, none keeps it flat",
              "optimization", choices=("inverse", "none")),
    ParamSpec("use_iqa", "bool", True, "Use image quality assessment when culling redundant keyframes", "culling"),
    ParamSpec("use_ig", "bool", True, "Use information gain when culling redundant keyframes", "culling"),
    ParamSpec("use_td", "bool", True, "Use temporal difference when culling redundant keyframes", "culling"),
    ParamSpec("device", "choice", "cuda", "Compute device for VPR and pose estimation", "advanced",
              choices=("cuda", "cpu"), advanced=True),
    ParamSpec("viz", "bool", True, "Write the per-step diagnostic figures under preds/", "advanced", advanced=True),
    ParamSpec("rerun_viz", "bool", True, "Record runtime events (demo_events.jsonl) used by the live view",
              "advanced", advanced=True),
)
_BY_NAME = {p.name: p for p in MERGE_PARAMS}
_VALUE_FLAGS = ("pose_estimation_method", "vpr_match_model", "vpr_match_seq_len", "pgo_robust",
                "pgo_loop_sigma_trans", "pgo_loop_sigma_rot", "pgo_loop_conf_scaling", "device")
_SWITCH_FLAGS = ("use_iqa", "use_ig", "use_td", "viz")


def params_schema() -> List[Dict[str, Any]]:
    return [p.to_dict() for p in MERGE_PARAMS]


def validate_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce user-supplied values and fill defaults; unknown names or bad values raise ValueError."""
    unknown = sorted(set(params) - set(_BY_NAME))
    if unknown:
        raise ValueError(f"unknown parameters: {', '.join(unknown)}")
    return {p.name: (p.coerce(params[p.name]) if p.name in params else p.default) for p in MERGE_PARAMS}


def merge_argv(settings: Settings, *, submap_list: Path, result_dir: Path, image_size: Sequence[int],
               params: Mapping[str, Any], append_from: Optional[Path] = None, start_step: int = 0,
               rerun_viz_dir: Optional[Path] = None) -> List[str]:
    """Command line for a merge/append job (spec §5.5 console mode; indexed step directories).

    --method defaults to "console" inside the pipeline when --submap_list is given, and
    --rerun-output is only read by the offline renderer, so neither is passed here.
    """
    if settings.fake_pipeline:
        argv = [str(settings.python), str(FAKE_SCRIPT), "--submap_list", str(submap_list), "--result_dir",
                str(result_dir), "--step_dir_style", "indexed", "--start_step", str(start_step)]
        if append_from is not None:
            argv += ["--append_from", str(append_from)]
        return argv
    p = validate_params(params)
    argv = [str(settings.python), str(settings.repo_root / "python" / "map_merge_pipeline.py"),
            "--submap_list", str(submap_list), "--result_dir", str(result_dir), "--step_dir_style", "indexed",
            "--image_size", *[str(int(v)) for v in image_size]]
    for name in _VALUE_FLAGS:
        argv += [f"--{name}", str(p[name])]
    for name in _SWITCH_FLAGS:
        if p[name]:
            argv.append(f"--{name}")
    if p["rerun_viz"] and rerun_viz_dir is not None:
        argv += ["--rerun-viz", "--rerun-viz-dir", str(rerun_viz_dir)]
    if append_from is not None:
        argv += ["--append_from", str(append_from), "--start_step", str(start_step)]
    return argv
