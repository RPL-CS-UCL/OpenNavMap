"""Settings come from environment variables; every field has a default."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = _PKG_DIR.parents[3]  # app/navmap_console/backend/navmap_console -> repo


@dataclass(frozen=True)
class Settings:
    data_root: Path
    host: str
    port: int
    repo_root: Path
    python: Path
    eval_python: Path
    allowed_roots: List[Path]
    cpu_list: Optional[str]
    cuda_visible_devices: Optional[str]

    @property
    def frontend_dist(self) -> Path:
        return self.repo_root / "app" / "navmap_console" / "frontend" / "dist"

    @property
    def regions_dir(self) -> Path:
        return self.data_root / "regions"

    @property
    def jobs_dir(self) -> Path:
        return self.data_root / "jobs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_root / "uploads"


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    e = os.environ if env is None else env
    repo_root = Path(e.get("NAVMAP_CONSOLE_REPO", str(DEFAULT_REPO_ROOT))).resolve()
    roots = [Path(p).resolve() for p in e.get("NAVMAP_CONSOLE_ALLOWED_ROOTS", "/Titan/dataset").split(":") if p]
    # MERGE_CPU_LIST="" means "no taskset" (same as scripts/run_map_merging.sh), so keep the empty string.
    cpu_list = e.get("MERGE_CPU_LIST")
    return Settings(
        data_root=Path(e.get("NAVMAP_CONSOLE_DATA_ROOT", "/Titan/dataset/data_opennavmap/navmap_console")).resolve(),
        host=e.get("NAVMAP_CONSOLE_HOST", "0.0.0.0"),
        port=int(e.get("NAVMAP_CONSOLE_PORT", "8765")),
        repo_root=repo_root,
        python=Path(e.get("NAVMAP_CONSOLE_PYTHON", "/root/miniconda3/envs/opennavmap/bin/python")),
        eval_python=Path(e.get("NAVMAP_CONSOLE_EVAL_PYTHON", "/root/miniconda3/envs/traj_evaluation/bin/python")),
        allowed_roots=roots,
        cpu_list=cpu_list,
        cuda_visible_devices=e.get("CUDA_VISIBLE_DEVICES"),
    )
