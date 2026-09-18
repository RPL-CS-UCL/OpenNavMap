"""GET /api/health: version, config, machine state."""
import shutil
import subprocess
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request

from .. import __version__
from ..jobs.env import detect_non_boost_cpus

router = APIRouter(prefix="/api", tags=["health"])


def query_gpu() -> Optional[Dict[str, Any]]:
    """First GPU's name and memory from nvidia-smi; None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    name, used, total = [x.strip() for x in out[0].split(",")]
    return {"name": name, "memory_used_mib": int(used), "memory_total_mib": int(total)}


@router.get("/health")
def health(request: Request) -> Dict[str, Any]:
    settings = request.app.state.settings
    usage = shutil.disk_usage(str(settings.data_root))
    cpu_list = settings.cpu_list if settings.cpu_list is not None else detect_non_boost_cpus()
    runner = getattr(request.app.state, "runner", None)
    return {
        "version": __version__,
        "data_root": str(settings.data_root),
        "repo_root": str(settings.repo_root),
        "queues": runner.queue_lengths() if runner is not None else {"gpu": 0, "cpu": 0},
        "gpu": query_gpu(),
        "disk": {"total": usage.total, "free": usage.free},
        "cpu_list": cpu_list or "",
    }
