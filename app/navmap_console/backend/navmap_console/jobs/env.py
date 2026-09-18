"""Process environment helpers shared with scripts/run_map_merging.sh semantics."""
import glob
import re
import shutil
from typing import Dict, List, Optional

from ..config import Settings


def detect_non_boost_cpus(sysfs_glob: str = "/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq") -> Optional[str]:
    """CPUs whose max frequency is below the machine maximum, as a taskset list.

    Mirrors detect_non_boost_cpus in scripts/run_map_merging.sh: the highest-clocked
    cores on this i9-14900K are the ones that corrupt memory during long runs.
    Returns None when sysfs has no frequency info (then no taskset is applied).
    """
    freqs: Dict[int, int] = {}
    for path in glob.glob(sysfs_glob):
        m = re.search(r"/cpu(\d+)/", path)
        if not m:
            continue
        try:
            with open(path) as f:
                freqs[int(m.group(1))] = int(f.read().strip())
        except (OSError, ValueError):
            continue
    if not freqs:
        return None
    top = max(freqs.values())
    cpus: List[int] = sorted(c for c, f in freqs.items() if f < top)
    return ",".join(str(c) for c in cpus) if cpus else None


PYTHONPATH_PARTS = ("python", "third_party/litevloc_code/python", "third_party/pose_estimation_models")


def build_env(settings: Settings) -> Dict[str, str]:
    """Variables overriding os.environ for pipeline subprocesses (scripts/run_map_merging.sh:29-67).

    LD_PRELOAD is assigned, not inherited: a desktop-level LD_PRELOAD with the system GL stack
    was one of the two suspects for random segfaults in long runs.
    """
    conda_prefix = settings.python.resolve().parents[1]
    env = {
        "LD_PRELOAD": str(conda_prefix / "lib" / "libstdc++.so.6"),
        "MKL_THREADING_LAYER": "GNU",
        "PYTHONPATH": ":".join(str(settings.repo_root / p) for p in PYTHONPATH_PARTS),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if settings.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
    return env


def resolve_cpu_list(settings: Settings) -> Optional[str]:
    """MERGE_CPU_LIST wins ('' disables pinning); otherwise detect the non-boost cores."""
    if settings.cpu_list is not None:
        return settings.cpu_list or None
    return detect_non_boost_cpus()


def command_prefix(cpu_list: Optional[str]) -> List[str]:
    if cpu_list and shutil.which("taskset"):
        return ["taskset", "-c", cpu_list]
    return []
