"""Process environment helpers shared with scripts/run_map_merging.sh semantics."""
import glob
import re
from typing import Dict, List, Optional


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
