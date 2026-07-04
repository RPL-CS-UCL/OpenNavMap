from __future__ import annotations

from pathlib import Path


def detect_merge_dirs(results_dir: Path) -> list[Path]:
    """Detect merge_* subdirectories in results_dir, sorted by merge order.

    Sorting key: number of underscore-separated parts (merge_0=1, merge_0_1=2, ...).
    Files (like merge_finalmap) are excluded.
    """
    candidates = [
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("merge_")
    ]
    return sorted(candidates, key=lambda d: d.name.count("_"))
