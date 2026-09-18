"""Check that a directory is a loadable OpenNavMap submap (spec §3.4)."""
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models import FileCheck, ValidationReport

# (file name, tokens per line including the key)
REQUIRED_FILES: List[Tuple[str, int]] = [
    ("timestamps.txt", 2),
    ("intrinsics.txt", 7),
    ("gps_data.txt", 6),  # PointGraph.save_to_file star-expands gps_data, so a missing file crashes the merge
    ("database_descriptors.txt", 0),  # 0 = any constant width >= 2
]
OPTIONAL_FILES: List[Tuple[str, int]] = [
    ("poses_abs_gt.txt", 8),
    ("iqa_data.txt", 2),
]
EDGE_FILES = ["edges_covis.txt", "edges_odom.txt", "edges_trav.txt"]
_FRAME_RE = re.compile(r"(\d+)\.color\.jpg$")


def frame_index(name: str) -> Optional[int]:
    m = _FRAME_RE.search(name)
    return int(m.group(1)) if m else None


def _read_keyed(path: Path) -> Tuple[Dict[str, List[str]], Optional[str]]:
    """key -> remaining tokens; second value is an error string for malformed input."""
    rows: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) < 2:
                return rows, f"line {lineno}: expected a key and values"
            rows[parts[0]] = parts[1:]
    return rows, None


def _check_keyed_file(map_dir: Path, name: str, width: int, required: bool,
                      keys: List[str]) -> Tuple[FileCheck, Optional[Dict[str, List[str]]]]:
    path = map_dir / name
    if not path.is_file():
        return FileCheck(name=name, required=required, status="missing"), None
    rows, err = _read_keyed(path)
    if err:
        return FileCheck(name=name, required=required, status="invalid", lines=len(rows), detail=err), None
    widths = {len(v) + 1 for v in rows.values()}
    if width and widths and widths != {width}:
        return FileCheck(name=name, required=required, status="invalid", lines=len(rows),
                         detail=f"expected {width} columns, found {sorted(widths)}"), None
    if not width and len(widths) > 1:
        return FileCheck(name=name, required=required, status="invalid", lines=len(rows),
                         detail=f"descriptor width varies: {sorted(widths)}"), None
    missing = [k for k in keys if k not in rows]
    if missing:
        return FileCheck(name=name, required=required, status="incomplete", lines=len(rows),
                         detail=f"{len(missing)} of {len(keys)} frames missing, e.g. {missing[0]}"), rows
    return FileCheck(name=name, required=required, status="ok", lines=len(rows)), rows


def _check_edge_file(map_dir: Path, name: str, num_frames: int) -> FileCheck:
    path = map_dir / name
    if not path.is_file():
        return FileCheck(name=name, required=True, status="missing")
    count = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                return FileCheck(name=name, required=True, status="invalid", lines=count,
                                 detail=f"line {lineno}: expected 'a b weight'")
            try:
                a, b = int(float(parts[0])), int(float(parts[1]))
                float(parts[2])
            except ValueError:
                return FileCheck(name=name, required=True, status="invalid", lines=count,
                                 detail=f"line {lineno}: not numeric")
            if not (0 <= a < num_frames and 0 <= b < num_frames):
                return FileCheck(name=name, required=True, status="invalid", lines=count,
                                 detail=f"line {lineno}: node id out of range 0..{num_frames - 1}")
            count += 1
    return FileCheck(name=name, required=True, status="ok", lines=count)


def _all_zero(rows: Dict[str, List[str]]) -> bool:
    for values in rows.values():
        for v in values:
            try:
                x = float(v)
            except ValueError:
                return False
            if not math.isnan(x) and x != 0.0:
                return False
    return True


def validate_submap_dir(map_dir: Path) -> ValidationReport:
    map_dir = Path(map_dir)
    files: List[FileCheck] = []
    warnings: List[str] = []
    errors: List[str] = []

    poses_check, poses = _check_keyed_file(map_dir, "poses.txt", 8, True, [])
    files.append(poses_check)
    if poses is None or not poses:
        errors.append("poses.txt is missing or unreadable; nothing else can be checked")
        return ValidationReport(ok=False, files=files, errors=errors)
    keys = list(poses.keys())
    num_frames = len(keys)

    for name, width in REQUIRED_FILES:
        check, _ = _check_keyed_file(map_dir, name, width, True, keys)
        files.append(check)
    descriptor_dim: Optional[int] = None
    desc_check = next(f for f in files if f.name == "database_descriptors.txt")
    if desc_check.status in ("ok", "incomplete"):
        rows, _ = _read_keyed(map_dir / "database_descriptors.txt")
        descriptor_dim = len(next(iter(rows.values()))) if rows else None

    has_gt = has_gps = has_iqa = False
    for name, width in OPTIONAL_FILES:
        check, rows = _check_keyed_file(map_dir, name, width, False, keys)
        files.append(check)
        if check.status == "ok" and rows is not None:
            if name == "poses_abs_gt.txt":
                has_gt = not _all_zero(rows)
            elif name == "iqa_data.txt":
                has_iqa = True
    gps_check = next(f for f in files if f.name == "gps_data.txt")
    if gps_check.status == "ok":
        rows, _ = _read_keyed(map_dir / "gps_data.txt")
        has_gps = not _all_zero(rows)

    for name in EDGE_FILES:
        files.append(_check_edge_file(map_dir, name, num_frames))

    missing_images = [k for k in keys if not (map_dir / k).is_file()]
    if missing_images:
        errors.append(f"{len(missing_images)} image(s) referenced by poses.txt are missing, "
                      f"e.g. {missing_images[0]}")

    for check in files:
        if check.status == "missing" and check.required:
            errors.append(f"{check.name} is required but missing")
        elif check.status == "incomplete":
            msg = f"{check.name}: {check.detail}; the loader would silently drop those frames"
            (errors if check.required else warnings).append(msg)
        elif check.status == "invalid":
            errors.append(f"{check.name}: {check.detail}")
        elif check.status == "missing":
            warnings.append(f"{check.name} not present (optional)")

    return ValidationReport(
        ok=not errors, num_frames=num_frames, has_gt=has_gt, has_gps=has_gps, has_iqa=has_iqa,
        descriptor_dim=descriptor_dim, files=files, warnings=warnings, errors=errors,
    )
