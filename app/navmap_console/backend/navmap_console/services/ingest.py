"""Bring session data in: extract uploads, resolve registered paths, build Session records."""
import subprocess
import zipfile
from pathlib import Path
from typing import Sequence

from ..models import Session
from ..store import new_id
from .validation import validate_submap_dir

ARCHIVE_SUFFIXES = (".zip", ".7z")


def extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        root = dest.resolve()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (dest / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise ValueError(f"archive member escapes destination: {member.filename}")
            z.extractall(dest)
    elif suffix == ".7z":
        subprocess.run(["7z", "x", "-y", f"-o{dest}", str(archive)],
                       check=True, capture_output=True, text=True)
    else:
        raise ValueError(f"unsupported archive type: {archive.name} (use {', '.join(ARCHIVE_SUFFIXES)})")


def locate_map_root(root: Path) -> Path:
    if (root / "poses.txt").is_file():
        return root
    found = sorted(p.parent for p in root.rglob("poses.txt"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError("no poses.txt found inside the archive")
    raise ValueError(f"archive holds {len(found)} map directories; upload one session at a time")


def resolve_allowed_path(raw: str, allowed_roots: Sequence[Path]) -> Path:
    path = Path(raw).expanduser().resolve()
    if not any(path == r or r in path.parents for r in allowed_roots):
        raise PermissionError(f"{path} is outside NAVMAP_CONSOLE_ALLOWED_ROOTS")
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory")
    return path


def session_from_dir(rid: str, name: str, source: str, data_dir: Path) -> Session:
    report = validate_submap_dir(data_dir)
    return Session(
        id=new_id("ses"), region_id=rid, name=name, source=source, path=str(data_dir.resolve()),
        num_frames=report.num_frames, has_gt=report.has_gt, has_gps=report.has_gps,
        has_iqa=report.has_iqa, validation=report,
    )
