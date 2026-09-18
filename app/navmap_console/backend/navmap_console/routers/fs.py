"""Server-side directory browser, restricted to NAVMAP_CONSOLE_ALLOWED_ROOTS."""
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ..services.ingest import ARCHIVE_SUFFIXES, resolve_allowed_path

router = APIRouter(prefix="/api/fs", tags=["fs"])
_LISTED_SUFFIXES = ARCHIVE_SUFFIXES + (".txt",)


def _looks_like_session(d: Path) -> bool:
    return (d / "poses.txt").is_file() and (d / "seq").is_dir()


@router.get("/roots")
def roots(request: Request) -> Dict[str, List[str]]:
    return {"roots": [str(r) for r in request.app.state.settings.allowed_roots]}


@router.get("/list")
def list_dir(request: Request, path: str = Query(...)) -> Dict[str, Any]:
    allowed = request.app.state.settings.allowed_roots
    try:
        target = resolve_allowed_path(path, allowed)
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    parent: Optional[str] = None if target in allowed else str(target.parent)
    entries: List[Dict[str, Any]] = []
    try:
        children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append({"name": child.name, "path": str(child), "is_dir": True,
                            "looks_like_session": _looks_like_session(child), "size": None})
        elif child.suffix.lower() in _LISTED_SUFFIXES:
            entries.append({"name": child.name, "path": str(child), "is_dir": False,
                            "looks_like_session": False, "size": child.stat().st_size})
    return {"path": str(target), "parent": parent, "entries": entries}
