"""Sessions of a region: upload, register, validate, delete."""
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile

from ..models import RegisterSessionRequest, Session
from ..services.catalog import RegionStore, SessionStore, get_region_store, get_session_store
from ..services.ingest import (ARCHIVE_SUFFIXES, extract_archive, locate_map_root,
                               resolve_allowed_path, session_from_dir)
from ..services.validation import validate_submap_dir
from ..store import new_id

router = APIRouter(prefix="/api/regions/{rid}/sessions", tags=["sessions"])


def _region_or_404(regions: RegionStore, rid: str) -> None:
    try:
        regions.get(rid)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"region {rid} not found")


def _session_or_404(sessions: SessionStore, rid: str, sid: str) -> Session:
    try:
        return sessions.get(rid, sid)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"session {sid} not found")


@router.get("")
def list_sessions(rid: str, regions: RegionStore = Depends(get_region_store),
                  sessions: SessionStore = Depends(get_session_store)) -> List[Session]:
    _region_or_404(regions, rid)
    return sessions.list(rid)


@router.post("/register", status_code=201)
def register_session(rid: str, data: RegisterSessionRequest, request: Request,
                     regions: RegionStore = Depends(get_region_store),
                     sessions: SessionStore = Depends(get_session_store)) -> Session:
    _region_or_404(regions, rid)
    try:
        path = resolve_allowed_path(data.path, request.app.state.settings.allowed_roots)
    except (PermissionError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    session = session_from_dir(rid, data.name or path.name, "path", path)
    return sessions.save(session)


@router.post("/upload", status_code=201)
async def upload_session(rid: str, request: Request, file: UploadFile = File(...),
                         name: Optional[str] = Form(default=None),
                         regions: RegionStore = Depends(get_region_store),
                         sessions: SessionStore = Depends(get_session_store)) -> Session:
    _region_or_404(regions, rid)
    filename = Path(file.filename or "upload.zip")
    if filename.suffix.lower() not in ARCHIVE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"unsupported archive type {filename.suffix}")
    settings = request.app.state.settings
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    sid = new_id("ses")
    tmp_archive = settings.uploads_dir / f"{sid}{filename.suffix.lower()}"
    async with aiofiles.open(tmp_archive, "wb") as out:
        while True:
            chunk = await file.read(4 * 1024 * 1024)
            if not chunk:
                break
            await out.write(chunk)

    data_dir = sessions.session_dir(rid, sid) / "data"
    try:
        extract_archive(tmp_archive, data_dir)
        map_root = locate_map_root(data_dir)
    except (ValueError, zipfile.BadZipFile, subprocess.CalledProcessError, OSError) as e:
        shutil.rmtree(sessions.session_dir(rid, sid), ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"could not extract archive: {e}")
    finally:
        tmp_archive.unlink(missing_ok=True)

    session = session_from_dir(rid, name or filename.stem, "upload", map_root)
    session = session.model_copy(update={"id": sid})
    return sessions.save(session)


@router.get("/{sid}")
def get_session(rid: str, sid: str, sessions: SessionStore = Depends(get_session_store)) -> Session:
    return _session_or_404(sessions, rid, sid)


@router.post("/{sid}/validate")
def revalidate(rid: str, sid: str, sessions: SessionStore = Depends(get_session_store)) -> Session:
    session = _session_or_404(sessions, rid, sid)
    report = validate_submap_dir(Path(session.path))
    updated = session.model_copy(update={
        "validation": report, "num_frames": report.num_frames,
        "has_gt": report.has_gt, "has_gps": report.has_gps, "has_iqa": report.has_iqa,
    })
    return sessions.save(updated)


@router.delete("/{sid}", status_code=204, response_class=Response)
def delete_session(rid: str, sid: str, sessions: SessionStore = Depends(get_session_store)) -> Response:
    session = _session_or_404(sessions, rid, sid)
    sessions.delete(rid, sid, remove_data=(session.source == "upload"))
    return Response(status_code=204)
