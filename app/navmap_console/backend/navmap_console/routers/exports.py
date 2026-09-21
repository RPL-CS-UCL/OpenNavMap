"""Export endpoints: create/list/download/delete bundles (spec §4.4)."""
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..models import Job
from ..services.exports import ExportService
from .results import _call

router = APIRouter(prefix="/api/regions/{rid}/runs/{run_id}/exports", tags=["exports"])


class ExportCreate(BaseModel):
    kind: Literal["map", "report", "preds"]
    steps: Optional[List[int]] = None  # preds only; all steps when omitted


def get_exports(request: Request) -> ExportService:
    return request.app.state.export_service


@router.post("", response_model=Job, status_code=201)
def create_export(rid: str, run_id: str, data: ExportCreate, request: Request) -> Job:
    svc = get_exports(request)
    try:
        if data.kind == "map":
            sources = request.app.state.run_service.image_sources(rid, run_id)
            return svc.create(rid, run_id, data.kind, image_sources=sources)
        return svc.create(rid, run_id, data.kind, steps=data.steps)
    except KeyError:
        raise HTTPException(404, f"run {run_id} not found")


@router.get("")
async def list_exports(rid: str, run_id: str, request: Request) -> Dict[str, Any]:
    return {"items": await _call(get_exports(request).list, rid, run_id)}


@router.get("/{name}/download")
async def download_export(rid: str, run_id: str, name: str, request: Request) -> FileResponse:
    path = await _call(get_exports(request).path, rid, run_id, name)
    return FileResponse(str(path), filename=f"{name}.tar.gz", media_type="application/gzip")


@router.delete("/{name}", status_code=204)
def delete_export(rid: str, run_id: str, name: str, request: Request) -> None:
    try:
        get_exports(request).delete(rid, run_id, name)
    except KeyError:
        raise HTTPException(404, f"export {name} not found")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
