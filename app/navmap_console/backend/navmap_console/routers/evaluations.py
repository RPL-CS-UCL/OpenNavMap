"""Evaluation endpoints: manual re-run of the official report, list/detail, report files."""
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ..models import Job
from ..services.evaluations import EvalService
from .results import _call

router = APIRouter(prefix="/api", tags=["evaluations"])


def get_evals(request: Request) -> EvalService:
    return request.app.state.run_service.evals


@router.post("/regions/{rid}/runs/{run_id}/evaluate", response_model=Job)
def evaluate(rid: str, run_id: str, request: Request) -> Job:
    try:
        job = get_evals(request).enqueue_final_eval(rid, run_id)
    except KeyError:
        raise HTTPException(404, f"run {run_id} not found")
    if job is None:
        raise HTTPException(404, "no GT poses for this run: nothing to evaluate")
    return job


@router.get("/regions/{rid}/runs/{run_id}/evaluations")
async def list_evaluations(rid: str, run_id: str, request: Request) -> Dict[str, Any]:
    items = await _call(get_evals(request).list_reports, rid, run_id)
    return {"items": items, "evaluating": any(i.get("status") in ("queued", "running") for i in items)}


@router.get("/regions/{rid}/runs/{run_id}/evaluations/{eid}")
async def evaluation_detail(rid: str, run_id: str, eid: str, request: Request) -> Dict[str, Any]:
    return await _call(get_evals(request).report_detail, rid, run_id, eid)


@router.get("/regions/{rid}/runs/{run_id}/evaluations/{eid}/files/{name:path}")
async def evaluation_file(rid: str, run_id: str, eid: str, name: str, request: Request) -> FileResponse:
    path = await _call(get_evals(request).report_file, rid, run_id, eid, name)
    # inline: the browser renders pdf/png previews; the download link uses the `download` attribute
    return FileResponse(path, filename=path.name, content_disposition_type="inline")
