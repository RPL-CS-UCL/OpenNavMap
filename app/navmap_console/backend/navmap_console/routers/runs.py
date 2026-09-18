"""Runs and the merge parameter schema."""
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from ..jobs.specs import params_schema
from ..models import Run, RunCreate, StepRecord
from ..services.runs import RegionBusyError, RunService

router = APIRouter(prefix="/api", tags=["runs"])


def get_run_service(request: Request) -> RunService:
    return request.app.state.run_service


@router.get("/params/merge")
def merge_params() -> List[Dict[str, Any]]:
    return params_schema()


@router.get("/regions/{rid}/runs", response_model=List[Run])
def list_runs(rid: str, request: Request) -> List[Run]:
    svc = get_run_service(request)
    try:
        svc.regions.get(rid)
    except KeyError:
        raise HTTPException(404, f"region {rid} not found")
    return sorted(svc.runs.list(rid), key=lambda r: r.created_at, reverse=True)


@router.post("/regions/{rid}/runs", response_model=Run, status_code=201)
def create_run(rid: str, data: RunCreate, request: Request) -> Run:
    svc = get_run_service(request)
    try:
        return svc.create(rid, data)
    except KeyError as exc:
        raise HTTPException(404, f"not found: {exc}")
    except RegionBusyError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/regions/{rid}/runs/{run_id}")
def run_detail(rid: str, run_id: str, request: Request) -> Dict[str, Any]:
    try:
        return get_run_service(request).detail(rid, run_id)
    except KeyError:
        raise HTTPException(404, f"run {run_id} not found")


@router.get("/regions/{rid}/runs/{run_id}/steps", response_model=List[StepRecord])
def run_steps(rid: str, run_id: str, request: Request) -> List[StepRecord]:
    svc = get_run_service(request)
    try:
        svc.runs.get(rid, run_id)
    except KeyError:
        raise HTTPException(404, f"run {run_id} not found")
    return svc.runs.read_steps(rid, run_id)


@router.post("/regions/{rid}/runs/{run_id}/cancel", response_model=Run)
async def cancel_run(rid: str, run_id: str, request: Request) -> Run:
    try:
        return await get_run_service(request).cancel(rid, run_id)
    except KeyError:
        raise HTTPException(404, f"run {run_id} not found")
