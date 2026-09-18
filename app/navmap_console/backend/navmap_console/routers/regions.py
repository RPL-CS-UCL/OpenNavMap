"""Region CRUD and the region's current final map pointer (head)."""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from ..models import Region, RegionCreate, RegionHead, RegionPatch
from ..services.catalog import RegionStore, SessionStore, get_region_store, get_session_store

router = APIRouter(prefix="/api/regions", tags=["regions"])


class PromoteRequest(BaseModel):
    run_id: str
    step_index: int


def _with_counts(region: Region, regions: RegionStore, sessions: SessionStore) -> Dict[str, Any]:
    body = region.model_dump()
    body["session_count"] = len(sessions.list(region.id))
    runs_dir = regions.runs_dir(region.id)
    body["run_count"] = sum(1 for p in runs_dir.iterdir() if (p / "run.json").is_file()) if runs_dir.is_dir() else 0
    return body


def _get_or_404(regions: RegionStore, rid: str) -> Region:
    try:
        return regions.get(rid)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"region {rid} not found")


@router.get("")
def list_regions(regions: RegionStore = Depends(get_region_store),
                 sessions: SessionStore = Depends(get_session_store)) -> List[Dict[str, Any]]:
    return [_with_counts(r, regions, sessions) for r in regions.list()]


@router.post("", status_code=201)
def create_region(data: RegionCreate, regions: RegionStore = Depends(get_region_store),
                  sessions: SessionStore = Depends(get_session_store)) -> Dict[str, Any]:
    return _with_counts(regions.create(data), regions, sessions)


@router.get("/{rid}")
def get_region(rid: str, regions: RegionStore = Depends(get_region_store),
               sessions: SessionStore = Depends(get_session_store)) -> Dict[str, Any]:
    return _with_counts(_get_or_404(regions, rid), regions, sessions)


@router.patch("/{rid}")
def patch_region(rid: str, data: RegionPatch, regions: RegionStore = Depends(get_region_store),
                 sessions: SessionStore = Depends(get_session_store)) -> Dict[str, Any]:
    _get_or_404(regions, rid)
    return _with_counts(regions.patch(rid, data), regions, sessions)


@router.delete("/{rid}", status_code=204, response_class=Response)
def delete_region(rid: str, regions: RegionStore = Depends(get_region_store)) -> Response:
    _get_or_404(regions, rid)
    regions.delete(rid)
    return Response(status_code=204)


@router.get("/{rid}/map")
def get_head(rid: str, regions: RegionStore = Depends(get_region_store)) -> Optional[RegionHead]:
    return _get_or_404(regions, rid).head


@router.post("/{rid}/map/promote")
def promote(rid: str, data: PromoteRequest, regions: RegionStore = Depends(get_region_store),
            sessions: SessionStore = Depends(get_session_store)) -> Dict[str, Any]:
    region = _get_or_404(regions, rid)
    previous = region.head.lineage if region.head else []
    head = RegionHead(run_id=data.run_id, step_index=data.step_index,
                      session_ids=region.head.session_ids if region.head else [],
                      lineage=previous + [data.run_id] if data.run_id not in previous else previous)
    return _with_counts(regions.save(region.model_copy(update={"head": head})), regions, sessions)
