"""Job queue endpoints."""
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ..jobs.progress import read_log_lines
from ..jobs.runner import JobRunner
from ..models import Job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def get_runner(request: Request) -> JobRunner:
    return request.app.state.runner


@router.get("", response_model=List[Job])
def list_jobs(request: Request, status: Optional[str] = None,
              limit: int = Query(100, ge=1, le=1000)) -> List[Job]:
    jobs = get_runner(request).store.list()
    if status:
        wanted = set(status.split(","))
        jobs = [j for j in jobs if j.status in wanted]
    return jobs[:limit]


@router.get("/{jid}", response_model=Job)
def get_job(jid: str, request: Request) -> Job:
    try:
        return get_runner(request).store.get(jid)
    except KeyError:
        raise HTTPException(404, f"job {jid} not found")


@router.post("/{jid}/cancel", response_model=Job)
async def cancel_job(jid: str, request: Request) -> Job:
    try:
        return await get_runner(request).cancel(jid)
    except KeyError:
        raise HTTPException(404, f"job {jid} not found")


@router.get("/{jid}/log")
def job_log(jid: str, request: Request, after: int = Query(0, ge=0),
            limit: int = Query(5000, ge=1, le=50000)) -> Dict[str, Any]:
    """Lines [after, after+limit) of the job log; `next` is the cursor to pass on the next call."""
    runner = get_runner(request)
    try:
        job = runner.store.get(jid)
    except KeyError:
        raise HTTPException(404, f"job {jid} not found")
    lines = read_log_lines(Path(job.log_path))
    chunk = lines[after:after + limit]
    return {"lines": chunk, "next": after + len(chunk), "total": len(lines)}
