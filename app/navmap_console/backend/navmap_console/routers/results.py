"""Viewer data for one run: step summaries, scene.bin, distance matrix, culling, node details, images, events."""
import asyncio
import functools
from typing import Any, Callable, Dict, Optional, TypeVar

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from ..services.results import ResultService

router = APIRouter(prefix="/api/regions/{rid}/runs/{run_id}", tags=["results"])
T = TypeVar("T")


def get_results(request: Request) -> ResultService:
    return request.app.state.results


async def _call(fn: Callable[..., T], *args: Any) -> T:
    """File parsing runs in the default executor so a large step never blocks the event loop."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, functools.partial(fn, *args))
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0]) if exc.args else "not found")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/summaries")
async def summaries(rid: str, run_id: str, request: Request) -> Dict[str, Any]:
    steps = await _call(get_results(request).summaries, rid, run_id)
    return {"steps": [s.model_dump() for s in steps]}


@router.get("/steps/{k}/scene.bin")
async def scene(rid: str, run_id: str, k: int, request: Request) -> Response:
    data = await _call(get_results(request).scene, rid, run_id, k)
    return Response(content=data, media_type="application/octet-stream", headers={"Cache-Control": "no-cache"})


@router.get("/steps/{k}/dmatrix.json")
async def dmatrix_json(rid: str, run_id: str, k: int, request: Request) -> Dict[str, Any]:
    return await _call(get_results(request).dmatrix_json, rid, run_id, k)


@router.get("/steps/{k}/dmatrix.png")
async def dmatrix_png(rid: str, run_id: str, k: int, request: Request) -> Response:
    data, media = await _call(get_results(request).dmatrix_png, rid, run_id, k)
    return Response(content=data, media_type=media, headers={"Cache-Control": "no-cache"})


@router.get("/steps/{k}/culling.json")
async def culling(rid: str, run_id: str, k: int, request: Request) -> Dict[str, Any]:
    return await _call(get_results(request).culling, rid, run_id, k)


@router.get("/steps/{k}/nodes/{nid}")
async def node_detail(rid: str, run_id: str, k: int, nid: int, request: Request) -> Dict[str, Any]:
    return await _call(get_results(request).node_detail, rid, run_id, k, nid)


@router.get("/steps/{k}/preds/{name:path}")
async def preds_file(rid: str, run_id: str, k: int, name: str, request: Request) -> FileResponse:
    path = await _call(get_results(request).preds_file, rid, run_id, k, name)
    return FileResponse(str(path))


@router.get("/nodes/{nid}/image")
async def node_image(rid: str, run_id: str, nid: int, request: Request,
                     w: int = Query(512, ge=32, le=2048)) -> FileResponse:
    path = await _call(get_results(request).node_image, rid, run_id, nid, w)
    return FileResponse(str(path), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


@router.get("/pairs/{a}/{b}/image")
async def pair_image(rid: str, run_id: str, a: int, b: int, request: Request,
                     w: int = Query(384, ge=32, le=2048)) -> FileResponse:
    path = await _call(get_results(request).pair_image, rid, run_id, a, b, w)
    return FileResponse(str(path), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


@router.get("/events")
async def events(rid: str, run_id: str, request: Request, step: Optional[int] = None,
                 types: Optional[str] = None, limit: int = Query(500, ge=1, le=5000),
                 snapshots: bool = False) -> Any:
    wanted = [t for t in types.split(",") if t] if types else None
    return await _call(get_results(request).events, rid, run_id, step, wanted, limit, snapshots)
