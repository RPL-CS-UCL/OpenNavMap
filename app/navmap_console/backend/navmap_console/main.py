"""FastAPI application factory; serves the built frontend when present."""
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, load_settings
from .jobs.bus import EventBus
from .jobs.runner import JobRunner
from .jobs.store import JobStore
from .routers import fs, health, jobs, regions, runs, sessions, ws
from .services.catalog import RegionStore, SessionStore
from .services.runs import RunJobHooks, RunService, RunStore


def _ensure_layout(settings: Settings) -> None:
    for d in (settings.regions_dir, settings.jobs_dir, settings.uploads_dir):
        d.mkdir(parents=True, exist_ok=True)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _ensure_layout(settings)
        py_dir = str(settings.repo_root / "python")
        if py_dir not in sys.path:
            sys.path.insert(0, py_dir)  # map_merge_pack (numpy-only) is imported lazily by services.runs
        bus = EventBus()
        job_store = JobStore(settings.jobs_dir)
        runner = JobRunner(settings, job_store, bus)
        run_service = RunService(settings, runner, RegionStore(settings.regions_dir),
                                 SessionStore(settings.regions_dir), RunStore(settings.regions_dir))
        runner.hooks = RunJobHooks(run_service)
        app.state.bus, app.state.job_store = bus, job_store
        app.state.runner, app.state.run_service = runner, run_service
        await runner.start()  # queues must be created inside the running loop
        try:
            yield
        finally:
            await runner.stop()  # leaves running subprocesses alive; they are re-adopted on restart

    app = FastAPI(
        title="navmap_console", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(regions.router)
    app.include_router(sessions.router)
    app.include_router(runs.router)
    app.include_router(jobs.router)
    app.include_router(fs.router)
    app.include_router(ws.router)

    dist = settings.frontend_dist
    if dist.is_dir():
        app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = dist / path
            if path and candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(dist / "index.html"))

    return app


app = create_app()
