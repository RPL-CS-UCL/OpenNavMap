"""FastAPI application factory; serves the built frontend when present."""
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, load_settings
from .routers import health, regions, sessions


def _ensure_layout(settings: Settings) -> None:
    for d in (settings.regions_dir, settings.jobs_dir, settings.uploads_dir):
        d.mkdir(parents=True, exist_ok=True)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        _ensure_layout(settings)
        yield

    app = FastAPI(
        title="navmap_console", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(regions.router)
    app.include_router(sessions.router)

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
