"""Region and session records on disk (spec §5.3 layout)."""
import shutil
from pathlib import Path
from typing import List

from fastapi import Request

from ..models import Region, RegionCreate, RegionPatch, Session, VprConfig
from ..store import list_records, new_id, read_json, write_json_atomic


class RegionStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def region_dir(self, rid: str) -> Path:
        return self.root / rid

    def sessions_dir(self, rid: str) -> Path:
        return self.region_dir(rid) / "sessions"

    def runs_dir(self, rid: str) -> Path:
        return self.region_dir(rid) / "runs"

    def list(self) -> List[Region]:
        return [Region.model_validate(r) for r in list_records(self.root, "region.json")]

    def get(self, rid: str) -> Region:
        path = self.region_dir(rid) / "region.json"
        if not path.is_file():
            raise KeyError(rid)
        return Region.model_validate(read_json(path))

    def save(self, region: Region) -> Region:
        write_json_atomic(self.region_dir(region.id) / "region.json", region.model_dump())
        return region

    def create(self, data: RegionCreate) -> Region:
        region = Region(id=new_id("reg"), name=data.name, description=data.description,
                        vpr=data.vpr or VprConfig(), image_size=list(data.image_size))
        self.sessions_dir(region.id).mkdir(parents=True, exist_ok=True)
        self.runs_dir(region.id).mkdir(parents=True, exist_ok=True)
        (self.region_dir(region.id) / "exports").mkdir(exist_ok=True)
        return self.save(region)

    def patch(self, rid: str, data: RegionPatch) -> Region:
        region = self.get(rid)
        updated = region.model_copy(update=data.model_dump(exclude_none=True))
        return self.save(updated)

    def delete(self, rid: str) -> None:
        d = self.region_dir(rid)
        if not d.is_dir():
            raise KeyError(rid)
        shutil.rmtree(d)


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root  # regions dir

    def session_dir(self, rid: str, sid: str) -> Path:
        return self.root / rid / "sessions" / sid

    def list(self, rid: str) -> List[Session]:
        return [Session.model_validate(r) for r in list_records(self.root / rid / "sessions", "session.json")]

    def get(self, rid: str, sid: str) -> Session:
        path = self.session_dir(rid, sid) / "session.json"
        if not path.is_file():
            raise KeyError(sid)
        return Session.model_validate(read_json(path))

    def save(self, session: Session) -> Session:
        write_json_atomic(self.session_dir(session.region_id, session.id) / "session.json", session.model_dump())
        return session

    def delete(self, rid: str, sid: str, remove_data: bool) -> None:
        """Remove the session record; `remove_data` also deletes an uploaded data/ directory.

        Registered sessions keep their external directory untouched either way.
        """
        d = self.session_dir(rid, sid)
        if not d.is_dir():
            raise KeyError(sid)
        data = d / "data"
        if data.is_symlink():
            data.unlink()
        elif data.is_dir() and not remove_data:
            raise ValueError(f"session {sid} owns uploaded data; pass remove_data=True to delete it")
        shutil.rmtree(d)


def get_region_store(request: Request) -> RegionStore:
    return RegionStore(request.app.state.settings.regions_dir)


def get_session_store(request: Request) -> SessionStore:
    return SessionStore(request.app.state.settings.regions_dir)
