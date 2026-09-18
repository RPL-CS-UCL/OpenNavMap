"""Job records: jobs/<id>.json next to the subprocess log jobs/<id>.log."""
from pathlib import Path
from typing import List

from ..models import Job
from ..store import read_json, write_json_atomic


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, jid: str) -> Path:
        return self.root / f"{jid}.json"

    def log_path(self, jid: str) -> Path:
        return self.root / f"{jid}.log"

    def list(self) -> List[Job]:
        """All jobs, newest first (ids embed a timestamp, so name order is time order)."""
        if not self.root.is_dir():
            return []
        files = sorted((p for p in self.root.glob("job_*.json")), reverse=True)
        return [Job.model_validate(read_json(p)) for p in files]

    def get(self, jid: str) -> Job:
        p = self.path(jid)
        if not p.is_file():
            raise KeyError(jid)
        return Job.model_validate(read_json(p))

    def save(self, job: Job) -> Job:
        write_json_atomic(self.path(job.id), job.model_dump())
        return job
