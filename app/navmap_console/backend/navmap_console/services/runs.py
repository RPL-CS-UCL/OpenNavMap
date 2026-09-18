"""Run records under regions/<rid>/runs/<run_id>/ (spec §5.3) and the run service."""
from pathlib import Path
from typing import List

from ..models import Run, StepRecord
from ..store import list_records, read_json, write_json_atomic


class RunStore:
    def __init__(self, regions_root: Path) -> None:
        self.root = regions_root

    def run_dir(self, rid: str, run_id: str) -> Path:
        return self.root / rid / "runs" / run_id

    def list(self, rid: str) -> List[Run]:
        return [Run.model_validate(r) for r in list_records(self.root / rid / "runs", "run.json")]

    def get(self, rid: str, run_id: str) -> Run:
        p = self.run_dir(rid, run_id) / "run.json"
        if not p.is_file():
            raise KeyError(run_id)
        return Run.model_validate(read_json(p))

    def save(self, run: Run) -> Run:
        write_json_atomic(self.run_dir(run.region_id, run.id) / "run.json", run.model_dump())
        return run

    def read_steps(self, rid: str, run_id: str) -> List[StepRecord]:
        p = self.run_dir(rid, run_id) / "steps.json"
        if not p.is_file():
            return []
        return [StepRecord.model_validate(s) for s in read_json(p)["steps"]]

    def write_steps(self, rid: str, run_id: str, steps: List[StepRecord]) -> None:
        write_json_atomic(self.run_dir(rid, run_id) / "steps.json", {"steps": [s.model_dump() for s in steps]})
