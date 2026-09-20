"""Byte-offset index over rerun_viz/demo_events.jsonl; never loads the whole file."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SNAPSHOT_TYPES = ("map_committed",)


class EventIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._rows: List[Tuple[int, int, int, str]] = []  # (byte offset, byte length, merge_step, event_type)

    def refresh(self) -> int:
        if not self.path.is_file():
            return 0
        size = self.path.stat().st_size
        if size < self._offset:  # file was truncated/rewritten: start over
            self._offset, self._rows = 0, []
        if size == self._offset:
            return 0
        added = 0
        with self.path.open("rb") as f:
            f.seek(self._offset)
            chunk = f.read(size - self._offset)
        pos = self._offset
        for raw in chunk.split(b"\n")[:-1]:  # the last piece is either "" or a half line: not consumed
            length = len(raw) + 1
            try:
                ev = json.loads(raw.decode("utf-8"))
                self._rows.append((pos, len(raw), int(ev.get("merge_step", -1)), str(ev.get("event_type", ""))))
                added += 1
            except (ValueError, TypeError):
                pass  # skip corrupt line but keep the offset moving
            pos += length
        self._offset = pos
        return added

    def read(self, step: Optional[int], types: Optional[Sequence[str]], limit: int,
             include_snapshots: bool) -> List[Dict[str, Any]]:
        wanted = set(types) if types else None
        picks = [r for r in self._rows if (step is None or r[2] == step) and (wanted is None or r[3] in wanted)]
        picks = picks[:max(0, limit)]
        out: List[Dict[str, Any]] = []
        if not picks:
            return out
        with self.path.open("rb") as f:
            for offset, length, _, etype in picks:
                f.seek(offset)
                ev = json.loads(f.read(length).decode("utf-8"))
                if etype in SNAPSHOT_TYPES and not include_snapshots:
                    ev["payload"] = {"omitted": True}
                out.append(ev)
        return out
