"""File-backed storage: one JSON per object, atomic writes, directory listings."""
import json
import os
import random
import string
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ID_ALPHABET = string.ascii_lowercase + string.digits


def new_id(prefix: str, now: Optional[datetime] = None) -> str:
    """Human-sortable id: <prefix>_<YYYYmmdd_HHMMSS>_<4 random chars>."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    tail = "".join(random.choice(_ID_ALPHABET) for _ in range(4))
    return f"{prefix}_{stamp}_{tail}"


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write to a temp file in the same directory, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(path))
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_records(root: Path, filename: str) -> List[Dict[str, Any]]:
    """Load root/<dir>/<filename> for every subdirectory that has it, sorted by dir name."""
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        record = d / filename
        if record.is_file():
            out.append(read_json(record))
    return out
