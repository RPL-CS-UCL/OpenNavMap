"""Log line handling for pipeline subprocesses: tqdm-aware splitting, progress anchors, crash kinds."""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

STAGE_NAMES: Dict[int, str] = {
    1: "load_reference", 2: "load_submap", 3: "difference_matrix", 4: "vpr_matching",
    5: "geometric_verification", 6: "metric_localization", 7: "pose_graph_optimization", 8: "finish",
}

_STEP_START = re.compile(r"--- Merging submap (\d+): (\S+) ---")
_DALL = re.compile(r"D_all shape:")
_PGO_INITIAL = re.compile(r"PGO: initial error: ([-+0-9.eE]+)")
_PGO_FINAL = re.compile(r"PGO: final error: ([-+0-9.eE]+)")
_SAVED = re.compile(r"Saved intermediate result: (\S+)")
_STEP_DONE = re.compile(r"STEP_DONE (.+)$")
_INT_FIELDS = ("index", "id_offset", "odom_nodes", "covis_nodes", "components", "registry")


def _last_segment(line: str) -> str:
    """tqdm redraws a line with '\\r'; only the text after the last one is what the terminal shows."""
    return line.rsplit("\r", 1)[-1]


class LineTailer:
    """Accumulates bytes and yields complete lines (without the newline)."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, chunk: bytes) -> List[str]:
        self._buf += chunk
        parts = self._buf.split(b"\n")
        self._buf = parts.pop()
        return [_last_segment(p.decode("utf-8", errors="replace")) for p in parts]

    @property
    def pending(self) -> int:
        """Bytes of an incomplete last line still buffered."""
        return len(self._buf)

    def flush(self) -> Optional[str]:
        if not self._buf:
            return None
        text, self._buf = self._buf, b""
        return _last_segment(text.decode("utf-8", errors="replace"))


def read_log_lines(path: Path) -> List[str]:
    if not path.is_file():
        return []
    text = path.read_bytes().decode("utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [_last_segment(l) for l in lines]


def parse_progress(line: str) -> Optional[Dict[str, Any]]:
    m = _STEP_START.search(line)
    if m:
        return {"kind": "step_start", "index": int(m.group(1)), "sid": m.group(2)}
    if _DALL.search(line):
        return {"kind": "stage", "stage_index": 4}
    m = _PGO_INITIAL.search(line)
    if m:
        return {"kind": "pgo_initial", "value": float(m.group(1))}
    m = _PGO_FINAL.search(line)
    if m:
        return {"kind": "pgo_final", "value": float(m.group(1))}
    m = _SAVED.search(line)
    if m:
        return {"kind": "step_saved", "dir": m.group(1)}
    m = _STEP_DONE.search(line)
    if m:
        out: Dict[str, Any] = {"kind": "step_done"}
        for token in m.group(1).split():
            key, _, value = token.partition("=")
            out[key] = int(value) if key in _INT_FIELDS else value
        return out
    return None


def classify_crash(returncode: Optional[int], tail: Sequence[str]) -> Optional[str]:
    """Best-effort crash kind from the exit status and the last log lines (spec §5.4)."""
    if returncode == 0:
        return None
    text = "\n".join(tail)
    if "CUDA out of memory" in text or "CUDA error: out of memory" in text:
        return "cuda_oom"
    if "'range_iterator' object is not callable" in text:
        return "bit_flip"  # CPython frame-slot corruption; see CLAUDE.md (non-ECC host, favored cores)
    if returncode == -11 or "Segmentation fault" in text or "segfault" in text:
        return "segfault"
    if returncode in (-9, 137) or "Killed" in text or "MemoryError" in text:
        return "oom"
    if returncode in (-15, -2, 143, 130):
        return "terminated"
    return "error"
