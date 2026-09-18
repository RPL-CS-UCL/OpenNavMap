"""Make `rosbag_convert` and litevloc `utils` importable without installing."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "python", _REPO / "third_party" / "litevloc_code" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
