"""Configure import paths for paper_writing scripts.

Importing this module makes the paper_writing scripts runnable without
requiring users to export PYTHONPATH. It follows the project convention
documented in opennavmap/CLAUDE.md:

    opennavmap/python
    opennavmap/third_party/litevloc_code/python
"""

from pathlib import Path
import sys


_OPENNAVMAP_ROOT = Path(__file__).resolve().parents[2]
_IMPORT_PATHS = (
    _OPENNAVMAP_ROOT / "python",
    _OPENNAVMAP_ROOT / "third_party" / "litevloc_code" / "python",
)

for _path in reversed(_IMPORT_PATHS):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
