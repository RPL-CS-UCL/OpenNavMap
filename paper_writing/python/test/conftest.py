import sys
from pathlib import Path

# Add opennavmap root so `paper_writing.python` is importable
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
