"""Shared fixtures: an app whose data root is a temp dir."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[2]
for p in (BACKEND_DIR, REPO_ROOT / "python", REPO_ROOT / "third_party" / "litevloc_code" / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SYNTHETIC_MAP = REPO_ROOT / "python" / "visualization" / "example_data" / "synthetic_map"


@pytest.fixture
def settings(tmp_path: Path):
    from navmap_console.config import load_settings

    return load_settings({
        "NAVMAP_CONSOLE_DATA_ROOT": str(tmp_path / "data"),
        "NAVMAP_CONSOLE_ALLOWED_ROOTS": str(tmp_path),
        "NAVMAP_CONSOLE_REPO": str(REPO_ROOT),
        "NAVMAP_CONSOLE_FAKE_PIPELINE": "1",
        "NAVMAP_CONSOLE_PYTHON": sys.executable,
        "NAVMAP_CONSOLE_EVAL_PYTHON": sys.executable,
        "MERGE_CPU_LIST": "",
    })


@pytest.fixture
def client(settings):
    from navmap_console.main import create_app

    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def synthetic_map() -> Path:
    assert SYNTHETIC_MAP.is_dir(), SYNTHETIC_MAP
    return SYNTHETIC_MAP
