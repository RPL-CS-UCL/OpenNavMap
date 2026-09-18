import io
import zipfile
from pathlib import Path

import pytest

from navmap_console.services.ingest import (extract_archive, locate_map_root,
                                            resolve_allowed_path, session_from_dir)


def _zip_of(src: Path, prefix: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, prefix + str(p.relative_to(src)))
    return buf.getvalue()


def test_extract_zip_and_locate_root(tmp_path: Path, synthetic_map: Path):
    archive = tmp_path / "m.zip"
    archive.write_bytes(_zip_of(synthetic_map, prefix="000/"))
    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert locate_map_root(dest) == dest / "000"


def test_extract_zip_rejects_traversal(tmp_path: Path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "x")
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_unknown_suffix(tmp_path: Path):
    (tmp_path / "x.tar").write_bytes(b"")
    with pytest.raises(ValueError):
        extract_archive(tmp_path / "x.tar", tmp_path / "o")


def test_locate_root_ambiguous(tmp_path: Path):
    for n in ("a", "b"):
        (tmp_path / n).mkdir()
        (tmp_path / n / "poses.txt").write_text("")
    with pytest.raises(ValueError):
        locate_map_root(tmp_path)


def test_resolve_allowed_path(tmp_path: Path):
    inside = tmp_path / "ok"
    inside.mkdir()
    assert resolve_allowed_path(str(inside), [tmp_path]) == inside.resolve()
    with pytest.raises(PermissionError):
        resolve_allowed_path("/etc", [tmp_path])
    with pytest.raises(FileNotFoundError):
        resolve_allowed_path(str(tmp_path / "missing"), [tmp_path])


def test_session_from_dir(tmp_path: Path, synthetic_map: Path):
    s = session_from_dir("reg_1", "000", "path", synthetic_map)
    assert s.num_frames == 12 and s.validation is not None and s.validation.ok
    assert s.id.startswith("ses_") and s.path == str(synthetic_map.resolve())
