import shutil
from pathlib import Path


def test_fs_roots(client, settings):
    assert client.get("/api/fs/roots").json() == {"roots": [str(r) for r in settings.allowed_roots]}


def test_fs_list_marks_sessions(client, tmp_path: Path, synthetic_map: Path):
    shutil.copytree(synthetic_map, tmp_path / "browse" / "000")
    (tmp_path / "browse" / "notes.txt").write_text("x")
    (tmp_path / "browse" / "junk.bin").write_bytes(b"0")
    r = client.get("/api/fs/list", params={"path": str(tmp_path / "browse")})
    assert r.status_code == 200, r.text
    body = r.json()
    names = [e["name"] for e in body["entries"]]
    assert names == ["000", "notes.txt"]
    assert body["entries"][0]["looks_like_session"] is True
    assert body["parent"] == str(tmp_path)


def test_fs_list_root_has_no_parent(client, tmp_path: Path):
    assert client.get("/api/fs/list", params={"path": str(tmp_path)}).json()["parent"] is None


def test_fs_list_outside_roots(client):
    assert client.get("/api/fs/list", params={"path": "/etc"}).status_code == 400


def test_fs_list_missing(client, tmp_path: Path):
    assert client.get("/api/fs/list", params={"path": str(tmp_path / "nope")}).status_code == 404
