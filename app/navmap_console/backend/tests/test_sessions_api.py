import io
import shutil
import zipfile
from pathlib import Path


def _zip_bytes(src: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, "000/" + str(p.relative_to(src)))
    return buf.getvalue()


def _region(client) -> str:
    return client.post("/api/regions", json={"name": "r"}).json()["id"]


def test_register_path(client, synthetic_map, settings, tmp_path):
    rid = _region(client)
    r = client.post(f"/api/regions/{rid}/sessions/register", json={"path": str(synthetic_map)})
    assert r.status_code == 400  # synthetic_map is outside allowed roots (tmp_path)

    local = tmp_path / "s01"
    shutil.copytree(synthetic_map, local)
    r = client.post(f"/api/regions/{rid}/sessions/register", json={"path": str(local)})
    assert r.status_code == 201, r.text
    s = r.json()
    assert s["source"] == "path" and s["name"] == "s01" and s["num_frames"] == 12
    assert s["validation"]["ok"]
    assert client.get(f"/api/regions/{rid}/sessions").json()[0]["id"] == s["id"]

    v = client.post(f"/api/regions/{rid}/sessions/{s['id']}/validate")
    assert v.status_code == 200 and v.json()["validation"]["num_frames"] == 12

    assert client.delete(f"/api/regions/{rid}/sessions/{s['id']}").status_code == 204
    assert local.is_dir()  # registered data is never deleted
    assert client.get(f"/api/regions/{rid}/sessions").json() == []


def test_upload_zip(client, synthetic_map, settings):
    rid = _region(client)
    files = {"file": ("mymap.zip", _zip_bytes(synthetic_map), "application/zip")}
    r = client.post(f"/api/regions/{rid}/sessions/upload", files=files)
    assert r.status_code == 201, r.text
    s = r.json()
    assert s["source"] == "upload" and s["name"] == "mymap"
    data_dir = Path(s["path"])
    assert str(data_dir).startswith(str(settings.regions_dir))
    assert (data_dir / "poses.txt").is_file()
    assert not list(settings.uploads_dir.iterdir())  # temp upload removed

    assert client.delete(f"/api/regions/{rid}/sessions/{s['id']}").status_code == 204
    assert not data_dir.exists()


def test_upload_rejects_bad_archive(client):
    rid = _region(client)
    files = {"file": ("x.zip", b"not a zip", "application/zip")}
    assert client.post(f"/api/regions/{rid}/sessions/upload", files=files).status_code == 400


def test_session_404(client):
    rid = _region(client)
    assert client.get(f"/api/regions/{rid}/sessions/nope").status_code == 404
    assert client.get("/api/regions/nope/sessions").status_code == 404
