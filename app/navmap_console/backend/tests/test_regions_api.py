def test_region_crud(client):
    assert client.get("/api/regions").json() == []
    r = client.post("/api/regions", json={"name": "campus", "description": "d"})
    assert r.status_code == 201
    region = r.json()
    assert region["id"].startswith("reg_") and region["vpr"]["dim"] == 256
    assert region["session_count"] == 0

    rid = region["id"]
    assert client.get(f"/api/regions/{rid}").json()["name"] == "campus"
    assert client.patch(f"/api/regions/{rid}", json={"name": "campus2"}).json()["name"] == "campus2"
    assert [x["id"] for x in client.get("/api/regions").json()] == [rid]
    assert client.get(f"/api/regions/{rid}/map").json() is None

    p = client.post(f"/api/regions/{rid}/map/promote", json={"run_id": "run_x", "step_index": 3})
    assert p.status_code == 200 and p.json()["head"]["step_index"] == 3
    assert client.get(f"/api/regions/{rid}/map").json()["run_id"] == "run_x"

    assert client.delete(f"/api/regions/{rid}").status_code == 204
    assert client.get(f"/api/regions/{rid}").status_code == 404


def test_region_validation(client):
    assert client.post("/api/regions", json={"name": ""}).status_code == 422
    assert client.get("/api/regions/nope").status_code == 404


def test_region_persists_on_disk(client, settings):
    rid = client.post("/api/regions", json={"name": "a"}).json()["id"]
    assert (settings.regions_dir / rid / "region.json").is_file()
    assert (settings.regions_dir / rid / "sessions").is_dir()
    assert (settings.regions_dir / rid / "runs").is_dir()
