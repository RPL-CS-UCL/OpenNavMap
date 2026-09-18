def test_health_reports_config_and_machine(client, settings):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["version"]
    assert body["data_root"] == str(settings.data_root)
    assert body["queues"] == {"gpu": 0, "cpu": 0}
    assert set(body["disk"]) == {"total", "free"}
    assert "gpu" in body and "cpu_list" in body


def test_data_root_is_created_on_startup(client, settings):
    assert (settings.data_root / "regions").is_dir()
    assert (settings.data_root / "jobs").is_dir()
    assert (settings.data_root / "uploads").is_dir()


def test_unknown_api_path_is_not_served_as_spa(client, settings):
    """The SPA fallback must not swallow /api and /ws paths, or a typo'd endpoint returns HTML with 200."""
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
