import json
import re
from datetime import datetime
from pathlib import Path

from navmap_console.store import list_records, new_id, read_json, write_json_atomic


def test_new_id_is_sortable_and_random():
    a = new_id("run", datetime(2026, 9, 18, 9, 30, 12))
    assert re.fullmatch(r"run_20260918_093012_[a-z0-9]{4}", a)
    assert new_id("run") != new_id("run")


def test_write_json_atomic_creates_parents_and_leaves_no_temp(tmp_path: Path):
    target = tmp_path / "a" / "b" / "x.json"
    write_json_atomic(target, {"k": 1, "nested": {"z": [1, 2]}})
    assert read_json(target) == {"k": 1, "nested": {"z": [1, 2]}}
    assert [p.name for p in target.parent.iterdir()] == ["x.json"]


def test_write_json_atomic_replaces_existing(tmp_path: Path):
    target = tmp_path / "x.json"
    write_json_atomic(target, {"v": 1})
    write_json_atomic(target, {"v": 2})
    assert json.loads(target.read_text())["v"] == 2


def test_list_records_sorted_and_skips_incomplete(tmp_path: Path):
    for name, payload in [("r_b", {"id": "r_b"}), ("r_a", {"id": "r_a"})]:
        write_json_atomic(tmp_path / name / "region.json", payload)
    (tmp_path / "r_c").mkdir()  # no region.json
    assert [r["id"] for r in list_records(tmp_path, "region.json")] == ["r_a", "r_b"]


def test_list_records_missing_root(tmp_path: Path):
    assert list_records(tmp_path / "nope", "x.json") == []


def test_models_roundtrip():
    from navmap_console.models import Region, Session, ValidationReport

    r = Region(id="reg_1", name="campus")
    assert Region.model_validate(r.model_dump()) == r
    s = Session(id="ses_1", region_id="reg_1", name="000", source="path", path="/x",
                validation=ValidationReport(ok=True, num_frames=3))
    assert Session.model_validate(s.model_dump()).validation.num_frames == 3
