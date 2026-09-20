# tests/test_image_events.py
import json
import shutil
from pathlib import Path

from conftest import SYNTHETIC_MAP
from PIL import Image


def _step_dirs(tmp_path: Path):
    """Two fake step dirs: the first holds frames 0-11, the second frames 12-23 plus an override of frame 0."""
    a, b = tmp_path / "merge_000_a", tmp_path / "merge_001_b"
    for d in (a, b):
        (d / "seq").mkdir(parents=True)
    src = sorted((SYNTHETIC_MAP / "seq").glob("*.color.jpg"))
    for i, img in enumerate(src):
        shutil.copy2(img, a / "seq" / f"{i:06d}.color.jpg")
        shutil.copy2(img, b / "seq" / f"{i + 12:06d}.color.jpg")
    shutil.copy2(src[1], b / "seq" / "000000.color.jpg")
    return a, b


def test_image_index_prefers_later_sources(tmp_path: Path):
    from navmap_console.readers.image_index import ImageIndex

    a, b = _step_dirs(tmp_path)
    idx = ImageIndex([a, b])
    assert idx.path(5) == a / "seq" / "000005.color.jpg"
    assert idx.path(17) == b / "seq" / "000017.color.jpg"
    assert idx.path(0) == b / "seq" / "000000.color.jpg"
    assert idx.path(99) is None
    (b / "seq" / "000099.color.jpg").write_bytes((a / "seq" / "000001.color.jpg").read_bytes())
    assert idx.path(99) is None  # stale until refreshed
    idx.refresh()
    assert idx.path(99) is not None


def test_thumbnail_and_pair_image(tmp_path: Path):
    from navmap_console.readers.image_index import pair_image, thumbnail

    a, _ = _step_dirs(tmp_path)
    cache = tmp_path / "thumbs"
    t = thumbnail(a / "seq" / "000001.color.jpg", 64, cache)
    assert t.parent == cache and t.suffix == ".jpg"
    with Image.open(t) as im:
        assert im.width == 64 and im.height > 0
    assert thumbnail(a / "seq" / "000001.color.jpg", 64, cache) == t  # cached, same path
    p = pair_image(a / "seq" / "000001.color.jpg", a / "seq" / "000002.color.jpg", 64, cache)
    with Image.open(p) as im:
        assert im.width == 64 * 2 + 4


def _event(step: int, etype: str, **payload) -> str:
    return json.dumps({"demo_step": step, "merge_step": step, "stage": "vpr", "event_type": etype,
                       "submap_id": step, "keyframe_id": None, "payload": payload, "artifacts": {}})


def test_event_index_incremental_and_filters(tmp_path: Path):
    from navmap_console.readers.events import EventIndex

    path = tmp_path / "demo_events.jsonl"
    idx = EventIndex(path)
    assert idx.refresh() == 0 and idx.read(None, None, 10, False) == []
    lines = [_event(0, "vpr_candidate", db=1), _event(1, "vpr_candidate", db=2),
             _event(1, "map_committed", nodes=[1, 2, 3])]
    partial = _event(1, "gv_candidate", db=3)
    path.write_text("\n".join(lines) + "\n" + partial[:20])
    assert idx.refresh() == 3
    assert [e["event_type"] for e in idx.read(1, None, 10, False)] == ["vpr_candidate", "map_committed"]
    assert idx.read(1, None, 10, False)[1]["payload"] == {"omitted": True}
    assert idx.read(1, None, 10, True)[1]["payload"] == {"nodes": [1, 2, 3]}
    assert idx.read(None, ["vpr_candidate"], 10, False)[1]["payload"] == {"db": 2}
    assert len(idx.read(None, None, 2, False)) == 2
    with path.open("a") as f:
        f.write(partial[20:] + "\n")
    assert idx.refresh() == 1
    assert [e["event_type"] for e in idx.read(1, ["gv_candidate"], 10, False)] == ["gv_candidate"]
