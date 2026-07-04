from pathlib import Path
from visualization.map_merge_offline_to_events import detect_merge_dirs


def test_detect_merge_dirs_finds_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "merge_0").mkdir()
    (tmp_path / "merge_0_1_2").mkdir()
    (tmp_path / "merge_0_1").mkdir()
    (tmp_path / "merge_finalmap").write_text("link")  # file, not dir
    (tmp_path / "not_merge").mkdir()

    result = detect_merge_dirs(tmp_path)
    names = [p.name for p in result]
    assert names == ["merge_0", "merge_0_1", "merge_0_1_2"]


def test_detect_merge_dirs_empty_when_no_merge_dirs(tmp_path: Path) -> None:
    (tmp_path / "other").mkdir()
    assert detect_merge_dirs(tmp_path) == []
