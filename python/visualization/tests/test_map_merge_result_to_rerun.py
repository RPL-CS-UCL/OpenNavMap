from pathlib import Path

from visualization import map_merge_result_to_rerun


def test_parse_args_uses_rerun_prefixed_options() -> None:
    args = map_merge_result_to_rerun.parse_args(
        [
            "--result-dir",
            "/tmp/result",
            "--rerun-output",
            "/tmp/out.rrd",
            "--mode",
            "readonly",
            "--rerun-image-format",
            "jpg",
            "--rerun-jpeg-quality",
            "85",
            "--rerun-dmatrix-format",
            "png",
            "--rerun-axis-scale",
            "auto",
        ]
    )

    assert args.rerun_output == Path("/tmp/out.rrd")
    assert args.rerun_dmatrix_format == "png"


def test_main_hands_events_to_writer(tmp_path: Path, monkeypatch) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    captured = {}

    class FakeReplay:
        def __init__(self, path: Path) -> None:
            captured["result_dir"] = path

        def build_events(self):
            return ["event"]

    class FakeWriter:
        def __init__(self, *args, **kwargs) -> None:
            captured["writer_kwargs"] = kwargs

        def write(self, events, output_path: Path) -> None:
            captured["events"] = events
            captured["output"] = output_path

    monkeypatch.setattr(map_merge_result_to_rerun, "MapMergeResultReplay", FakeReplay)
    monkeypatch.setattr(map_merge_result_to_rerun, "MapMergeRerunWriter", FakeWriter)

    map_merge_result_to_rerun.main(
        [
            "--result-dir",
            str(result_dir),
            "--rerun-output",
            str(tmp_path / "out.rrd"),
        ]
    )

    assert captured["result_dir"] == result_dir
    assert captured["events"] == ["event"]
    assert captured["output"] == tmp_path / "out.rrd"
    assert captured["writer_kwargs"]["jpeg_quality"] == 85
