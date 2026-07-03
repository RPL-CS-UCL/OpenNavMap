import json
from pathlib import Path

from visualization.map_merge_runtime_events_to_rerun import parse_args
from visualization.map_merge_runtime_rerun_renderer import (
    MapMergeRuntimeRerunRenderer,
    load_runtime_events,
)


class FakeRerun:
    def __init__(self) -> None:
        self.logged = []
        self.times = []

    @staticmethod
    def TextDocument(text):
        return ("text", text)

    @staticmethod
    def Points3D(*args, **kwargs):
        return ("points3d", args, kwargs)

    @staticmethod
    def LineStrips3D(*args, **kwargs):
        return ("lines3d", args, kwargs)

    @staticmethod
    def EncodedImage(**kwargs):
        return ("encoded_image", kwargs)

    @staticmethod
    def Arrows3D(*args, **kwargs):
        return ("arrows3d", args, kwargs)

    def set_time_sequence(self, timeline, value) -> None:
        self.times.append((timeline, value))

    def log(self, entity_path, archetype) -> None:
        self.logged.append((entity_path, archetype))


def _write_event(event_dir: Path, event: dict) -> None:
    with (event_dir / "demo_events.jsonl").open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(event) + "\n")


def test_parse_args_accepts_event_dir_and_rerun_output() -> None:
    args = parse_args(
        [
            "--event-dir",
            "/tmp/rerun_viz",
            "--rerun-output",
            "/tmp/out.rrd",
        ]
    )

    assert args.event_dir == Path("/tmp/rerun_viz")
    assert args.rerun_output == Path("/tmp/out.rrd")


def test_load_runtime_events_reads_jsonl_in_order(tmp_path: Path) -> None:
    event_dir = tmp_path / "rerun_viz"
    event_dir.mkdir()
    _write_event(event_dir, {"demo_step": 0, "event_type": "a"})
    _write_event(event_dir, {"demo_step": 1, "event_type": "b"})

    events = load_runtime_events(event_dir)

    assert [event["event_type"] for event in events] == ["a", "b"]


def test_renderer_logs_stage_node_edge_and_dmatrix(tmp_path: Path) -> None:
    event_dir = tmp_path / "rerun_viz"
    artifact_dir = event_dir / "artifacts" / "step_001"
    artifact_dir.mkdir(parents=True)
    dmatrix_path = artifact_dir / "dmatrix.png"
    dmatrix_path.write_bytes(b"fake-png")
    renderer = MapMergeRuntimeRerunRenderer(event_dir)
    rr = FakeRerun()

    renderer.log_event(
        rr,
        {
            "demo_step": 1,
            "merge_step": 0,
            "keyframe_id": None,
            "event_type": "stage_annotation",
            "payload": {"display_text": "Stage 1 / 8\nLoad Reference Submap 0"},
            "artifacts": {},
        },
    )
    renderer.log_event(
        rr,
        {
            "demo_step": 2,
            "merge_step": 0,
            "submap_id": 0,
            "keyframe_id": 3,
            "event_type": "vio_node_observed",
            "payload": {"node_id": 3, "position": [1.0, 2.0, 3.0]},
            "artifacts": {},
        },
    )
    renderer.log_event(
        rr,
        {
            "demo_step": 3,
            "merge_step": 0,
            "submap_id": 0,
            "keyframe_id": 4,
            "event_type": "covis_edge_observed",
            "payload": {
                "edge_type": "covis",
                "nodeAid": 3,
                "nodeBid": 4,
                "position_a": [1.0, 2.0, 3.0],
                "position_b": [2.0, 3.0, 4.0],
                "weight": 0.5,
            },
            "artifacts": {},
        },
    )
    renderer.log_event(
        rr,
        {
            "demo_step": 4,
            "merge_step": 1,
            "keyframe_id": None,
            "event_type": "dmatrix_computed",
            "payload": {},
            "artifacts": {"dmatrix_png": "artifacts/step_001/dmatrix.png"},
        },
    )

    assert ("/status/stage_summary", ("text", "Stage 1 / 8\nLoad Reference Submap 0")) in rr.logged
    assert any(path == "/world/submaps/0/nodes/000003" for path, _ in rr.logged)
    assert any(path == "/world/submaps/0/edges/covis/000003_000004" for path, _ in rr.logged)
    assert (
        "/evidence/dmatrix",
        ("encoded_image", {"path": dmatrix_path, "media_type": "image/png"}),
    ) in rr.logged
    assert ("demo_step", 4) in rr.times
