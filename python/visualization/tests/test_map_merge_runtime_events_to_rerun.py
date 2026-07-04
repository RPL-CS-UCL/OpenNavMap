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
    def Transform3D(*args, **kwargs):
        return ("transform3d", args, kwargs)

    @staticmethod
    def Pinhole(*args, **kwargs):
        return ("pinhole", args, kwargs)

    @staticmethod
    def ImageEncoded(**kwargs):
        return ("image_encoded", kwargs)

    @staticmethod
    def LineStrips3D(*args, **kwargs):
        return ("lines3d", args, kwargs)

    @staticmethod
    def Arrows3D(*args, **kwargs):
        return ("arrows3d", args, kwargs)

    @staticmethod
    def Quaternion(**kwargs):
        return ("quaternion", kwargs)

    class ViewCoordinates:
        RDF = "RDF"

    def set_time_sequence(self, timeline, value) -> None:
        self.times.append((timeline, value))

    def log(self, entity_path, archetype) -> None:
        self.logged.append((entity_path, archetype))


def _write_event(event_dir: Path, event: dict) -> None:
    with (event_dir / "demo_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def test_parse_args_accepts_event_dir_and_rerun_output() -> None:
    args = parse_args(["--event-dir", "/tmp/rerun_viz", "--rerun-output", "/tmp/out.rrd"])
    assert args.event_dir == Path("/tmp/rerun_viz")
    assert args.rerun_output == Path("/tmp/out.rrd")


def test_parse_args_accepts_render_trace() -> None:
    args = parse_args([
        "--event-dir", "/tmp/rerun_viz",
        "--rerun-output", "/tmp/out.rrd",
        "--render-trace", "/tmp/trace.jsonl",
    ])
    assert args.render_trace == Path("/tmp/trace.jsonl")


def test_load_runtime_events_reads_jsonl_in_order(tmp_path: Path) -> None:
    event_dir = tmp_path / "rerun_viz"
    event_dir.mkdir()
    _write_event(event_dir, {"demo_step": 0, "event_type": "a"})
    _write_event(event_dir, {"demo_step": 1, "event_type": "b"})
    events = load_runtime_events(event_dir)
    assert [e["event_type"] for e in events] == ["a", "b"]


def test_renderer_logs_stage_keyframe_and_edges(tmp_path: Path) -> None:
    event_dir = tmp_path / "rerun_viz"
    event_dir.mkdir()
    img_path = event_dir / "test.jpg"
    img_path.write_bytes(b"fake-jpg")

    renderer = MapMergeRuntimeRerunRenderer(event_dir)
    rr = FakeRerun()

    renderer.log_event(rr, {
        "demo_step": 0, "merge_step": 0, "keyframe_id": None,
        "submap_id": 0,
        "event_type": "stage_annotation",
        "payload": {"display_text": "Stage 1 / 8\nLoad Reference Submap 0"},
        "artifacts": {},
    })
    renderer.log_event(rr, {
        "demo_step": 1, "merge_step": 0, "submap_id": 0, "keyframe_id": 3,
        "event_type": "vio_node_observed",
        "payload": {
            "node_id": 3,
            "position": [1.0, 2.0, 3.0],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "K": [[100.0, 0.0, 50.0], [0.0, 110.0, 60.0], [0.0, 0.0, 1.0]],
            "img_size": [512, 288],
            "rgb_img_path": str(img_path),
        },
        "artifacts": {},
    })
    renderer.log_event(rr, {
        "demo_step": 2, "merge_step": 1, "submap_id": 1, "keyframe_id": 5,
        "event_type": "vio_node_observed",
        "payload": {
            "node_id": 5,
            "position": [4.0, 5.0, 6.0],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "K": [[100.0, 0.0, 50.0], [0.0, 110.0, 60.0], [0.0, 0.0, 1.0]],
            "img_size": [512, 288],
            "rgb_img_path": str(img_path),
        },
        "artifacts": {},
    })
    renderer.log_event(rr, {
        "demo_step": 3, "merge_step": 0, "submap_id": 0, "keyframe_id": 4,
        "event_type": "covis_edge_observed",
        "payload": {
            "edge_type": "covis",
            "nodeAid": 3, "nodeBid": 4,
            "position_a": [1.0, 2.0, 3.0],
            "position_b": [2.0, 3.0, 4.0],
            "weight": 0.5,
        },
        "artifacts": {},
    })
    renderer.log_event(rr, {
        "demo_step": 4, "merge_step": 1, "keyframe_id": None,
        "event_type": "dmatrix_computed",
        "payload": {},
        "artifacts": {},
    })

    assert ("/status/stage_summary", ("text", "Stage 1 / 8\nLoad Reference Submap 0")) in rr.logged

    assert any(p == "sfm/cameras/ref/3" and v[0] == "transform3d" for p, v in rr.logged)
    assert any(p == "sfm/cameras/ref/3/image" and v[0] == "pinhole" for p, v in rr.logged)
    assert any(p == "sfm/cameras/ref/3/image" and v[0] == "image_encoded" for p, v in rr.logged)

    assert any(p == "sfm/cameras/query/5" and v[0] == "transform3d" for p, v in rr.logged)
    assert any(p == "sfm/cameras/query/5/image" and v[0] == "pinhole" for p, v in rr.logged)

    assert any(p == "evidence/current_keyframe_image" and v[0] == "image_encoded" for p, v in rr.logged)

    assert any(p == "sfm/edges/ref/covis/3_4" and v[0] == "lines3d" for p, v in rr.logged)

    assert not any(p == "/evidence/dmatrix" for p, v in rr.logged)

    assert ("demo_step", 3) in rr.times
