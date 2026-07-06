from pathlib import Path
from visualization.map_merge_offline_to_events import detect_merge_dirs


def test_detect_merge_dirs_finds_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "merge_0").mkdir()
    (tmp_path / "merge_0_1_2").mkdir()
    (tmp_path / "merge_0_1").mkdir()
    (tmp_path / "merge_finalmap").write_text("link")
    (tmp_path / "not_merge").mkdir()

    result = detect_merge_dirs(tmp_path)
    names = [p.name for p in result]
    assert names == ["merge_0", "merge_0_1", "merge_0_1_2"]


def test_detect_merge_dirs_empty_when_no_merge_dirs(tmp_path: Path) -> None:
    (tmp_path / "other").mkdir()
    assert detect_merge_dirs(tmp_path) == []


from visualization.map_merge_offline_to_events import (
    PoseEntry,
    EdgeEntry,
    IntrinsicsEntry,
    load_poses,
    load_edges,
    load_intrinsics,
)


def test_load_poses_parses_w2c_wxyz_and_converts_to_c2w(tmp_path: Path) -> None:
    poses_file = tmp_path / "poses.txt"
    poses_file.write_text(
        "seq/000000.color.jpg 1.0 0.0 0.0 0.0 1.0 2.0 3.0\n"
        "seq/000001.color.jpg 0.0 0.0 0.0 1.0 4.0 5.0 6.0\n"
    )
    result = load_poses(poses_file)
    assert len(result) == 2
    assert result[0].img_name == "seq/000000.color.jpg"
    assert result[0].quat_xyzw == [0.0, 0.0, 0.0, 1.0]
    assert result[0].position == [-1.0, -2.0, -3.0]
    assert result[1].img_name == "seq/000001.color.jpg"
    assert result[1].quat_xyzw == [0.0, 0.0, -1.0, 0.0]
    assert result[1].position == [4.0, 5.0, -6.0]


def test_load_edges_parses_src_dst_weight(tmp_path: Path) -> None:
    edges_file = tmp_path / "edges_covis.txt"
    edges_file.write_text("0 1 0.310245\n1 2 0.402091\n")
    result = load_edges(edges_file)
    assert len(result) == 2
    assert result[0].src == 0
    assert result[0].dst == 1
    assert abs(result[0].weight - 0.310245) < 1e-6


def test_load_intrinsics_parses_fx_fy_cx_cy_w_h(tmp_path: Path) -> None:
    intr_file = tmp_path / "intrinsics.txt"
    intr_file.write_text(
        "seq/000000.color.jpg 444.49 444.49 511.5 287.5 1024 576\n"
    )
    result = load_intrinsics(intr_file)
    assert "seq/000000.color.jpg" in result
    entry = result["seq/000000.color.jpg"]
    assert entry.K[0][0] == 444.49
    assert entry.K[1][2] == 287.5
    assert entry.img_size == [1024, 576]


from visualization.map_merge_offline_to_events import identify_new_nodes


def test_identify_new_nodes_returns_indices_not_in_prev(tmp_path: Path) -> None:
    prev_names = ["seq/000000.color.jpg", "seq/000001.color.jpg", "seq/000002.color.jpg"]
    curr_names = ["seq/000000.color.jpg", "seq/000001.color.jpg",
                  "seq/000002.color.jpg", "seq/000003.color.jpg", "seq/000004.color.jpg"]
    result = identify_new_nodes(prev_names, curr_names)
    assert result == [3, 4]


def test_identify_new_nodes_empty_when_no_new(tmp_path: Path) -> None:
    names = ["seq/000000.color.jpg", "seq/000001.color.jpg"]
    assert identify_new_nodes(names, names) == []


def test_identify_new_nodes_all_new_when_prev_empty(tmp_path: Path) -> None:
    curr = ["seq/000000.color.jpg", "seq/000001.color.jpg"]
    assert identify_new_nodes([], curr) == [0, 1]


from visualization.map_merge_offline_to_events import find_cross_submap_edges


def test_find_cross_submap_edges_finds_non_consecutive():
    prev = {"odom": [EdgeEntry(0, 1, 0.5)], "covis": [], "trav": []}
    curr = {
        "odom": [EdgeEntry(0, 1, 0.5), EdgeEntry(57, 81, 1.4)],
        "covis": [],
        "trav": [],
    }
    result = find_cross_submap_edges(prev, curr)
    assert len(result) == 1
    assert result[0] == (57, 81, "odom")


def test_find_cross_submap_edges_skips_consecutive():
    prev = {"odom": [EdgeEntry(0, 1, 0.5)], "covis": [], "trav": []}
    curr = {
        "odom": [EdgeEntry(0, 1, 0.5), EdgeEntry(1, 2, 0.3)],
        "covis": [],
        "trav": [],
    }
    result = find_cross_submap_edges(prev, curr)
    assert len(result) == 0


def test_find_cross_submap_edges_deduplicates():
    prev = {"odom": [], "covis": [], "trav": []}
    curr = {
        "odom": [EdgeEntry(57, 81, 1.4)],
        "covis": [EdgeEntry(57, 81, 0.3)],
        "trav": [],
    }
    result = find_cross_submap_edges(prev, curr)
    assert len(result) == 1


from visualization.map_merge_offline_to_events import generate_events


def _make_fake_merge_dir(base: Path, name: str, num_poses: int, start_idx: int = 0) -> Path:
    merge_dir = base / name
    merge_dir.mkdir()
    seq_dir = merge_dir / "seq"
    seq_dir.mkdir()

    pose_lines = []
    intr_lines = []
    for i in range(start_idx, start_idx + num_poses):
        img_name = f"seq/{i:06d}.color.jpg"
        pose_lines.append(f"{img_name} 1.0 0.0 0.0 0.0 {float(i)} 0.0 0.0")
        intr_lines.append(f"{img_name} 444.0 444.0 511.5 287.5 1024 576")
        (seq_dir / f"{i:06d}.color.jpg").write_bytes(b"fake-jpg")
    (merge_dir / "poses.txt").write_text("\n".join(pose_lines) + "\n")
    (merge_dir / "intrinsics.txt").write_text("\n".join(intr_lines) + "\n")

    edge_lines = []
    for i in range(num_poses - 1):
        edge_lines.append(f"{start_idx + i} {start_idx + i + 1} 0.5")
    for et in ("odom", "covis", "trav"):
        (merge_dir / f"edges_{et}.txt").write_text("\n".join(edge_lines) + "\n")

    return merge_dir


def _make_fake_raw_dir(base: Path, name: str, num_poses: int) -> Path:
    raw_dir = base / name
    raw_dir.mkdir()
    seq_dir = raw_dir / "seq"
    seq_dir.mkdir()

    pose_lines = []
    intr_lines = []
    for i in range(num_poses):
        img_name = f"seq/{i:06d}.color.jpg"
        pose_lines.append(f"{img_name} 1.0 0.0 0.0 0.0 {float(i) + 100.0} 0.0 0.0")
        intr_lines.append(f"{img_name} 444.0 444.0 511.5 287.5 1024 576")
        (seq_dir / f"{i:06d}.color.jpg").write_bytes(b"fake-jpg")
    (raw_dir / "poses.txt").write_text("\n".join(pose_lines) + "\n")
    (raw_dir / "intrinsics.txt").write_text("\n".join(intr_lines) + "\n")

    edge_lines = [f"{i} {i + 1} 0.5" for i in range(num_poses - 1)]
    for et in ("odom", "covis", "trav"):
        (raw_dir / f"edges_{et}.txt").write_text("\n".join(edge_lines) + "\n")

    return raw_dir


def test_generate_events_produces_valid_sequence(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _make_fake_merge_dir(results_dir, "merge_0", num_poses=3, start_idx=0)
    _make_fake_merge_dir(results_dir, "merge_0_1", num_poses=5, start_idx=0)

    events = generate_events(results_dir, tmp_path / "output")

    types = [e["event_type"] for e in events]
    assert "stage_annotation" in types
    assert "vio_node_observed" in types
    assert "odom_edge_observed" in types
    assert "covis_edge_observed" in types
    assert "trav_edge_observed" in types
    assert "map_committed" in types
    assert "dmatrix_computed" not in types

    assert events[0]["event_type"] == "stage_annotation"
    assert events[0]["payload"]["title"] == "Load Reference Map"

    vio_step0 = [e for e in events
                 if e["event_type"] == "vio_node_observed" and e["merge_step"] == 0]
    assert len(vio_step0) == 3

    vio_step1 = [e for e in events
                 if e["event_type"] == "vio_node_observed" and e["merge_step"] == 1]
    assert len(vio_step1) == 2

    committed = [e for e in events if e["event_type"] == "map_committed"]
    assert len(committed) == 2

    assert events[-1]["event_type"] == "stage_annotation"
    assert events[-1]["payload"]["title"] == "Finish Map Merging"


def test_generate_events_with_raw_data_plots_raw_submap(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _make_fake_merge_dir(results_dir, "merge_0", num_poses=3, start_idx=0)
    _make_fake_merge_dir(results_dir, "merge_0_1", num_poses=5, start_idx=0)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_fake_raw_dir(raw_dir, "1", num_poses=2)

    events = generate_events(results_dir, tmp_path / "output", raw_data_dir=raw_dir)

    # Raw submap cameras should appear as vio_node_observed with local IDs
    vio_step1 = [e for e in events
                 if e["event_type"] == "vio_node_observed" and e["merge_step"] == 1]
    # 2 raw + 2 new merged = 4 total
    assert len(vio_step1) == 4
    # Raw cameras have local IDs (0, 1)
    raw_nodes = [e for e in vio_step1 if e["keyframe_id"] in (0, 1)]
    assert len(raw_nodes) == 2


def test_generate_events_with_cross_submap_edges(tmp_path: Path) -> None:
    """Green edges connect final map nodes to raw submap nodes (local IDs, before map_committed)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _make_fake_merge_dir(results_dir, "merge_0", num_poses=3, start_idx=0)
    # merge_0_1: 5 poses (0-4), cross-submap edge (1, 4) — non-consecutive
    merge_01 = _make_fake_merge_dir(results_dir, "merge_0_1", num_poses=5, start_idx=0)
    with (merge_01 / "edges_covis.txt").open("a") as f:
        f.write("1 4 0.9\n")

    # Raw submap 1 with 2 poses (local IDs 0-1)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_fake_raw_dir(raw_dir, "1", num_poses=2)

    events = generate_events(results_dir, tmp_path / "output", raw_data_dir=raw_dir)

    metric_edges = [e for e in events if e["event_type"] == "metric_edge_added"]
    assert len(metric_edges) == 1
    # db_node_id = 1 (final map node, global ID)
    assert metric_edges[0]["payload"]["db_node_id"] == 1
    # query_node_id = 1 (raw submap LOCAL ID = 4 - prev_pose_count(3) = 1)
    assert metric_edges[0]["payload"]["query_node_id"] == 1
    # submap_id = 1 (raw submap's ID, not 0)
    assert metric_edges[0]["submap_id"] == 1

    # Metric edges should be emitted BEFORE map_committed
    metric_step = metric_edges[0]["demo_step"]
    committed_steps = [e["demo_step"] for e in events
                       if e["event_type"] == "map_committed" and e["merge_step"] == 1]
    assert len(committed_steps) == 1
    assert metric_step < committed_steps[0]


def test_generate_events_demo_steps_are_monotonic(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _make_fake_merge_dir(results_dir, "merge_0", num_poses=2, start_idx=0)

    events = generate_events(results_dir, tmp_path / "output")
    demo_steps = [e["demo_step"] for e in events]
    assert demo_steps == sorted(demo_steps)
    assert demo_steps[0] == 0


import json
from visualization.map_merge_offline_to_events import write_events, parse_args


def test_write_events_creates_jsonl(tmp_path: Path) -> None:
    event_dir = tmp_path / "rerun_viz"
    events = [
        {"demo_step": 0, "event_type": "stage_annotation", "payload": {}, "artifacts": {}},
        {"demo_step": 1, "event_type": "vio_node_observed", "payload": {}, "artifacts": {}},
    ]
    write_events(events, event_dir)
    jsonl_path = event_dir / "demo_events.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["event_type"] == "stage_annotation"
    assert parsed[1]["event_type"] == "vio_node_observed"


def test_parse_args_accepts_required_args() -> None:
    args = parse_args(["--results-dir", "/tmp/results", "--output-dir", "/tmp/output"])
    assert args.results_dir == Path("/tmp/results")
    assert args.output_dir == Path("/tmp/output")


def test_parse_args_accepts_render_flag() -> None:
    args = parse_args([
        "--results-dir", "/tmp/results",
        "--output-dir", "/tmp/output",
        "--render",
        "--rerun-output", "/tmp/out.rrd",
    ])
    assert args.render is True
    assert args.rerun_output == Path("/tmp/out.rrd")


def test_parse_args_accepts_raw_data_dir() -> None:
    args = parse_args([
        "--results-dir", "/tmp/results",
        "--output-dir", "/tmp/output",
        "--raw-data-dir", "/tmp/raw",
    ])
    assert args.raw_data_dir == Path("/tmp/raw")
