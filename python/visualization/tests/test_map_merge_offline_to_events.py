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


import numpy as np
from visualization.map_merge_offline_to_events import (
    PoseEntry,
    EdgeEntry,
    IntrinsicsEntry,
    load_poses,
    load_edges,
    load_intrinsics,
    load_descriptors,
)


def test_load_poses_parses_quat_xyzw_and_translation(tmp_path: Path) -> None:
    poses_file = tmp_path / "poses.txt"
    poses_file.write_text(
        "seq/000000.color.jpg 0.0 0.0 0.0 1.0 1.0 2.0 3.0\n"
        "seq/000001.color.jpg 0.1 0.2 0.3 0.4 4.0 5.0 6.0\n"
    )
    result = load_poses(poses_file)
    assert len(result) == 2
    assert result[0].img_name == "seq/000000.color.jpg"
    assert result[0].quat_xyzw == [0.0, 0.0, 0.0, 1.0]
    assert result[0].position == [1.0, 2.0, 3.0]
    assert result[1].img_name == "seq/000001.color.jpg"
    assert result[1].quat_xyzw == [0.1, 0.2, 0.3, 0.4]
    assert result[1].position == [4.0, 5.0, 6.0]


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


def test_load_descriptors_parses_256_dim(tmp_path: Path) -> None:
    desc_file = tmp_path / "database_descriptors.txt"
    vals = " ".join(str(float(i)) for i in range(256))
    desc_file.write_text(f"seq/000000.color.jpg {vals}\n")
    result = load_descriptors(desc_file)
    assert "seq/000000.color.jpg" in result
    assert result["seq/000000.color.jpg"].shape == (256,)
    assert result["seq/000000.color.jpg"][0] == 0.0
    assert result["seq/000000.color.jpg"][255] == 255.0


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


from visualization.map_merge_offline_to_events import compute_dmatrix


def test_compute_dmatrix_shape_and_values() -> None:
    ref = {
        "a.jpg": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "b.jpg": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    query = {
        "c.jpg": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "d.jpg": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    result = compute_dmatrix(ref, query)
    assert result.shape == (2, 2)
    assert abs(result[0, 0] - 1.0) < 1e-5  # a-c identical
    assert abs(result[0, 1] - 0.0) < 1e-5  # a-d orthogonal
    assert abs(result[1, 0] - 0.0) < 1e-5  # b-c orthogonal
    assert abs(result[1, 1] - 0.0) < 1e-5  # b-d orthogonal


def test_compute_dmatrix_empty_query_returns_empty() -> None:
    ref = {"a.jpg": np.array([1.0, 0.0], dtype=np.float32)}
    result = compute_dmatrix(ref, {})
    assert result.shape == (1, 0)


from visualization.map_merge_offline_to_events import plot_dmatrix


def test_plot_dmatrix_creates_png(tmp_path: Path) -> None:
    dmatrix = np.random.rand(10, 5).astype(np.float32)
    output_path = tmp_path / "dmatrix.png"
    plot_dmatrix(dmatrix, output_path, ref_label="Ref", query_label="Query")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


from visualization.map_merge_offline_to_events import generate_events


def _make_fake_merge_dir(base: Path, name: str, num_poses: int, start_idx: int = 0) -> Path:
    """Create a fake merge directory with minimal data files."""
    merge_dir = base / name
    merge_dir.mkdir()
    seq_dir = merge_dir / "seq"
    seq_dir.mkdir()

    pose_lines = []
    intr_lines = []
    desc_lines = []
    for i in range(start_idx, start_idx + num_poses):
        img_name = f"seq/{i:06d}.color.jpg"
        pose_lines.append(f"{img_name} 0.0 0.0 0.0 1.0 {float(i)} 0.0 0.0")
        intr_lines.append(f"{img_name} 444.0 444.0 511.5 287.5 1024 576")
        (seq_dir / f"{i:06d}.color.jpg").write_bytes(b"fake-jpg")
        desc_vals = " ".join(str(float(j) / 256.0) for j in range(256))
        desc_lines.append(f"{img_name} {desc_vals}")
    (merge_dir / "poses.txt").write_text("\n".join(pose_lines) + "\n")
    (merge_dir / "intrinsics.txt").write_text("\n".join(intr_lines) + "\n")
    (merge_dir / "database_descriptors.txt").write_text("\n".join(desc_lines) + "\n")

    edge_lines = []
    for i in range(num_poses - 1):
        edge_lines.append(f"{start_idx + i} {start_idx + i + 1} 0.5")
    for et in ("odom", "covis", "trav"):
        (merge_dir / f"edges_{et}.txt").write_text("\n".join(edge_lines) + "\n")

    return merge_dir


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
    assert "dmatrix_computed" in types
    assert "map_committed" in types

    assert events[0]["event_type"] == "stage_annotation"
    assert events[0]["payload"]["title"] == "Load Reference Map"

    vio_step0 = [e for e in events
                 if e["event_type"] == "vio_node_observed" and e["merge_step"] == 0]
    assert len(vio_step0) == 3

    vio_step1 = [e for e in events
                 if e["event_type"] == "vio_node_observed" and e["merge_step"] == 1]
    assert len(vio_step1) == 2

    dmatrix_events = [e for e in events if e["event_type"] == "dmatrix_computed"]
    assert len(dmatrix_events) == 1
    assert dmatrix_events[0]["merge_step"] == 1

    committed = [e for e in events if e["event_type"] == "map_committed"]
    assert len(committed) == 2

    assert events[-1]["event_type"] == "stage_annotation"
    assert events[-1]["payload"]["title"] == "Finish Map Merging"


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
