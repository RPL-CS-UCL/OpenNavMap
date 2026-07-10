from pathlib import Path

from visualization.map_merge_result_replay import MapMergeResultReplay


def _write_merge_dir(root: Path, name: str) -> Path:
    merge_dir = root / name
    preds_dir = merge_dir / "preds"
    preds_dir.mkdir(parents=True)
    (merge_dir / "poses.txt").write_text(
        "seq/000000.color.jpg 1 0 0 0 0 0 0\n"
        "seq/000001.color.jpg 1 0 0 0 -1 0 0\n",
        encoding="utf-8",
    )
    (merge_dir / "timestamps.txt").write_text(
        "seq/000000.color.jpg 1.0\nseq/000001.color.jpg 2.0\n",
        encoding="utf-8",
    )
    seq_dir = merge_dir / "seq"
    seq_dir.mkdir()
    (seq_dir / "000000.color.jpg").write_bytes(b"fake-jpeg-0")
    (seq_dir / "000001.color.jpg").write_bytes(b"fake-jpeg-1")
    (merge_dir / "edges_odom.txt").write_text("0 1 1.0\n", encoding="utf-8")
    (merge_dir / "edges_covis.txt").write_text("0 1 0.8\n", encoding="utf-8")
    (merge_dir / "edges_trav.txt").write_text("0 1 1.0\n", encoding="utf-8")
    (preds_dir / "difference_matrix_fitting_10.jpg").write_bytes(b"fake")
    (preds_dir / "initial_pose_graph.g2o").write_text(
        "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n", encoding="utf-8"
    )
    return merge_dir


def test_discovers_merge_chain_in_step_order(tmp_path: Path) -> None:
    _write_merge_dir(tmp_path, "merge_0_1")
    _write_merge_dir(tmp_path, "merge_0")

    replay = MapMergeResultReplay(tmp_path)

    assert [step.name for step in replay.discover_steps()] == ["merge_0", "merge_0_1"]


def test_build_events_emits_readonly_warnings_and_artifacts(tmp_path: Path) -> None:
    _write_merge_dir(tmp_path, "merge_0")

    replay = MapMergeResultReplay(tmp_path)
    events = replay.build_events()

    event_types = [event.event_type for event in events]
    assert "submap_merged" in event_types
    assert "dmatrix_ready" in event_types
    assert any("read-only" in event.payload.get("message", "") for event in events)


def test_build_events_counts_edges_and_positions(tmp_path: Path) -> None:
    _write_merge_dir(tmp_path, "merge_0")

    replay = MapMergeResultReplay(tmp_path)

    merged_event = next(
        event for event in replay.build_events() if event.event_type == "submap_merged"
    )

    assert merged_event.payload["num_poses"] == 2
    assert merged_event.payload["edge_counts"] == {"odom": 1, "covis": 1, "trav": 1}
    assert merged_event.payload["positions"].shape == (2, 3)


def test_discovers_single_result_directory_with_poses(tmp_path: Path) -> None:
    merge_dir = _write_merge_dir(tmp_path, "custom_output")

    replay = MapMergeResultReplay(merge_dir)

    assert [step.path for step in replay.discover_steps()] == [merge_dir]


def test_build_events_emits_incremental_nodes_with_current_images(tmp_path: Path) -> None:
    _write_merge_dir(tmp_path, "merge_0")

    replay = MapMergeResultReplay(tmp_path)
    node_events = [
        event for event in replay.build_events() if event.event_type == "node_observed"
    ]

    assert [event.payload["node_id"] for event in node_events] == [0, 1]
    assert [event.keyframe_id for event in node_events] == [0, 1]
    assert node_events[0].artifact_refs.current_image == (
        tmp_path / "merge_0" / "seq" / "000000.color.jpg"
    )


def test_build_events_emits_incremental_odom_edges_after_endpoint(tmp_path: Path) -> None:
    _write_merge_dir(tmp_path, "merge_0")

    replay = MapMergeResultReplay(tmp_path)
    edge_events = [
        event
        for event in replay.build_events()
        if event.event_type == "intrasubmap_edge_observed"
    ]

    assert len(edge_events) == 1
    assert edge_events[0].payload["edge_type"] == "odom"
    assert edge_events[0].payload["node_a"] == 0
    assert edge_events[0].payload["node_b"] == 1
    assert edge_events[0].keyframe_id == 1


def test_build_events_detects_intersubmap_edges_from_edge_diff(tmp_path: Path) -> None:
    _write_merge_dir(tmp_path, "merge_0")
    merge_01 = _write_merge_dir(tmp_path, "merge_0_1")
    with (merge_01 / "poses.txt").open("a", encoding="utf-8") as pose_file:
        pose_file.write("seq/000002.color.jpg 1 0 0 0 -2 0 0\n")
    (merge_01 / "seq" / "000002.color.jpg").write_bytes(b"fake-jpeg-2")
    with (merge_01 / "edges_covis.txt").open("a", encoding="utf-8") as edge_file:
        edge_file.write("0 2 0.9\n")

    replay = MapMergeResultReplay(tmp_path)
    inter_events = [
        event
        for event in replay.build_events()
        if event.event_type == "intersubmap_edge_observed"
    ]

    assert len(inter_events) == 1
    assert inter_events[0].payload["edge_type"] == "covis"
    assert inter_events[0].payload["node_a"] == 0
    assert inter_events[0].payload["node_b"] == 2
    assert inter_events[0].keyframe_id == 100002
