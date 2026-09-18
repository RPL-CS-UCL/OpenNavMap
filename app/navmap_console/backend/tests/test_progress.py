from pathlib import Path


def test_line_tailer_handles_partial_lines_and_carriage_returns():
    from navmap_console.jobs.progress import LineTailer

    t = LineTailer()
    assert t.feed(b"abc\ndef") == ["abc"]
    assert t.feed(b"g\n 10%\r 50%\r100%\nlast") == ["defg", "100%"]
    assert t.flush() == "last" and t.flush() is None


def test_parse_progress_anchors():
    from navmap_console.jobs.progress import parse_progress

    assert parse_progress("INFO:root:--- Merging submap 3: 002 ---") == {"kind": "step_start", "index": 3, "sid": "002"}
    assert parse_progress("D_all shape: (100, 5000)") == {"kind": "stage", "stage_index": 4}
    assert parse_progress("PGO: initial error: 12.500") == {"kind": "pgo_initial", "value": 12.5}
    assert parse_progress("PGO: final error: 0.750") == {"kind": "pgo_final", "value": 0.75}
    assert parse_progress("Saved intermediate result: /x/merge_003_002") == {"kind": "step_saved", "dir": "/x/merge_003_002"}
    done = parse_progress("STEP_DONE index=3 sid=002 dir=/x/merge_003_002 id_offset=120 odom_nodes=180 covis_nodes=150 components=1 registry=14")
    assert done == {"kind": "step_done", "index": 3, "sid": "002", "dir": "/x/merge_003_002", "id_offset": 120,
                    "odom_nodes": 180, "covis_nodes": 150, "components": 1, "registry": 14}
    assert parse_progress("random noise") is None


def test_classify_crash():
    from navmap_console.jobs.progress import classify_crash

    assert classify_crash(0, []) is None
    assert classify_crash(-11, []) == "segfault"
    assert classify_crash(1, ["...", "RuntimeError: CUDA out of memory. Tried to allocate"]) == "cuda_oom"
    assert classify_crash(1, ["TypeError: 'range_iterator' object is not callable"]) == "bit_flip"
    assert classify_crash(-9, []) == "oom"
    assert classify_crash(137, ["Killed"]) == "oom"
    assert classify_crash(-15, []) == "terminated"
    assert classify_crash(1, ["Traceback", "ValueError: x"]) == "error"


def test_read_log_lines(tmp_path: Path):
    from navmap_console.jobs.progress import read_log_lines

    p = tmp_path / "j.log"
    p.write_bytes(b"one\nstep 1%\rstep 99%\nthree\xff\n")
    assert read_log_lines(p) == ["one", "step 99%", "three�"]
    assert read_log_lines(tmp_path / "missing.log") == []
