from pathlib import Path

from visualization import record_rerun_replay_video


def test_parse_args_defaults_to_1080p_mp4() -> None:
    args = record_rerun_replay_video.parse_args(
        ["--rrd", "/tmp/in.rrd", "--output", "/tmp/out.mp4"]
    )

    assert args.rrd == Path("/tmp/in.rrd")
    assert args.output == Path("/tmp/out.mp4")
    assert args.width == 1920
    assert args.height == 1080
    assert args.fps == 30
    assert args.duration_sec == 30.0
    assert args.backend == "web"
    assert args.autoplay is True


def test_build_web_viewer_command_uses_fixed_ports(tmp_path: Path) -> None:
    command = record_rerun_replay_video.build_web_viewer_command(
        rrd_path=tmp_path / "input.rrd",
        web_viewer_port=19090,
        ws_server_port=19877,
    )

    assert command == [
        "rerun",
        str(tmp_path / "input.rrd"),
        "--web-viewer",
        "--web-viewer-port",
        "19090",
        "--ws-server-port",
        "19877",
        "--hide-welcome-screen",
        "--renderer",
        "webgl",
    ]


def test_build_record_command_uses_xvfb_rerun_and_ffmpeg(tmp_path: Path) -> None:
    rrd_path = tmp_path / "input.rrd"
    output_path = tmp_path / "output.mp4"
    command = record_rerun_replay_video.build_record_command(
        rrd_path=rrd_path,
        output_path=output_path,
        width=1280,
        height=720,
        fps=24,
        duration_sec=5.0,
        startup_sec=1.0,
    )

    assert command[:3] == ["xvfb-run", "-a", "bash"]
    shell = command[-1]
    assert f"rerun {rrd_path}" in shell
    assert "--window-size 1280x720" in shell
    assert "ffmpeg" in shell
    assert "-video_size 1280x720" in shell
    assert "-framerate 24" in shell
    assert "-t 5.0" in shell
    assert str(output_path) in shell
