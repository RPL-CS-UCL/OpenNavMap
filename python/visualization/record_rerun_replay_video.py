from __future__ import annotations

import argparse
import shlex
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a Rerun .rrd replay window to an MP4 video."
    )
    parser.add_argument("--rrd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--startup-sec", type=float, default=3.0)
    parser.add_argument("--backend", type=str, default="web", choices=["web", "native"])
    parser.add_argument("--web-viewer-port", type=int, default=19090)
    parser.add_argument("--ws-server-port", type=int, default=19877)
    parser.add_argument("--autoplay", dest="autoplay", action="store_true", default=True)
    parser.add_argument("--no-autoplay", dest="autoplay", action="store_false")
    parser.add_argument("--renderer", type=str, default="gl", choices=["gl", "vulkan"])
    return parser.parse_args(argv)


def build_web_viewer_command(
    rrd_path: Path,
    web_viewer_port: int,
    ws_server_port: int,
) -> List[str]:
    return [
        "rerun",
        str(rrd_path),
        "--web-viewer",
        "--web-viewer-port",
        str(web_viewer_port),
        "--ws-server-port",
        str(ws_server_port),
        "--hide-welcome-screen",
        "--renderer",
        "webgl",
    ]


def build_record_command(
    rrd_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    duration_sec: float,
    startup_sec: float,
    renderer: str = "gl",
) -> List[str]:
    rrd = shlex.quote(str(rrd_path))
    output = shlex.quote(str(output_path))
    window_size = f"{width}x{height}"
    shell = (
        "set -e; "
        f"rerun {rrd} --hide-welcome-screen --window-size {window_size} "
        f"--renderer {renderer} >/tmp/rerun-record.log 2>&1 & "
        "viewer_pid=$!; "
        f"sleep {startup_sec}; "
        f"ffmpeg -y -f x11grab -video_size {window_size} -framerate {fps} "
        f"-t {duration_sec} -i $DISPLAY -c:v libx264 -pix_fmt yuv420p {output}; "
        "kill $viewer_pid >/dev/null 2>&1 || true"
    )
    return ["xvfb-run", "-a", "bash", "-lc", shell]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.rrd.exists():
        raise FileNotFoundError(f"Rerun recording not found: {args.rrd}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "native":
        command = build_record_command(
            rrd_path=args.rrd,
            output_path=args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration_sec=args.duration_sec,
            startup_sec=args.startup_sec,
            renderer=args.renderer,
        )
        subprocess.run(command, check=True)
        return

    record_web_viewer(
        rrd_path=args.rrd,
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_sec=args.duration_sec,
        web_viewer_port=args.web_viewer_port,
        ws_server_port=args.ws_server_port,
        autoplay=args.autoplay,
    )


def record_web_viewer(
    rrd_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    duration_sec: float,
    web_viewer_port: int,
    ws_server_port: int,
    autoplay: bool,
) -> None:
    from playwright.sync_api import sync_playwright

    server = subprocess.Popen(
        build_web_viewer_command(rrd_path, web_viewer_port, ws_server_port),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://127.0.0.1:{web_viewer_port}"
        _wait_for_http(url)
        with tempfile.TemporaryDirectory(prefix="rerun-video-") as tmp_dir:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=playwright.chromium.executable_path,
                    args=["--no-sandbox"],
                )
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    record_video_dir=tmp_dir,
                    record_video_size={"width": width, "height": height},
                )
                page = context.new_page()
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(1000)
                if autoplay:
                    page.mouse.click(width / 2, height / 2)
                    page.keyboard.press("Space")
                page.wait_for_timeout(duration_sec * 1000)
                context.close()
                browser.close()
            webm_files = sorted(Path(tmp_dir).glob("*.webm"))
            if not webm_files:
                raise RuntimeError("Playwright did not produce a video file")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(webm_files[0]),
                    "-r",
                    str(fps),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(output_path),
                ],
                check=True,
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


def _wait_for_http(url: str, timeout_sec: float = 20.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for Rerun web viewer: {url}")


if __name__ == "__main__":
    main()
