from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from visualization.map_merge_result_replay import MapMergeResultReplay
from visualization.map_merge_rerun_writer import MapMergeRerunWriter


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an OpenNavMap map-merge result directory to a Rerun .rrd file."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--rerun-output", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="readonly", choices=["readonly"])
    parser.add_argument("--rerun-image-format", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--rerun-jpeg-quality", type=int, default=85)
    parser.add_argument("--rerun-dmatrix-format", type=str, default="png", choices=["png"])
    parser.add_argument("--rerun-axis-scale", type=str, default="auto", choices=["auto"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    replay = MapMergeResultReplay(args.result_dir)
    events = replay.build_events()
    writer = MapMergeRerunWriter(
        image_format=args.rerun_image_format,
        jpeg_quality=args.rerun_jpeg_quality,
        dmatrix_format=args.rerun_dmatrix_format,
        axis_scale=args.rerun_axis_scale,
    )
    writer.write(events, args.rerun_output)


if __name__ == "__main__":
    main()
