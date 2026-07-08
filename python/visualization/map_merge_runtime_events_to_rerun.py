from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from visualization.map_merge_runtime_rerun_renderer import MapMergeRuntimeRerunRenderer


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render runtime map-merge demo_events.jsonl to a Rerun .rrd file."
    )
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--rerun-output", type=Path, required=True)
    parser.add_argument("--render-trace", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    MapMergeRuntimeRerunRenderer(args.event_dir, args.render_trace).write(args.rerun_output)


if __name__ == "__main__":
    main()
