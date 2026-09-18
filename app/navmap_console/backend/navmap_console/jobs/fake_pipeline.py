"""Stand-in for python/map_merge_pipeline.py: same CLI subset, same log anchors, no GPU.

Each submap directory is copied verbatim as its step directory (a submap is itself a
complete map), so downstream readers, consolidation and the frontend see real files.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional


def _read_list(path: Path) -> List[Path]:
    return [Path(l.strip()) for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submap_list", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--step_dir_style", default="indexed", choices=["cumulative", "indexed"])
    ap.add_argument("--start_step", type=int, default=0)
    ap.add_argument("--append_from", default=None)
    args = ap.parse_args(argv)
    step_seconds = float(os.environ.get("NAVMAP_CONSOLE_FAKE_STEP_SECONDS", "0.2"))
    fail_at = os.environ.get("NAVMAP_CONSOLE_FAKE_FAIL_AT")
    fail_at_i = int(fail_at) if fail_at else None
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    submaps = _read_list(Path(args.submap_list))
    if args.append_from:
        print(f"Loaded base map from {args.append_from}", flush=True)
    prev_name: Optional[str] = None
    total_nodes = 0
    for k, sub in enumerate(submaps):
        i = args.start_step + k
        sid = sub.name
        print(f"--- Merging submap {i}: {sid} ---", flush=True)
        if fail_at_i is not None and i == fail_at_i:
            print("Traceback (most recent call last):\n  File \"fake\", line 1\nRuntimeError: fake failure", flush=True)
            return 1
        time.sleep(step_seconds / 4)
        print("D_all shape: (10, 12)", flush=True)
        time.sleep(step_seconds / 4)
        print("PGO: initial error: 1.234", flush=True)
        sys.stdout.write("progress 50%\rprogress 100%\n")  # tqdm-style redraw on one line
        sys.stdout.flush()
        time.sleep(step_seconds / 4)
        print("PGO: final error: 0.456", flush=True)
        if args.step_dir_style == "indexed":
            name = f"merge_{i:03d}_{sid}"
        else:
            name = f"{prev_name}_{sid}" if prev_name else f"merge_{sid}"
        out = result_dir / name
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(sub, out)
        (out / "preds").mkdir(exist_ok=True)
        n = sum(1 for l in (out / "poses.txt").read_text().splitlines() if l.strip())
        id_offset = total_nodes
        total_nodes += n
        link = result_dir / "merge_finalmap"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(out)
        print(f"Saved intermediate result: {out}", flush=True)
        print(f"STEP_DONE index={i} sid={sid} dir={out} id_offset={id_offset} odom_nodes={total_nodes} "
              f"covis_nodes={total_nodes} components=1 registry={k}", flush=True)
        prev_name = name
        time.sleep(step_seconds / 4)
    print(f"merge_finalmap -> {result_dir / prev_name}" if prev_name else "no submaps", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
