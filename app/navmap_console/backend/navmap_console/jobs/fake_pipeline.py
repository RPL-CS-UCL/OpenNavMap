"""Stand-in for python/map_merge_pipeline.py: same CLI subset, same log anchors, no GPU.

Every step directory is cumulative like the real pipeline: the previous step's map files plus the
new submap appended with node ids shifted by id_offset, frames renamed to seq/{id:06d}.color.jpg,
one odometry edge bridging the two, and the new submap translated +2 m in x per step so a merged
map spreads out in 3D. Only the new submap's images are copied (as in the real pipeline).
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from .fake_preds import _quat_wxyz_to_matrix, write_fake_preds
except ImportError:  # run as a script: python fake_pipeline.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_preds import _quat_wxyz_to_matrix, write_fake_preds  # type: ignore

_NAME_FILES = ("poses.txt", "timestamps.txt", "intrinsics.txt", "gps_data.txt", "iqa_data.txt",
               "database_descriptors.txt", "poses_abs_gt.txt")
_EDGE_FILES = ("edges_odom.txt", "edges_covis.txt", "edges_trav.txt")
_STEP_SHIFT = 2.0  # metres along x per step


def _read_list(path: Path) -> List[Path]:
    return [Path(l.strip()) for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def _rows(path: Path) -> List[str]:
    return [l for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def _local_id(name: str) -> int:
    digits = "".join(ch for ch in Path(name).name.split(".")[0] if ch.isdigit())
    return int(digits) if digits else 0


def _shift_pose_row(line: str, new_name: str, dx: float) -> str:
    """poses.txt row is w2c (q t); moving the camera centre by d changes t by -R d."""
    parts = line.split()
    q = [float(v) for v in parts[1:5]]
    t = np.asarray([float(v) for v in parts[5:8]])
    t = t - _quat_wxyz_to_matrix(q) @ np.array([dx, 0.0, 0.0])
    return " ".join([new_name] + parts[1:5] + ["%.6f" % v for v in t])


def build_step_dir(prev: Optional[Path], sub: Path, out: Path, id_offset: int) -> int:
    if out.exists():
        shutil.rmtree(out)
    (out / "seq").mkdir(parents=True)
    (out / "preds").mkdir()
    step = 0 if prev is None else int(round(_local_id(prev.name.split("_")[1]) + 1)) if prev.name.startswith(
        "merge_") and prev.name.split("_")[1].isdigit() else 0
    dx = _STEP_SHIFT * step
    for name in _NAME_FILES:
        old = _rows(prev / name) if prev is not None else []
        new = []
        for line in _rows(sub / name):
            parts = line.split()
            gid = id_offset + _local_id(parts[0])
            new_name = f"seq/{gid:06d}.color.jpg"
            if name == "poses.txt":
                new.append(_shift_pose_row(line, new_name, dx))
            else:
                new.append(" ".join([new_name] + parts[1:]))
        if old or new or (sub / name).is_file():
            (out / name).write_text("\n".join(old + new) + ("\n" if old or new else ""))
    for name in _EDGE_FILES:
        old = _rows(prev / name) if prev is not None else []
        new = []
        for line in _rows(sub / name):
            a, b, *rest = line.split()
            w = rest[0] if rest else "1.0"
            new.append(f"{int(float(a)) + id_offset} {int(float(b)) + id_offset} {w}")
        if name == "edges_odom.txt" and prev is not None and id_offset > 0:
            new.insert(0, f"{id_offset - 1} {id_offset} 1.0")
        (out / name).write_text("\n".join(old + new) + ("\n" if old or new else ""))
    for img in sorted((sub / "seq").glob("*.color.jpg")):  # synthetic_map also has *.depth.png; skip those
        gid = id_offset + _local_id(img.name)
        shutil.copy2(img, out / "seq" / f"{gid:06d}.color.jpg")
    return len(_rows(out / "poses.txt"))


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
    prev: Optional[Path] = Path(args.append_from) if args.append_from else None
    if prev is not None:
        print(f"Loaded base map from {prev}", flush=True)
    total_nodes = len(_rows(prev / "poses.txt")) if prev is not None else 0
    accepted: List[Tuple[int, int]] = []
    prev_name: Optional[str] = None
    for k, sub in enumerate(submaps):
        i = args.start_step + k
        sid = sub.name
        print(f"--- Merging submap {i}: {sid} ---", flush=True)
        if fail_at_i is not None and i == fail_at_i:
            print("Traceback (most recent call last):\n  File \"fake\", line 1\nRuntimeError: fake failure", flush=True)
            return 1
        if not sub.is_dir():
            print(f"ERROR: submap directory not found: {sub}", flush=True)
            return 2
        time.sleep(step_seconds / 4)
        print(f"D_all shape: ({max(total_nodes, 1)}, {len(_rows(sub / 'poses.txt'))})", flush=True)
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
        id_offset = total_nodes
        total_nodes = build_step_dir(prev, sub, out, id_offset)
        pair = write_fake_preds(out, id_offset, total_nodes, accepted, seed=i)
        if pair[0] >= 0:
            accepted.append(pair)
        link = result_dir / "merge_finalmap"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(out)
        print(f"Saved intermediate result: {out}", flush=True)
        print(f"STEP_DONE index={i} sid={sid} dir={out} id_offset={id_offset} odom_nodes={total_nodes} "
              f"covis_nodes={total_nodes} components=1 registry={len(accepted)}", flush=True)
        prev, prev_name = out, name
        time.sleep(step_seconds / 4)
    print(f"merge_finalmap -> {result_dir / prev_name}" if prev_name else "no submaps", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
