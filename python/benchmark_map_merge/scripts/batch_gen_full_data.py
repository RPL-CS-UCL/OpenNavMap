"""
Batch run gen_full_data_from_raw.py for all 55 subfolders of s00000_aria_data.
Auto-matches each subfolder to its raw session based on timestamp overlap.
"""

import os
import sys
import subprocess
import numpy as np
import argparse
from glob import glob


def get_raw_sessions(raw_base_dir):
    sessions = {}
    for d in sorted(glob(os.path.join(raw_base_dir, 'out_general_ucl_campus_*'))):
        name = os.path.basename(d)
        poses_cl = np.loadtxt(os.path.join(d, 'poses_closed_loop.txt'))
        sessions[name] = {
            'dir': d,
            'ts_start': poses_cl[0, 0],
            'ts_end': poses_cl[-1, 0],
            'num_frames': len(poses_cl),
        }
    return sessions


def find_raw_session(ts, sessions):
    for name, info in sessions.items():
        if info['ts_start'] <= ts <= info['ts_end']:
            return name, info
    best = min(sessions.items(),
               key=lambda x: min(abs(x[1]['ts_start'] - ts), abs(x[1]['ts_end'] - ts)))
    return best[0], best[1]


def main():
    parser = argparse.ArgumentParser(description="Batch generate full-frame data for all submaps")
    parser.add_argument('--submap_root', type=str, required=True,
                        help='Root dir containing submap folders (0, 1, ..., N)')
    parser.add_argument('--raw_root', type=str, required=True,
                        help='Root dir containing raw out_general sessions')
    parser.add_argument('--out_root', type=str, required=True,
                        help='Root output dir for full-frame data')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'gen_full_data_from_raw.py')

    sessions = get_raw_sessions(args.raw_root)
    print(f"Found {len(sessions)} raw sessions")

    submap_dirs = sorted(glob(os.path.join(args.submap_root, '[0-9]*')),
                         key=lambda x: int(os.path.basename(x)))
    print(f"Found {len(submap_dirs)} submap folders")

    end_idx = args.end if args.end is not None else len(submap_dirs) - 1

    failed = []
    for i in range(args.start, min(end_idx + 1, len(submap_dirs))):
        sub_dir = submap_dirs[i]
        sub_name = os.path.basename(sub_dir)

        timestamps_sub = np.loadtxt(os.path.join(sub_dir, 'timestamps.txt'), dtype=str)
        ts0 = float(timestamps_sub[0, 1])

        raw_name, raw_info = find_raw_session(ts0, sessions)
        out_dir = os.path.join(args.out_root, sub_name)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(submap_dirs)}] subfolder {sub_name} -> {raw_name}")
        print(f"       raw frames: {raw_info['num_frames']}")

        cmd = [
            'python3', script,
            '--submap_dir', sub_dir,
            '--raw_dir', raw_info['dir'],
            '--out_dir', out_dir,
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  FAILED")
            failed.append(sub_name)

    print(f"\n{'='*60}")
    print(f"Done. {len(submap_dirs) - len(failed)}/{len(submap_dirs)} succeeded")
    if failed:
        print(f"Failed: {failed}")


if __name__ == '__main__':
    main()
