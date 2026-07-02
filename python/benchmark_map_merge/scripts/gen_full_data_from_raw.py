"""
Author: Jianhao Jiao
Date: 2025-06-14
Description: Recover full sequence data (all frames between first/last keyframe timestamps)
             from raw out_general sessions, producing the same file format as
             gendataset_from_files.py (map_merge mode) but without keyframe selection.
             Output includes: seq/ (images), intrinsics.txt, poses_abs_gt.txt, poses.txt,
             timestamps.txt, gps_data.txt.

Usage:
    python gen_full_data_from_raw.py \
        --submap_dir /path/to/s00000_aria_data/0 \
        --raw_dir    /path/to/out_general_ucl_campus_20240904_0835 \
        --out_dir    /path/to/s00000_aria_data_000/0
"""

import os
import sys
import numpy as np
import argparse
from scipy.spatial.transform import Rotation


def convert_vec_to_matrix(vec_p, vec_q, mode='xyzw'):
    tf = np.eye(4)
    if mode == 'xyzw':
        tf[:3, :3] = Rotation.from_quat(vec_q).as_matrix()
        tf[:3, 3] = vec_p
    elif mode == 'wxyz':
        tf[:3, :3] = Rotation.from_quat(np.roll(vec_q, -1)).as_matrix()
        tf[:3, 3] = vec_p
    return tf


def convert_matrix_to_vec(tf_matrix, mode='wxyz'):
    vec_p = tf_matrix[:3, 3]
    if mode == 'xyzw':
        vec_q = Rotation.from_matrix(tf_matrix[:3, :3]).as_quat()
    elif mode == 'wxyz':
        vec_q = np.roll(Rotation.from_matrix(tf_matrix[:3, :3]).as_quat(), 1)
    return vec_p, vec_q


def find_nearest_index(timestamps, target):
    return int(np.argmin(np.abs(timestamps - target)))


def main():
    parser = argparse.ArgumentParser(description="Generate full-frame data from raw session")
    parser.add_argument('--submap_dir', type=str, required=True,
                        help='Existing submap directory (e.g. s00000_aria_data/0) to read timestamps from')
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Raw out_general session directory')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for full-frame data')
    args = parser.parse_args()

    print(f"Submap dir: {args.submap_dir}")
    print(f"Raw dir:    {args.raw_dir}")
    print(f"Out dir:    {args.out_dir}")

    timestamps_sub = np.loadtxt(os.path.join(args.submap_dir, 'timestamps.txt'), dtype=str)
    ts_start = float(timestamps_sub[0, 1])
    ts_end = float(timestamps_sub[-1, 1])
    print(f"Submap time range: {ts_start:.6f} -> {ts_end:.6f}")

    poses_cl = np.loadtxt(os.path.join(args.raw_dir, 'poses_closed_loop.txt'))
    poses_ol = np.loadtxt(os.path.join(args.raw_dir, 'poses_open_loop.txt'))
    intrinsics_raw = np.loadtxt(os.path.join(args.raw_dir, 'intrinsics.txt'))

    idx_start = find_nearest_index(poses_cl[:, 0], ts_start)
    idx_end = find_nearest_index(poses_cl[:, 0], ts_end)
    print(f"Frame indices in raw: [{idx_start}, {idx_end}]  (total {idx_end - idx_start + 1} frames)")

    gps_raw = None
    gps_path = os.path.join(args.raw_dir, 'gps_data.txt')
    if os.path.exists(gps_path):
        gps_raw = np.loadtxt(gps_path)

    os.makedirs(os.path.join(args.out_dir, 'seq'), exist_ok=True)

    seg_timestamps = []
    seg_intrinsics = []
    seg_gps = []
    seg_poses_abs_gt = []
    seg_poses_odom = []

    for new_idx, raw_idx in enumerate(range(idx_start, idx_end + 1)):
        img_name = f'seq/{new_idx:06d}.color.jpg'

        src_img = os.path.join(args.raw_dir, 'seq', f'{raw_idx:06d}.color.jpg')
        dst_img = os.path.join(args.out_dir, 'seq', f'{new_idx:06d}.color.jpg')
        os.system(f'cp {src_img} {dst_img}')

        seg_timestamps.append([img_name, f'{poses_ol[raw_idx, 0]:.9f}'])

        K = intrinsics_raw[raw_idx]
        seg_intrinsics.append(
            [img_name,
             f'{K[0]:.6f}', f'{K[1]:.6f}', f'{K[2]:.6f}', f'{K[3]:.6f}',
             f'{int(K[4])}', f'{int(K[5])}'])

        if gps_raw is not None:
            gps_row = gps_raw[raw_idx, 1:]
            seg_gps.append([img_name] + [f'{v:.6f}' for v in gps_row])
        else:
            seg_gps.append([img_name] + ['nan'] * 5)

        Twc = convert_vec_to_matrix(poses_ol[raw_idx, 1:4], poses_ol[raw_idx, 4:], 'xyzw')
        tsl, quat = convert_matrix_to_vec(np.linalg.inv(Twc), 'wxyz')
        seg_poses_odom.append([img_name] + [f'{v:.6f}' for v in quat] + [f'{v:.6f}' for v in tsl])

        Twc = convert_vec_to_matrix(poses_cl[raw_idx, 1:4], poses_cl[raw_idx, 4:], 'xyzw')
        tsl, quat = convert_matrix_to_vec(np.linalg.inv(Twc), 'wxyz')
        seg_poses_abs_gt.append([img_name] + [f'{v:.6f}' for v in quat] + [f'{v:.6f}' for v in tsl])

        if (new_idx + 1) % 200 == 0:
            print(f"  Processed {new_idx + 1}/{idx_end - idx_start + 1} frames")

    def write_entries(filepath, entries):
        with open(filepath, 'w') as f:
            for row in entries:
                f.write(' '.join(row) + '\n')

    write_entries(os.path.join(args.out_dir, 'timestamps.txt'), seg_timestamps)
    write_entries(os.path.join(args.out_dir, 'intrinsics.txt'), seg_intrinsics)
    write_entries(os.path.join(args.out_dir, 'gps_data.txt'), seg_gps)
    write_entries(os.path.join(args.out_dir, 'poses_abs_gt.txt'), seg_poses_abs_gt)
    write_entries(os.path.join(args.out_dir, 'poses.txt'), seg_poses_odom)

    n_frames = idx_end - idx_start + 1
    print(f"\nDone: {n_frames} frames written to {args.out_dir}")
    print(f"  timestamps.txt     ({n_frames} entries)")
    print(f"  intrinsics.txt     ({n_frames} entries)")
    print(f"  poses_abs_gt.txt   ({n_frames} entries)")
    print(f"  poses.txt          ({n_frames} entries)")
    print(f"  gps_data.txt       ({n_frames} entries)")
    print(f"  seq/               ({n_frames} images)")


if __name__ == '__main__':
    main()
