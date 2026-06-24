#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import logging
import argparse
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from PIL import Image

from python.benchmark_vpr.dataloader import TestDataset
from python.utils.utils_setting_color_font import acquire_color_palette, acquire_marker, setting_font, acquire_linestyle
from python.utils.utils_geom import compute_pose_error, convert_vec_to_matrix, convert_matrix_to_vec

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Configure matplotlib
setting_font(fontsize=14, titlesize=14, legend_fontsize=14)
PALLETE = acquire_color_palette()
MARKERS = acquire_marker()
LINESTYLE = acquire_linestyle()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='VPR Visualization System')
    parser.add_argument('--dataset_name', type=str, default='ucl_campus',
                        choices=['ucl_campus', 'robocar', 'fusionportable'])
    parser.add_argument('--database_folder', type=Path, required=True)
    parser.add_argument('--queries_folder', type=Path, required=True, nargs='+')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--vpr_model', type=str, default='cosplace',
                        choices=['NetVLAD', 'CosPlace'])
    parser.add_argument('--backbone', type=str, default='ResNet18')
    parser.add_argument('--descriptors_dimension', type=int, default=256)
    parser.add_argument('--trans_thresh', type=float, default=7.5)
    parser.add_argument('--rot_thresh', type=float, default=75.0)
    parser.add_argument('--z_ratio', type=float, default=0.2,
                        help='Query z offset ratio relative to the reference trajectory span.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--output_path', type=Path, default='results')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def convert_pose_to_2d(pose: np.ndarray) -> tuple[float, float]:
    """Convert 7D pose (x,y,z + quaternion wxyz) to 2D (x, y) world coordinates."""
    Tc2w = convert_vec_to_matrix(pose[4:], pose[:4], 'wxyz')
    trans, _ = convert_matrix_to_vec(np.linalg.inv(Tc2w), 'xyzw')
    return (trans[0], trans[1])


def load_poses(test_ds: TestDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (db_poses, query_poses) as (N, 7) arrays from a TestDataset."""
    db_poses = np.array([
        test_ds.database_poses[test_ds.database_image_names[i]]
        for i in range(test_ds.num_database)
    ])
    query_poses = np.array([
        test_ds.queries_poses[test_ds.queries_image_names[i]]
        for i in range(test_ds.num_queries)
    ])
    return db_poses, query_poses


# ---------------------------------------------------------------------------
# Descriptor extraction
# ---------------------------------------------------------------------------

def extract_descriptors(
    model: torch.nn.Module,
    test_ds: TestDataset,
    descriptors_dimension: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (db_descs, query_descs) using the given VPR model."""
    all_descs = np.empty((len(test_ds), descriptors_dimension), dtype='float32')

    with torch.no_grad():
        db_subset = Subset(test_ds, list(range(test_ds.num_database)))
        db_loader = DataLoader(db_subset, batch_size=batch_size, num_workers=4)
        for images, indices, _ in db_loader:
            descs = model(images.to(device)).cpu().numpy()
            all_descs[indices.numpy()] = descs

        q_subset = Subset(test_ds, list(range(test_ds.num_database, len(test_ds))))
        q_loader = DataLoader(q_subset, batch_size=1, num_workers=4)
        for images, indices, _ in q_loader:
            descs = model(images.to(device)).cpu().numpy()
            all_descs[indices.numpy()] = descs

    return all_descs[:test_ds.num_database], all_descs[test_ds.num_database:]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def compute_diff_matrix(db_descs: np.ndarray, query_descs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute cosine distance matrix D of shape (n_db, n_query)."""
    dots = np.dot(db_descs, query_descs.T)
    db_norms = np.linalg.norm(db_descs, axis=1)[:, None]
    q_norms = np.linalg.norm(query_descs, axis=1)[None, :]
    return 1.0 - dots / (db_norms * q_norms + eps)


def find_valid_matches(
    test_ds: TestDataset,
    trans_thresh: float,
    rot_thresh: float,
) -> list[tuple[int, int]]:
    """Find ground-truth valid (query_idx, db_idx) pairs using pose thresholds.

    For each query frame the single nearest database frame within both thresholds
    is returned (one-to-one, best-distance match).
    """
    valid_pairs: list[tuple[int, int]] = []

    for q_idx in range(test_ds.num_queries):
        query_pose = test_ds.queries_poses[test_ds.queries_image_names[q_idx]]
        Tc2w_q = convert_vec_to_matrix(query_pose[4:], query_pose[:4], 'wxyz')
        trans_q, quat_q = convert_matrix_to_vec(np.linalg.inv(Tc2w_q), 'xyzw')

        best_db_idx = None
        best_trans_err = float('inf')

        for db_idx in range(test_ds.num_database):
            db_pose = test_ds.database_poses[test_ds.database_image_names[db_idx]]
            Tc2w_db = convert_vec_to_matrix(db_pose[4:], db_pose[:4], 'wxyz')
            trans_db, quat_db = convert_matrix_to_vec(np.linalg.inv(Tc2w_db), 'xyzw')

            trans_err, rot_err = compute_pose_error(
                (trans_q, quat_q), (trans_db, quat_db), mode='vector'
            )

            if trans_err <= trans_thresh and rot_err <= rot_thresh and trans_err < best_trans_err:
                best_db_idx = db_idx
                best_trans_err = trans_err

        if best_db_idx is not None:
            valid_pairs.append((q_idx, best_db_idx))

    return valid_pairs


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_vpr_data_dmatrix(
    db_poses: np.ndarray,
    query_poses: np.ndarray,
    D_all: np.ndarray,
    valid_pairs: list[tuple[int, int]],
    output_path: Path,
) -> None:
    """Three-panel figure: difference matrix, weight matrix, and 2-D trajectories."""
    fig = plt.figure(figsize=(18, 10))

    # 1. Difference Matrix
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(D_all, cmap='Greys', aspect='auto')
    for q_idx, db_idx in valid_pairs:
        ax1.plot(q_idx, db_idx, 'r.', markersize=4, alpha=1.0)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    im1.set_clim(0.0, 1.0)
    ax1.set_xlabel('Query Index', fontsize=12)
    ax1.set_ylabel('Reference Index', fontsize=12)
    ax1.set_title('Difference Matrix', fontsize=14)
    ax1.set_aspect('equal')

    # 2. Weight Matrix
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(1.0 / (D_all + 1e-8), cmap='Greys', aspect='auto')
    for q_idx, db_idx in valid_pairs:
        ax2.plot(q_idx, db_idx, 'r.', markersize=4, alpha=1.0)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    im2.set_clim(0.0, 5.0)
    ax2.set_xlabel('Query Index', fontsize=14)
    ax2.set_ylabel('Reference Index', fontsize=14)
    ax2.set_title('Weight Matrix', fontsize=16)
    ax2.set_aspect('equal')

    # 3. Trajectories
    ax3 = fig.add_subplot(133)
    db_xy = np.array([convert_pose_to_2d(p) for p in db_poses])
    query_xy = np.array([convert_pose_to_2d(p) for p in query_poses])

    ax3.plot(db_xy[:, 0], db_xy[:, 1], c=PALLETE[1], linewidth=2, linestyle='--', label='Reference', zorder=1)
    ax3.plot(query_xy[:, 0], query_xy[:, 1], c=PALLETE[2], linewidth=2, linestyle='-', label='Query', zorder=0)

    for q_idx, db_idx in valid_pairs:
        ax3.plot(
            [query_xy[q_idx, 0], db_xy[db_idx, 0]],
            [query_xy[q_idx, 1], db_xy[db_idx, 1]],
            'g-', linewidth=1, alpha=0.5,
        )

    ax3.set_xlabel('X [m]', fontsize=14)
    ax3.set_ylabel('Y [m]', fontsize=14)
    ax3.set_title('Trajectories', fontsize=16)
    ax3.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1.05, 1.0))
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_vpr_data_queries(
    db_poses: np.ndarray,
    all_query_poses: list[np.ndarray],
    all_valid_pairs: list[list[tuple[int, int]]],
    query_names: list[str],
    output_dir: Path,
    trans_thresh: float,
    rot_thresh: float,
    z_ratio: float,
) -> None:
    """One 3D figure per query sequence with query lifted above the reference map."""
    db_xy = np.array([convert_pose_to_2d(p) for p in db_poses])
    ref_x_span = np.max(db_xy[:, 0]) - np.min(db_xy[:, 0])
    ref_y_span = np.max(db_xy[:, 1]) - np.min(db_xy[:, 1])
    ref_span = max(ref_x_span, ref_y_span, 1.0)
    z_offset = max(z_ratio * ref_span, 1.0)
    db_z = np.zeros(len(db_xy))

    for query_poses, valid_pairs, query_name in zip(all_query_poses, all_valid_pairs, query_names):
        query_xy = np.array([convert_pose_to_2d(p) for p in query_poses])
        query_z = np.full(len(query_xy), z_offset)
        all_xy = np.vstack((db_xy, query_xy))
        x_min, y_min = np.min(all_xy, axis=0)
        x_max, y_max = np.max(all_xy, axis=0)
        x_pad = max((x_max - x_min) * 0.05, 2.0)
        y_pad = max((y_max - y_min) * 0.05, 2.0)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('ortho')
        ax.view_init(elev=25, azim=-60)
        query_color = PALLETE[1]
        match_color = PALLETE[0]

        reference_handle, = ax.plot(
            db_xy[:, 0], db_xy[:, 1], db_z,
            color='0.60', linewidth=1.0, marker='o', markersize=2.0,
            markeredgewidth=0.0, alpha=0.8,
            label='Reference', zorder=1,
        )

        matched_db_indices = [db_idx for _, db_idx in valid_pairs]

        for q_idx, db_idx in valid_pairs:
            ax.plot(
                [query_xy[q_idx, 0], db_xy[db_idx, 0]],
                [query_xy[q_idx, 1], db_xy[db_idx, 1]],
                [z_offset, 0.0],
                color=match_color, linewidth=0.5, alpha=0.35, zorder=2,
            )

        matched_handle = None
        if matched_db_indices:
            matched_db_xy = db_xy[matched_db_indices]
            matched_handle = ax.scatter(
                matched_db_xy[:, 0], matched_db_xy[:, 1],
                np.zeros(len(matched_db_xy)),
                marker='o', s=20.0, facecolors='none', edgecolors=match_color,
                linewidths=1.2, label='Matched node', zorder=3,
            )

        query_handle, = ax.plot(
            query_xy[:, 0], query_xy[:, 1], query_z,
            color=query_color, linewidth=1.8, marker='o', markersize=2.5,
            markeredgewidth=0.0, alpha=0.9,
            label='Query', zorder=4,
        )

        n_query = len(query_poses)
        n_pos = len(valid_pairs)
        ax.set_title(
            f'GT Loop [{trans_thresh:.1f}m, {rot_thresh:.1f}°]: '
            f'N_pos={n_pos}, N_query={n_query}',
            fontsize=10,
            y=0.85,
            pad=0,
        )
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_zlim(0.0, z_offset)
        ax.set_box_aspect((x_max - x_min + 2.0 * x_pad, y_max - y_min + 2.0 * y_pad, z_offset))
        ax.set_xlabel('X [m]', fontsize=7)
        ax.set_ylabel('Y [m]', fontsize=7)
        ax.set_zlabel('')
        ax.set_zticks([])
        ax.set_zticklabels([])
        ax.zaxis.line.set_linewidth(0.0)
        ax.zaxis.pane.set_visible(False)
        ax.zaxis._axinfo['grid']['color'] = (1.0, 1.0, 1.0, 0.0)
        ax.zaxis._axinfo['grid']['linewidth'] = 0.0
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        legend_handles = [reference_handle, query_handle]
        if matched_handle is not None:
            legend_handles.append(matched_handle)
        ax.legend(
            handles=legend_handles,
            fontsize=8,
            loc='upper center',
            bbox_to_anchor=(0.52, 0.75),
            ncol=3,
            framealpha=0.75,
            borderpad=0.3,
            columnspacing=0.8,
            handlelength=1.3,
        )

        for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            pane_axis.pane.set_edgecolor('0.85')

        # fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90)
        png_path = output_dir / f'vpr_data_{query_name}.png'
        pdf_path = output_dir / f'vpr_data_{query_name}.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f'Saved VPR visualization for {query_name} to {png_path}')


def save_img_valid_pairs(
    all_test_ds: list[TestDataset],
    all_valid_pairs: list[list[tuple[int, int]]],
    query_names: list[str],
    output_dir: Path,
) -> None:
    """Save side-by-side query/reference image pairs for each matched sequence."""
    for test_ds, valid_pairs, query_name in zip(all_test_ds, all_valid_pairs, query_names):
        query_output_dir = output_dir / query_name
        query_output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f'Saving matched image pairs for {query_name} ({len(valid_pairs)} pairs)')

        for q_idx, db_idx in valid_pairs:
            query_img = Image.open(test_ds.queries_image_paths[q_idx]).convert('RGB')
            db_img = Image.open(test_ds.database_image_paths[db_idx]).convert('RGB')

            query_img = query_img.resize(
                (query_img.width // 2, query_img.height // 2), resample=Image.BICUBIC
            )
            db_img = db_img.resize(
                (db_img.width // 2, db_img.height // 2), resample=Image.BICUBIC
            )

            fig, axes = plt.subplots(2, 1, figsize=(6, 6))
            axes[0].imshow(query_img)
            axes[0].set_title(f'Query Image (idx={q_idx})', fontsize=16)
            axes[0].axis('off')
            axes[1].imshow(db_img)
            axes[1].set_title(f'Reference Image (idx={db_idx})', fontsize=16)
            axes[1].axis('off')

            plt.tight_layout()
            out_path = query_output_dir / f'matched_query_{q_idx:04d}_ref_{db_idx:04d}.jpg'
            plt.savefig(out_path, dpi=100, bbox_inches='tight')
            plt.close()

        logging.info(f'Saved {len(valid_pairs)} matched image pairs to {query_output_dir}')


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def evaluate_vpr_system(args: argparse.Namespace) -> None:
    """Load datasets, find GT matches, and generate all visualisations."""
    output_dir: Path = args.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    db_poses: np.ndarray | None = None
    all_query_poses: list[np.ndarray] = []
    all_valid_pairs: list[list[tuple[int, int]]] = []
    all_test_ds: list[TestDataset] = []
    query_names: list[str] = []

    for query_folder in args.queries_folder:
        test_ds = TestDataset(args.database_folder, query_folder, args.image_size)
        cur_db_poses, query_poses = load_poses(test_ds)

        if db_poses is None:
            db_poses = cur_db_poses

        all_query_poses.append(query_poses)
        all_test_ds.append(test_ds)
        query_names.append(query_folder.name)

        valid_pairs = find_valid_matches(test_ds, args.trans_thresh, args.rot_thresh)
        all_valid_pairs.append(valid_pairs)
        logging.info(f'{query_folder.name}: {len(valid_pairs)} valid pairs out of {test_ds.num_queries} queries')

        # Optional: descriptor-based visualisation (uncomment to enable)
        # vpr_model = initialize_vpr_model(args.vpr_model, args.backbone, args.descriptors_dimension, args.device)
        # db_descs, query_descs = extract_descriptors(vpr_model, test_ds, args.descriptors_dimension, args.batch_size, args.device)
        # D_all = compute_diff_matrix(db_descs, query_descs)
        # visualize_vpr_data_dmatrix(cur_db_poses, query_poses, D_all, valid_pairs, output_dir / f'vpr_data_{query_folder.name}.jpg')
        # np.save(output_dir / f'D_all_{query_folder.name}.npy', D_all)

    visualize_vpr_data_queries(
        db_poses, all_query_poses, all_valid_pairs, query_names, output_dir,
        args.trans_thresh, args.rot_thresh, args.z_ratio,
    )
    # save_img_valid_pairs(all_test_ds, all_valid_pairs, query_names, output_dir)


if __name__ == '__main__':
    args = parse_arguments()
    evaluate_vpr_system(args)
