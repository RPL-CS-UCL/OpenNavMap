#!/usr/bin/env python

from __future__ import annotations

import _bootstrap_imports  # noqa: F401

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation

from utils.utils_setting_color_font import acquire_color_palette, setting_font

setting_font(fontsize=16, titlesize=16, legend_fontsize=16)
PALLETE = acquire_color_palette()


def load_mapfree_poses(poses_file: Path) -> dict[str, np.ndarray]:
    """Load poses.txt → dict[frame_path, T_c2w (4×4)].

    poses.txt format: frame_path qw qx qy qz tx ty tz  (world-to-camera)
    Lines starting with '#' are skipped.
    """
    poses: dict[str, np.ndarray] = {}
    with open(poses_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            frame_path = parts[0]
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # xyzw convention for scipy
            T_w2c = np.eye(4)
            T_w2c[:3, :3] = R
            T_w2c[:3, 3] = [tx, ty, tz]
            poses[frame_path] = np.linalg.inv(T_w2c)
    return poses


def sample_ref_images(seq1_dir: Path, n: int = 4, seed: int = 0) -> list[Path]:
    """Randomly select up to n .jpg images from seq1_dir (deterministic)."""
    imgs = sorted(seq1_dir.glob("*.jpg"))
    rng = random.Random(seed)
    return rng.sample(imgs, min(n, len(imgs)))


def draw_orientation_arrow(
    ax: plt.Axes,
    transform: np.ndarray,
    length: float,
    style: dict,
) -> None:
    """Draw a 2-D orientation arrow on the x-z top-down plane."""
    start = transform[:3, 3]
    direction = transform[:3, :3] @ np.array([0, 0, length])
    ax.arrow(
        start[0], start[2], direction[0], direction[2],
        head_width=style["head_width"] * 1.0,
        head_length=style["head_length"] * 1.2,
        width=style["head_width"] * 0.25,
        fc=style["fc"], ec=style["fc"],
        zorder=style["zorder"],
    )


def visualize_scene(
    scene_dir: Path,
    output_path: Path,
    n_refs: int = 4,
    seed: int = 0,
) -> None:
    """Render two-row figure: image strip (top) + top-down pose map (bottom)."""
    poses = load_mapfree_poses(scene_dir / "poses.txt")

    query_img_path = scene_dir / "seq0" / "frame_00000.jpg"
    ref_img_paths = sample_ref_images(scene_dir / "seq1", n=n_refs, seed=seed)

    # ── Dynamic layout based on image aspect ratio ───────────────────────────
    # portrait (ratio < 1, e.g. mapfree 540×720): 3 rows × 2 cols image grid
    # landscape (ratio ≥ 1, e.g. ucl/360loc 1024×576): 5 rows × 1 col image strip
    img_w, img_h = Image.open(query_img_path).size
    portrait = img_w < img_h

    if portrait:
        # 3×2 grid — left 48 %, pose right 55–100 %
        n_img_rows, n_img_cols = 3, 2
        fig = plt.figure(figsize=(13, 10))
        gs_imgs = fig.add_gridspec(
            n_img_rows, n_img_cols, hspace=0.00, wspace=-0.3,
            left=0.02, right=0.48, top=0.97, bottom=0.03,
        )
        gs_pose = fig.add_gridspec(
            1, 1, left=0.51, right=1.00, top=0.97, bottom=0.03,
        )
        grid_positions = [(r, c) for r in range(n_img_rows) for c in range(n_img_cols)]
        label_fontsize = 24  # 16 * 1.5
    else:
        # 5×1 strip — pose width adapts to data x/z span ratio
        n_total = 1 + len(ref_img_paths)  # query + refs
        n_img_rows, n_img_cols = n_total, 1

        # Compute pose data x/z span to set adaptive pose width
        pts = np.array([T[:3, 3] for T in poses.values()])
        x_span = float(pts[:, 0].max() - pts[:, 0].min())
        z_span = float(pts[:, 2].max() - pts[:, 2].min())
        pose_aspect = x_span / max(z_span, 1e-3)  # x/z ratio of data

        # figsize: height driven by 5 landscape images stacked
        # image col occupies left 30% of 13in → width ~3.9in
        # each image row height = 3.9 / img_aspect = 3.9 / 1.778 ≈ 2.19in
        # total img height ≈ 5 * 2.19 = 10.95in → use 11in
        fig_w, fig_h = 13.0, 11.0
        fig = plt.figure(figsize=(fig_w, fig_h))

        img_right = 0.32
        gap = 0.09  # small fixed gap between image strip and pose area

        # Pose area spans from (img_right + gap) to 1.0 in figure coords
        pose_area_left = img_right + gap
        pose_area_width = 1.0 - pose_area_left  # fraction of fig width
        pose_area_height = 0.94  # top - bottom = 0.97 - 0.03

        # Actual pose plot pixel dimensions if it fills pose_area fully:
        #   plot_w_in = pose_area_width * fig_w
        #   plot_h_in = pose_area_height * fig_h
        # With aspect='equal', the rendered data region has aspect = pose_aspect.
        # We want the pose axes to be just wide enough so the square-equal plot
        # fills the height. Needed plot width = plot_h_in * pose_aspect (in inches).
        plot_h_in = pose_area_height * fig_h
        needed_plot_w_in = plot_h_in * pose_aspect
        needed_pose_frac = min(needed_plot_w_in / fig_w, pose_area_width)

        gs_imgs = fig.add_gridspec(
            n_img_rows, n_img_cols, hspace=0.03, wspace=0.00,
            left=0.02, right=img_right, top=0.97, bottom=0.03,
        )
        gs_pose = fig.add_gridspec(
            1, 1,
            left=pose_area_left,
            right=pose_area_left + needed_pose_frac,
            top=0.97, bottom=0.03,
        )
        grid_positions = [(r, 0) for r in range(n_img_rows)]
        label_fontsize = 20
    # ── Image grid ───────────────────────────────────────
    all_img_paths = [query_img_path] + ref_img_paths
    titles = ["Query"] + [f"Ref.{i}" for i in range(len(ref_img_paths))]
    for idx, (img_path, title) in enumerate(zip(all_img_paths, titles)):
        row, col = grid_positions[idx]
        ax_img = fig.add_subplot(gs_imgs[row, col])
        ax_img.imshow(np.array(Image.open(img_path)))
        # Title overlaid inside image (top-left) to avoid consuming vertical space
        ax_img.text(
            0.03, 0.97, title,
            transform=ax_img.transAxes,
            fontsize=label_fontsize, color="white", va="top", ha="left",
            bbox=dict(facecolor="black", alpha=0.8, boxstyle="round,pad=0.2", edgecolor="none"),
        )
        ax_img.axis("off")

    # ── Right: top-down pose map ────────
    ax_pose = fig.add_subplot(gs_pose[0, 0])

    query_key = "seq0/frame_00000.jpg"
    ref_keys = [p.parent.name + "/" + p.name for p in ref_img_paths]

    ref_transforms = [poses[k] for k in ref_keys if k in poses]
    query_transform = poses.get(query_key)

    all_positions = np.array([T[:3, 3] for T in ref_transforms]) if ref_transforms else np.zeros((1, 3))
    if len(all_positions) > 1:
        bounds = np.max(all_positions, axis=0) - np.min(all_positions, axis=0)
        max_bound = max(float(np.max(bounds)) / 2 * 1.5, 0.5)
    else:
        max_bound = 1.0

    arrow_length = max_bound / 10
    head_size = max_bound / 20
    arrow_style = {
        "head_width": head_size,
        "head_length": head_size,
        "fc": PALLETE[0],
        "zorder": 0,
    }

    # Reference poses — PALLETE[0] (green)
    for i, T in enumerate(ref_transforms):
        draw_orientation_arrow(ax_pose, T, arrow_length, arrow_style)
        ax_pose.plot(
            T[0, 3], T[2, 3],
            color=PALLETE[0], marker="o", markersize=12,
            label="Reference Poses" if i == 0 else None,
            zorder=0, markerfacecolor=PALLETE[0],
            linestyle="none", markeredgewidth=3.0,
        )

    # Query pose — PALLETE[1] (red)
    if query_transform is not None:
        arrow_style_q = {**arrow_style, "fc": PALLETE[1], "zorder": 10}
        draw_orientation_arrow(ax_pose, query_transform, arrow_length, arrow_style_q)
        ax_pose.plot(
            query_transform[0, 3], query_transform[2, 3],
            color=PALLETE[1], marker="o", markersize=18,
            label="Query Pose", zorder=10, markerfacecolor="none",
            linestyle="none", markeredgewidth=4.0,
        )

    ax_pose.set_aspect("equal")
    ax_pose.grid(True, linestyle="--", alpha=0.7)
    ax_pose.set_xlabel("X [m]", fontsize=label_fontsize)
    ax_pose.set_ylabel("Y [m]", fontsize=label_fontsize)
    ax_pose.tick_params(axis="both", labelsize=label_fontsize)
    ax_pose.legend(fontsize=label_fontsize, loc='best')
    ax_pose.set_title("Top-down Pose View", fontsize=label_fontsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path.with_suffix('.png')} + .pdf")


def resolve_scene_dir(dataset_root: Path, dataset_name: str, split: str, scene: str) -> Path:
    """Resolve scene directory.

    All datasets follow: <dataset_root>/<dataset_name>/map_free_eval/<split>/<scene>/
    Supported: mapfree, ucl_campus_aria, 360loc_aria
    """
    return dataset_root / dataset_name / "map_free_eval" / split / scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize map-free format dataset scenes")
    parser.add_argument(
        "--dataset_root", type=Path,
        default=Path("/Titan/dataset/data_opennavmap/map_free_eval"),
        help="Root directory containing all datasets",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        choices=["mapfree", "ucl_campus_aria", "360loc_aria"],
        help="Dataset name",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        help="Split name, e.g. val / train / test",
    )
    parser.add_argument(
        "--scene", type=str, required=True,
        help="Scene name, e.g. s00460",
    )
    parser.add_argument(
        "--output_root", type=Path,
        default=Path("/Titan/dataset/data_opennavmap/map_free_eval/viz_mapfree_rpe_pose"),
        help="Root directory for output images",
    )
    parser.add_argument("--n_refs", type=int, default=4,
                        help="Number of reference images to show")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reference image sampling")
    args = parser.parse_args()

    scene_dir = resolve_scene_dir(args.dataset_root, args.dataset_name, args.split, args.scene)
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    output_path = args.output_root / args.dataset_name / args.split / f"{args.scene}_seed{args.seed}.png"
    visualize_scene(scene_dir, output_path, n_refs=args.n_refs, seed=args.seed)


if __name__ == "__main__":
    main()
