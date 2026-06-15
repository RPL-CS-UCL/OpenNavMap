from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
import trimesh


def sample_mesh_to_pcd(
    mesh: trimesh.Trimesh,
    num_points: int = 100_000,
) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def write_pcd(filepath: str, points: np.ndarray) -> None:
    with open(filepath, "w") as f:
        f.write("# .PCD v0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        np.savetxt(f, points, fmt="%.6f")


def crop_height_slice(
    points: np.ndarray,
    height: float,
    tolerance: float,
) -> np.ndarray:
    y = points[:, 1]
    mask = (y >= height - tolerance) & (y <= height + tolerance)
    return points[mask]


def generate_occupancy_grid(
    points: np.ndarray,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    resolution: float = 0.1,
) -> np.ndarray:
    x_min, x_max = x_range
    z_min, z_max = z_range
    nx = int((x_max - x_min) / resolution) + 1
    nz = int((z_max - z_min) / resolution) + 1
    grid = np.zeros((nz, nx), dtype=np.uint8)
    xi = np.clip(((points[:, 0] - x_min) / resolution).astype(int), 0, nx - 1)
    zi = np.clip(((points[:, 2] - z_min) / resolution).astype(int), 0, nz - 1)
    grid[zi, xi] = 1
    return grid


def render_occupancy_image(
    grid: np.ndarray,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    resolution: float,
    filepath: str,
) -> None:
    x_min, x_max = x_range
    z_min, z_max = z_range

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_facecolor("#111827")

    extent = [x_min, x_max, z_max, z_min]
    ax.imshow(grid, cmap="binary_r", extent=extent, interpolation="none", vmin=0, vmax=1)

    ax.set_xlabel("X (m)", color="white", fontsize=12)
    ax.set_ylabel("Z (m)", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.set_aspect("equal")

    fig.tight_layout(pad=1.0)
    fig.savefig(filepath, facecolor="#111827")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert STL to PCD + occupancy grid image")
    parser.add_argument("--input", default="octa_maze.stl", help="Input STL file path")
    parser.add_argument("--output", default="octa_maze.pcd", help="Output PCD file path")
    parser.add_argument("--num_points", type=int, default=250_000, help="Number of surface sample points")
    parser.add_argument("--resolution", type=float, default=0.1, help="Occupancy grid resolution (m/cell)")
    parser.add_argument("--occ_image", default="octa_maze_occupancy.png", help="Output occupancy grid image path")
    parser.add_argument("--height_slice", type=float, default=2.0, help="Height (Y) to slice for occupancy grid (m)")
    parser.add_argument("--height_tolerance", type=float, default=0.3, help="Tolerance band around height_slice (m)")
    parser.add_argument("--dilate", type=int, default=1, help="Dilation radius (pixels) for obstacle edges, 0 to disable")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, os.path.basename(args.input)) if not os.path.isabs(args.input) else args.input

    print(f"Loading STL: {input_path}")
    mesh = trimesh.load(input_path)
    print(f"  vertices: {mesh.vertices.shape[0]}, faces: {mesh.faces.shape[0]}")
    print(f"  bounds: {mesh.bounds}")

    print(f"Sampling {args.num_points} points from mesh surface...")
    points = sample_mesh_to_pcd(mesh, args.num_points)

    pcd_path = os.path.join(script_dir, os.path.basename(args.output)) if not os.path.isabs(args.output) else args.output
    print(f"Writing PCD: {pcd_path}")
    write_pcd(pcd_path, points)

    x_range = (mesh.bounds[0, 0], mesh.bounds[1, 0])
    z_range = (mesh.bounds[0, 2], mesh.bounds[1, 2])
    occ_path = os.path.join(script_dir, os.path.basename(args.occ_image)) if not os.path.isabs(args.occ_image) else args.occ_image
    print(f"Generating occupancy grid at {args.resolution} m/cell...")
    print(f"  height slice: Y={args.height_slice} +/-{args.height_tolerance} m")
    occ_points = crop_height_slice(points, args.height_slice, args.height_tolerance)
    print(f"  points in slice: {len(occ_points)} / {len(points)}")
    grid = generate_occupancy_grid(occ_points, x_range, z_range, args.resolution)
    print(f"  grid shape: {grid.shape}")
    print(f"  occupied (raw): {grid.sum()} / {grid.size} ({100 * grid.sum() / grid.size:.1f}%)")
    if args.dilate > 0:
        grid = binary_dilation(grid, structure=np.ones((2 * args.dilate + 1, 2 * args.dilate + 1), dtype=bool)).astype(np.uint8)
        print(f"  dilate {args.dilate}px -> occupied: {grid.sum()} / {grid.size} ({100 * grid.sum() / grid.size:.1f}%)")

    print(f"Writing occupancy image: {occ_path}")
    render_occupancy_image(grid, x_range, z_range, args.resolution, occ_path)

    print("Done.")


if __name__ == "__main__":
    main()
