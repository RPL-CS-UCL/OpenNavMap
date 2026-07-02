from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
import trimesh

HEADER_END = b"DATA" if hasattr(b"", "decode") else None


def _parse_pcd_header(filepath: str) -> tuple[int, bool]:
    with open(filepath, "rb") as f:
        raw = f.read(8192)

    header_lines = []
    data_start = 0
    for i in range(0, len(raw)):
        if raw[i:i + 1] == b"\n":
            line = raw[data_start:i].decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line.startswith("DATA"):
                return i + 1, line.endswith("binary")
            data_start = i + 1

    raise ValueError("Could not find DATA line in PCD header")


def read_pcd(filepath: str) -> np.ndarray:
    data_offset, is_binary = _parse_pcd_header(filepath)
    if is_binary:
        with open(filepath, "rb") as f:
            f.seek(data_offset)
            raw = f.read()
        arr = np.frombuffer(raw, dtype=np.float32)
        return arr.reshape(-1, 3).copy()
    else:
        raw = np.loadtxt(filepath, skiprows=11, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 3)
        return raw[:, :3]


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
    z = points[:, 2]
    mask = (z >= height - tolerance) & (z <= height + tolerance)
    return points[mask]


def generate_occupancy_grid(
    points: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float = 0.2,
) -> np.ndarray:
    x_min, x_max = x_range
    y_min, y_max = y_range
    nx = int((x_max - x_min) / resolution) + 1
    ny = int((y_max - y_min) / resolution) + 1
    grid = np.zeros((ny, nx), dtype=np.uint8)
    xi = np.clip(((points[:, 0] - x_min) / resolution).astype(int), 0, nx - 1)
    yi = np.clip(((points[:, 1] - y_min) / resolution).astype(int), 0, ny - 1)
    grid[yi, xi] = 1
    return grid


def render_occupancy_image(
    grid: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    filepath: str,
    start_world: tuple[float, float] | None = None,
    goal_world: tuple[float, float] | None = None,
) -> None:
    x_min, x_max = x_range
    y_min, y_max = y_range

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_facecolor("#111827")

    extent = [x_min, x_max, y_max, y_min]
    ax.imshow(grid, cmap="binary_r", extent=extent, interpolation="none", vmin=0, vmax=1)

    marker_size = max(6, min(20, int(5.0 / resolution)))
    if start_world is not None:
        ax.plot(start_world[0], start_world[1], "o",
                color="#22d3ee", markersize=marker_size, markeredgewidth=2,
                markerfacecolor="none", label="start")
    if goal_world is not None:
        ax.plot(goal_world[0], goal_world[1], "x",
                color="#f97316", markersize=marker_size, markeredgewidth=2,
                label="goal")
    if start_world is not None or goal_world is not None:
        ax.legend(loc="upper right", fontsize=9,
                  facecolor="#1f2937", edgecolor="#6b7280", labelcolor="white")

    ax.set_xlabel("X [m]", color="white", fontsize=12)
    ax.set_ylabel("Y [m]", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.set_aspect("equal")

    fig.tight_layout(pad=1.0)
    fig.savefig(filepath, facecolor="#111827")
    plt.close(fig)


def _compact_range(vals: np.ndarray, pad: float = 0.0) -> tuple[float, float]:
    return (float(vals.min()) - pad, float(vals.max()) + pad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate occupancy grid from STL or PCD")
    parser.add_argument("--input", default="octa_maze.stl", help="Input STL or PCD file path")
    parser.add_argument("--output", default="octa_maze.pcd", help="Output PCD path (only used when input is STL)")
    parser.add_argument("--num_points", type=int, default=250_000,
                        help="Surface sample points (only used when input is STL)")
    parser.add_argument("--resolution", type=float, default=0.2, help="Occupancy grid resolution (m/cell)")
    parser.add_argument("--height_slice", type=float, default=1.5, help="Height (Z) to slice for occupancy grid (m)")
    parser.add_argument("--height_tolerance", type=float, default=0.1, help="Tolerance band around height_slice (m)")
    parser.add_argument("--dilate", type=int, default=1, help="Dilation radius (pixels) for obstacle edges, 0 to disable")
    parser.add_argument("--start", type=float, nargs=2, default=None, metavar=("COL_M", "ROW_M"),
                        help="Start position in world coords (col_m row_m) to overlay on image")
    parser.add_argument("--goal", type=float, nargs=2, default=None, metavar=("COL_M", "ROW_M"),
                        help="Goal position in world coords (col_m row_m) to overlay on image")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, os.path.basename(args.input)) if not os.path.isabs(args.input) else args.input

    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    occ_path = os.path.join(script_dir, f"{input_stem}_occupancy.png")

    ext = os.path.splitext(input_path)[1].lower()
    is_pcd = ext == ".pcd"

    if is_pcd:
        print(f"Reading PCD: {input_path}")
        points = read_pcd(input_path)
        print(f"  points: {len(points)}")
        x_range = _compact_range(points[:, 0])
        y_range = _compact_range(points[:, 1])
    else:
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
        y_range = (mesh.bounds[0, 1], mesh.bounds[1, 1])

    print(f"Generating occupancy grid at {args.resolution} m/cell...")
    print(f"  height slice: Z={args.height_slice} +/-{args.height_tolerance} m")
    occ_points = crop_height_slice(points, args.height_slice, args.height_tolerance)
    print(f"  points in slice: {len(occ_points)} / {len(points)}")
    grid = generate_occupancy_grid(occ_points, x_range, y_range, args.resolution)
    print(f"  grid shape: {grid.shape}")
    print(f"  occupied (raw): {grid.sum()} / {grid.size} ({100 * grid.sum() / grid.size:.1f}%)")
    if args.dilate > 0:
        grid = binary_dilation(grid, structure=np.ones((2 * args.dilate + 1, 2 * args.dilate + 1), dtype=bool)).astype(np.uint8)
        print(f"  dilate {args.dilate}px -> occupied: {grid.sum()} / {grid.size} ({100 * grid.sum() / grid.size:.1f}%)")

    print(f"Writing occupancy image: {occ_path}")
    start_world = tuple(args.start) if args.start is not None else None
    goal_world  = tuple(args.goal)  if args.goal  is not None else None
    render_occupancy_image(grid, x_range, y_range, args.resolution, occ_path,
                           start_world=start_world, goal_world=goal_world)

    print("Done.")


if __name__ == "__main__":
    main()
