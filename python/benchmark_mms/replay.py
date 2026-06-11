#!/usr/bin/env python
"""Lightweight matplotlib replay for MMS benchmark results."""
import sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.utils.utils_setting_color_font import acquire_color_palette, setting_font

STYLE_FREE = np.array([243, 244, 246]) / 255.0
STYLE_OBS = np.array([31, 41, 55]) / 255.0
STYLE_UNK = np.array([107, 114, 128]) / 255.0
BG_COLOR = "#111827"
PALETTE = acquire_color_palette()
N_SESSIONS = 10


def load_data(data_dir):
    base = np.load(data_dir / "base_map.npy")
    sessions = []
    for k in range(1, N_SESSIONS + 1):
        sessions.append({
            "poses": np.load(data_dir / f"session_{k:02d}_poses.npy"),
            "obs": np.load(data_dir / f"session_{k:02d}_obs.npy").astype(np.int8),
            "merged": np.load(data_dir / f"merged_obs_k{k:02d}.npy").astype(np.int8),
        })
    with open(data_dir / "metrics.json") as f:
        metrics = json.load(f)
    return base, sessions, metrics


def replay(mode, k_stop=None, speed=1.0, save_gif=False):
    setting_font(fontsize=14, titlesize=16, legend_fontsize=12)
    data_dir = Path(__file__).resolve().parent / "output" / mode / "data"
    base, sessions, metrics = load_data(data_dir)

    if k_stop is None:
        k_stop = len(sessions)
    else:
        k_stop = min(k_stop, len(sessions))

    fig, (ax_map, ax_metrics) = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG_COLOR)
    interval = int(200 / speed)

    frames_idx = []
    for k in range(k_stop):
        for i in range(0, len(sessions[k]["poses"]), 10):
            frames_idx.append((k, i))

    def update(frame_info):
        k, i = frame_info
        ax_map.clear(); ax_metrics.clear()
        ax_map.set_facecolor(BG_COLOR); ax_metrics.set_facecolor(BG_COLOR)

        merged = sessions[k]["merged"]
        rgb = np.zeros((*merged.shape, 3))
        rgb[merged == -1] = STYLE_FREE * 0.6
        rgb[merged == 1] = STYLE_OBS * 0.6
        rgb[merged == 0] = STYLE_UNK * 0.4
        ax_map.imshow(rgb, origin="upper")

        r, c, yaw = sessions[k]["poses"][i]
        ax_map.scatter(c, r, color="yellow", s=40, zorder=5)
        traj = sessions[k]["poses"][:i + 1]
        ax_map.plot(traj[:, 1], traj[:, 0], "-", color=PALETTE[k % len(PALETTE)], linewidth=1.0)

        ax_map.set_title(f"Session {k+1}, Frame {i}", color="white")
        ax_map.axis("off")

        ks = list(range(1, k + 2))
        ax_metrics.plot(ks, [metrics["reachable_ratios"][j] * 100 for j in range(k + 1)],
                       "-o", color=PALETTE[0], linewidth=2, markersize=6, label="Reach%")
        mr = [metrics["metric_ratios"][j] for j in range(k + 1)]
        mr_plot = [m if m > 0.01 else None for m in mr]
        ax_metrics.plot(ks, mr_plot, "-^", color=PALETTE[1], linewidth=2, markersize=6, label="MetricRatio")
        ax_metrics.set_xlabel("Cumulative k", color="white")
        ax_metrics.set_ylabel("% or ratio", color="white")
        ax_metrics.set_title("Metrics", color="white")
        ax_metrics.legend(facecolor=BG_COLOR, edgecolor="white", labelcolor="white")
        ax_metrics.tick_params(colors="white")
        ax_metrics.grid(ls="--", alpha=0.3, color="white")
        return [ax_map, ax_metrics]

    ani = animation.FuncAnimation(fig, update, frames=frames_idx, interval=interval, blit=False, repeat=True)

    if save_gif:
        gif_path = data_dir.parent / "replay.gif"
        ani.save(str(gif_path), writer="pillow", fps=5)
        print(f"GIF saved to {gif_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMS Replay")
    parser.add_argument("--mode", type=str, required=True, choices=["synthetic", "osm"])
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--save_gif", action="store_true")
    args = parser.parse_args()
    replay(args.mode, args.k, args.speed, args.save_gif)
