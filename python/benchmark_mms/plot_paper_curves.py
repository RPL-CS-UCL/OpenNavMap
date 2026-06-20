"""Plot optimality-ratio and coverage curves for the multi-session benchmark.

Layout: 1 row × 2 columns
  Left:  Cumulative free-space coverage [%]
  Right: Topometric shortest-path ratio r = topo_len / GT_len

All three environments (Office / Maze / Tunnel) are overlaid in each panel.
Baseline = solid line; Dynamic = dashed line.
White background, black text. Unreachable sessions shown as × at a dedicated
"N/A" y-tick level.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

BENCHMARK_DIR = Path(__file__).parent
OUTPUT_ROOT   = BENCHMARK_DIR / "output"

sys.path.insert(0, str(BENCHMARK_DIR))
from frontier_explore_benchmark import (  # noqa: E402
    _LINESTYLES,
    _MARKERS,
    _PALETTE,
    _setting_font,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
LW = 2.0
MS = 7

ENV_CFG = [
    {
        "label":    "Office",
        "color":    _PALETTE[3],   # blue
        "marker":   _MARKERS[0],
        "baseline": "duplex_office",
        "dynamic":  "duplex_office_daychange",
    },
    {
        "label":    "Maze",
        "color":    _PALETTE[0],   # green
        "marker":   _MARKERS[1],
        "baseline": "octa_maze",
        "dynamic":  "octa_maze_daychange",
    },
    {
        "label":    "Tunnel",
        "color":    _PALETTE[7],   # orange
        "marker":   _MARKERS[2],
        "baseline": "tunnel",
        "dynamic":  "tunnel_daychange",
    },
]

# Y position reserved for "N/A" (unreachable) markers in the ratio panel.
# Must sit above the normal data range (0.9–1.65) with a visible gap.
NA_Y_RAT = 1.72   # ratio panel N/A level
NA_Y_COV = 108.0  # coverage panel N/A level (above 100 %)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(env_name: str) -> tuple[list[float], list[float]]:
    d = OUTPUT_ROOT / env_name / "data"
    ratios  = [float(v) for v in json.loads((d / "ratios.json").read_text())["ratios"]]
    cum_pct = json.loads((d / "coverage.json").read_text())["cum_pct"]
    return ratios, cum_pct

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _style_ax(ax) -> None:
    """White background, black axes."""
    ax.set_facecolor("white")
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")


def _draw_coverage(ax, xs: list[int], cum_pct: list[float],
                   color, marker: str, linestyle: str, label: str) -> None:
    ax.plot(xs, cum_pct, color=color, linestyle=linestyle, linewidth=LW,
            marker=marker, markersize=MS, label=label, zorder=3)


def _draw_ratio(ax, xs: list[int], ratios: list[float],
                color, marker: str, linestyle: str, label: str) -> None:
    """Plot finite values as a curve; mark inf values at NA_Y_RAT with ×."""
    finite_x, finite_y, inf_x = [], [], []
    for x, r in zip(xs, ratios):
        if np.isinf(r):
            inf_x.append(x)
        else:
            finite_x.append(x)
            finite_y.append(r)

    if finite_x:
        ax.plot(finite_x, finite_y, color=color, linestyle=linestyle, linewidth=LW,
                marker=marker, markersize=MS, label=label, zorder=3)
    if inf_x:
        ax.scatter(inf_x, [NA_Y_RAT] * len(inf_x),
                   marker="x", color=color, s=(MS * 2) ** 2,
                   linewidths=2.2, zorder=5, clip_on=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setting_font(fontsize=12, titlesize=14, legend_fontsize=10)

    fig, (ax_cov, ax_rat) = plt.subplots(
        1, 2, figsize=(14, 5), facecolor="white"
    )
    fig.subplots_adjust(wspace=0.30, top=0.78)

    X_MAX = 5  # show only first 5 sessions for all environments

    # ------------------------------------------------------------------ draw
    for env in ENV_CFG:
        base_ratios, base_cov = _load(env["baseline"])
        dyn_ratios,  dyn_cov  = _load(env["dynamic"])

        xs_base = list(range(1, min(len(base_ratios), X_MAX) + 1))
        xs_dyn  = list(range(1, min(len(dyn_ratios),  X_MAX) + 1))

        _draw_coverage(ax_cov, xs_base, base_cov[:X_MAX],
                       env["color"], env["marker"], _LINESTYLES[0],
                       env["label"] + " Baseline")
        _draw_coverage(ax_cov, xs_dyn, dyn_cov[:X_MAX],
                       env["color"], env["marker"], _LINESTYLES[1],
                       env["label"] + " Dynamic")

        _draw_ratio(ax_rat, xs_base, base_ratios[:X_MAX],
                    env["color"], env["marker"], _LINESTYLES[0],
                    env["label"] + " Baseline")
        _draw_ratio(ax_rat, xs_dyn, dyn_ratios[:X_MAX],
                    env["color"], env["marker"], _LINESTYLES[1],
                    env["label"] + " Dynamic")

    # ------------------------------------------------- day-change vertical line
    for ax in (ax_cov, ax_rat):
        ax.axvline(x=2.5, color="#6B7280", linestyle=_LINESTYLES[1],
                   linewidth=1.2, alpha=0.8, zorder=1)

    # ------------------------------------------------- coverage panel
    ax_cov.axhline(100.0, color="#6B7280", linestyle=":", linewidth=1.2,
                   alpha=0.7, zorder=1)
    ax_cov.set_xlim(0.5, X_MAX + 0.5)
    ax_cov.set_xticks(range(1, X_MAX + 1))
    ax_cov.set_ylim(0, 108)
    ax_cov.set_yticks([0, 20, 40, 60, 80, 100])
    ax_cov.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax_cov.set_xlabel("Number of Sessions (k)", fontsize=12)
    ax_cov.set_ylabel(r"Cumulative Coverage [\%]", fontsize=12)
    ax_cov.set_title("Coverage Growth", fontsize=14)
    # day-change label
    ax_cov.text(2.58, 2, "obstacle\nremoved", color="#6B7280", fontsize=8,
                va="bottom", rotation=90, alpha=0.9)
    _style_ax(ax_cov)

    # ------------------------------------------------- ratio panel
    ax_rat.axhline(1.0, color=_PALETTE[1], linestyle=_LINESTYLES[1],
                   linewidth=1.5, zorder=1)
    ax_rat.set_xlim(0.5, X_MAX + 0.5)
    ax_rat.set_xticks(range(1, X_MAX + 1))
    ax_rat.set_ylim(0.88, NA_Y_RAT + 0.06)
    # Build y-ticks: regular ticks + "N/A" at NA_Y_RAT
    rat_ticks      = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, NA_Y_RAT]
    rat_ticklabels = ["0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "N/A"]
    ax_rat.set_yticks(rat_ticks)
    ax_rat.set_yticklabels(rat_ticklabels)
    # Dashed separator above 1.6 to visually isolate N/A row
    ax_rat.axhline(NA_Y_RAT - 0.04, color="#D1D5DB", linestyle="--",
                   linewidth=0.8, zorder=0)
    ax_rat.set_xlabel("Number of Sessions (k)", fontsize=12)
    ax_rat.set_ylabel("Optimality Ratio (topo / GT)", fontsize=12)
    ax_rat.set_title("Path Optimality", fontsize=14)
    ax_rat.text(2.58, 0.91, "obstacle\nremoved", color="#6B7280", fontsize=8,
                va="bottom", rotation=90, alpha=0.9)
    _style_ax(ax_rat)

    # ------------------------------------------------- shared legend (top)
    legend_elements = []
    for env in ENV_CFG:
        legend_elements.append(
            Line2D([0], [0], color=env["color"], linestyle=_LINESTYLES[0],
                   linewidth=LW, marker=env["marker"], markersize=MS,
                   label=env["label"] + " Baseline")
        )
        legend_elements.append(
            Line2D([0], [0], color=env["color"], linestyle=_LINESTYLES[1],
                   linewidth=LW, marker=env["marker"], markersize=MS,
                   label=env["label"] + " Dynamic")
        )
    legend_elements.append(
        Line2D([0], [0], color=_PALETTE[1], linestyle=_LINESTYLES[1],
               linewidth=1.5, label="GT optimal ($r$=1.0)")
    )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )

    out_pdf = OUTPUT_ROOT / "paper_figure_curves.pdf"
    out_png = OUTPUT_ROOT / "paper_figure_curves.png"
    fig.savefig(out_pdf, dpi=150, facecolor="white", bbox_inches="tight")
    fig.savefig(out_png, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()
