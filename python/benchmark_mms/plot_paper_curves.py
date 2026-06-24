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
LW = 3.0
MS = 10.5

ENV_CFG = [
    {
        "label":    "Office",
        "color":    _PALETTE[0],
        "marker":   _MARKERS[0],
        "baseline": "duplex_office",
        "dynamic":  "duplex_office_daychange",
    },
    {
        "label":    "Maze",
        "color":    _PALETTE[1],
        "marker":   _MARKERS[1],
        "baseline": "octa_maze",
        "dynamic":  "octa_maze_daychange",
    },
    {
        "label":    "Tunnel",
        "color":    _PALETTE[3],
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
    """White background, black axes, dashed grid."""
    ax.set_facecolor("white")
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.grid(True, linestyle="--", alpha=0.7)


def _draw_coverage(ax, xs: list[int], cum_pct: list[float],
                   color, marker: str, linestyle: str, label: str) -> None:
    ax.plot(xs, cum_pct, color=color, linestyle=linestyle, linewidth=LW,
            marker=marker, markersize=MS, label=label, zorder=3)


def _draw_ratio(ax, xs: list[int], ratios: list[float],
                color, marker: str, linestyle: str, label: str) -> None:
    """Plot finite values as a curve; mark inf values at NA_Y_RAT with ×.

    A connector line links the inf × markers to the adjacent finite point so
    the curve reads as continuous.
    """
    finite_x, finite_y, inf_x = [], [], []
    for x, r in zip(xs, ratios):
        if np.isinf(r):
            inf_x.append(x)
        else:
            finite_x.append(x)
            finite_y.append(r)

    # Draw × markers for unreachable sessions
    if inf_x:
        ax.plot(inf_x, [NA_Y_RAT] * len(inf_x),
                color=color, linestyle=linestyle, linewidth=LW,
                marker="x", markersize=MS + 2, markeredgewidth=2.2,
                zorder=5, clip_on=False, label=label if not finite_x else None)

    # Connect last inf point to first finite point (or first inf to last finite)
    if inf_x and finite_x:
        if inf_x[-1] < finite_x[0]:
            # inf comes before finite (Dynamic variant: sessions 1-2 blocked)
            connect_x = [inf_x[-1], finite_x[0]]
            connect_y = [NA_Y_RAT,  finite_y[0]]
        else:
            # finite comes before inf
            connect_x = [finite_x[-1], inf_x[0]]
            connect_y = [finite_y[-1],  NA_Y_RAT]
        ax.plot(connect_x, connect_y, color=color, linestyle=linestyle,
                linewidth=LW * 0.8, alpha=0.7, zorder=2)

    if finite_x:
        ax.plot(finite_x, finite_y, color=color, linestyle=linestyle, linewidth=LW,
                marker=marker, markersize=MS, label=label, zorder=3)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setting_font(fontsize=22, titlesize=31, legend_fontsize=18)

    fig, (ax_cov, ax_rat) = plt.subplots(
        1, 2, figsize=(16, 5), facecolor="white"
    )
    fig.subplots_adjust(wspace=0.26)

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
        # _draw_coverage(ax_cov, xs_dyn, dyn_cov[:X_MAX],
        #                env["color"], env["marker"], _LINESTYLES[1],
        #                env["label"] + " Dynamic")

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
    ax_cov.set_ylim(0, 100)
    ax_cov.set_yticks([0, 20, 40, 60, 80, 100])
    ax_cov.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax_cov.set_xlabel("Session Number", fontsize=22)
    ax_cov.set_ylabel(r"Coverage [\%]", fontsize=22)
    ax_cov.text(2.58, 98, "Remove\nObstacles", color="#6B7280", fontsize=18,
                va="top", rotation=0, alpha=0.9)
    _style_ax(ax_cov)

    # ------------------------------------------------- ratio panel
    ax_rat.axhline(1.0, color='k', linestyle=_LINESTYLES[1],
                   linewidth=1.5, zorder=1)
    ax_rat.set_xlim(0.5, X_MAX + 0.5)
    ax_rat.set_xticks(range(1, X_MAX + 1))
    ax_rat.set_ylim(0.88, NA_Y_RAT + 0.06)
    # Build y-ticks: regular ticks + "N/A" at NA_Y_RAT
    rat_ticks      = [1.0, 1.2, 1.4, 1.6, NA_Y_RAT]
    rat_ticklabels = ["1.0", "1.2", "1.4", "1.6", "N/A"]
    ax_rat.set_yticks(rat_ticks)
    ax_rat.set_yticklabels(rat_ticklabels)
    # Dashed separator above 1.6 to visually isolate N/A row
    ax_rat.axhline(NA_Y_RAT - 0.04, color="#D1D5DB", linestyle="--",
                   linewidth=0.8, zorder=0)
    ax_rat.set_xlabel("Session Number", fontsize=22)
    ax_rat.set_ylabel("Optimality Ratio", fontsize=22)
    ax_rat.text(2.58, NA_Y_RAT + 0.04, "Remove\nObstacles", color="#6B7280", fontsize=18,
                va="top", rotation=0, alpha=0.9)
    _style_ax(ax_rat)

    # ------------------------------------------------- left legend: environments
    leg_cov = [
        Line2D([0], [0], color=env["color"], marker=env["marker"], markersize=MS,
               linewidth=LW, linestyle=_LINESTYLES[0], label=env["label"])
        for env in ENV_CFG
    ]
    ax_cov.legend(handles=leg_cov, frameon=True, loc="center right", edgecolor="#D1D5DB", fontsize=18)

    # ------------------------------------------------- right legend: line styles
    leg_rat = [
        Line2D([0], [0], color="black", linewidth=LW, linestyle=_LINESTYLES[0],
               label="Baseline"),
        Line2D([0], [0], color="black", linewidth=LW, linestyle=_LINESTYLES[1],
               label="Dynamic"),
        # Line2D([0], [0], color=_PALETTE[1], linewidth=1.5, linestyle=_LINESTYLES[1],
        #        label="GT optimal ($r$=1.0)"),
    ]
    ax_rat.legend(handles=leg_rat, frameon=True, edgecolor="#D1D5DB", fontsize=18)

    out_pdf = OUTPUT_ROOT / "paper_figure_curves.pdf"
    out_png = OUTPUT_ROOT / "paper_figure_curves.png"
    fig.savefig(out_pdf, dpi=150, facecolor="white", bbox_inches="tight")
    fig.savefig(out_png, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()
