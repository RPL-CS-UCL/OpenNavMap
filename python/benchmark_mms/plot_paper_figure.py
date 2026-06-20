"""Plot a 3×6 paper figure from pre-computed benchmark data.

Layout (3 rows × 6 columns):
  Row 0: duplex_office  — sessions 0-4, merged k0..4
  Row 1: octa_maze      — sessions 0-4, merged k0..4
  Row 2: tunnel         — sessions 0-4, merged k0..4  (first 5 of 10)

Data is loaded purely from disk; no simulation is re-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).parent
OUTPUT_ROOT = BENCHMARK_DIR / "output"

sys.path.insert(0, str(BENCHMARK_DIR))

from frontier_explore_benchmark import (  # noqa: E402
    BG_COLOR,
    COLOR_GOAL,
    COLOR_START,
    COLOR_TOPO_EDGE_INTRA,
    COLOR_TOPO_PATH,
    COLOR_TRAJ,
    GRID_RES_M,
    MAX_TOPO_EDGES_DRAW,
    MAX_TOPO_EDGES_PER_NODE_DRAW,
    _draw_gt_path,
    _draw_path,
    _draw_topo_graph,
    _setting_font,
    astar,
    obs_to_rgb,
    topo_path_length,
)

# ---------------------------------------------------------------------------
# NPZ → networkx graph
# ---------------------------------------------------------------------------

def npz_to_graph(path: Path, node_counts: list[int] | None = None) -> nx.Graph:
    """Reconstruct a networkx Graph from a saved .npz file.

    For merged graphs, the node ids are non-contiguous (offset = sum + 1 between
    sessions).  Pass ``node_counts`` (list of per-session node counts) so the
    correct id→xy mapping can be reconstructed.  For single-session graphs the
    ids are 0-based contiguous and ``node_counts`` can be omitted.
    """
    data = np.load(path)
    G = nx.Graph()
    nodes_xy: np.ndarray = data["nodes_xy"]  # (N, 2) float32

    if node_counts is not None:
        # Reconstruct non-contiguous node ids from offset scheme
        offset = 0
        node_ids: list[int] = []
        for s in node_counts:
            node_ids.extend(range(offset, offset + s))
            offset += s + 1
    else:
        node_ids = list(range(len(nodes_xy)))

    for nid, (x, y) in zip(node_ids, nodes_xy):
        G.add_node(nid, x=float(x), y=float(y))

    if data["edges"].size > 0:
        edges: np.ndarray = data["edges"]           # (E, 2) int32
        weights: np.ndarray = data["edge_weights"]  # (E,) float32
        for (u, v), w in zip(edges, weights):
            G.add_edge(int(u), int(v), weight=float(w))

    # Optional graph-level metadata
    for key in ("start_node", "goal_node"):
        if key in data:
            G.graph[key] = int(data[key][0])
    for key in ("start_nodes", "goal_nodes"):
        if key in data:
            G.graph[key] = [int(x) for x in data[key]]

    return G


# ---------------------------------------------------------------------------
# Load environment data from disk
# ---------------------------------------------------------------------------

def load_env(data_dir: Path, sessions: list[int]) -> dict:
    """Load all data needed for one environment's panels."""
    metrics = json.loads((data_dir / "metrics.json").read_text())
    res: float = metrics["res_m"]
    start: tuple[int, int] = tuple(metrics["start"])   # (row, col)
    goal: tuple[int, int] = tuple(metrics["goal"])     # (row, col)
    gt_len: float = metrics["gt_len_m"]

    base_map: np.ndarray = np.load(data_dir / "base_map.npy")

    # GT path via A* — returns (path, length)
    gt_path, _ = astar(base_map, start, goal)

    k_max = max(sessions)

    all_obs = [np.load(data_dir / f"session_{k}_obs.npy") for k in sessions]
    all_poses = [
        [tuple(p) for p in np.load(data_dir / f"session_{k}_poses.npy")]
        for k in sessions
    ]
    subgraphs = [npz_to_graph(data_dir / f"topomap_k{k}.npz") for k in sessions]

    # merged npz uses non-contiguous node ids; pass cumulative node_counts up to k_max
    node_counts_all: list[int] = metrics["node_counts"]
    # node_counts for sessions 0..k_max (by original session index, not filtered)
    node_counts_merged = node_counts_all[: k_max + 1]
    merged_topo = npz_to_graph(
        data_dir / f"topomap_merged_k{k_max}.npz",
        node_counts=node_counts_merged,
    )
    merged_obs: np.ndarray = np.load(data_dir / f"merged_obs_k{k_max}.npy")

    return {
        "res": res,
        "start": start,
        "goal": goal,
        "gt_len": gt_len,
        "gt_path": gt_path,
        "base_map": base_map,
        "all_obs": all_obs,
        "all_poses": all_poses,
        "subgraphs": subgraphs,
        "merged_topo": merged_topo,
        "merged_obs": merged_obs,
        "sessions": sessions,
    }


# ---------------------------------------------------------------------------
# Panel drawing helpers
# ---------------------------------------------------------------------------

def _configure_ax(ax) -> None:
    ax.set_facecolor(BG_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_session_panel(
    ax,
    obs: np.ndarray,
    poses: list[tuple],
    G: nx.Graph,
    start: tuple[int, int],
    goal: tuple[int, int],
    res: float,
    session_idx: int,
    title_fontsize: int = 16,
) -> None:
    _configure_ax(ax)
    ax.imshow(obs_to_rgb(obs), origin="upper", interpolation="none")

    if poses:
        tr = np.array([(p[0], p[1]) for p in poses])
        ax.plot(tr[:, 1], tr[:, 0], color=COLOR_TRAJ, linewidth=1.2,
                alpha=0.7, zorder=3)

    reachable = "NO"
    if G.number_of_nodes() > 0:
        _draw_topo_graph(ax, G, res=res)
        tlen, sn, gn = topo_path_length(G, start, goal, res)
        if sn is not None and gn is not None:
            try:
                sp = nx.shortest_path(G, sn, gn, weight="weight")
                _draw_path(ax, sp, G, res=res)
                reachable = "YES"
            except nx.NetworkXNoPath:
                pass

    ax.scatter(start[1], start[0], color=COLOR_START, s=120, marker="o",
               edgecolors="white", linewidths=1.5, zorder=7)
    ax.scatter(goal[1], goal[0], color=COLOR_GOAL, s=120, marker="X",
               edgecolors="white", linewidths=1.5, zorder=7)

    ax.set_title(f"Session {session_idx}  reach={reachable}",
                 color="white", fontsize=title_fontsize, pad=4)


def draw_merged_panel(
    ax,
    merged_obs: np.ndarray,
    merged_topo: nx.Graph,
    subgraphs: list[nx.Graph],
    start: tuple[int, int],
    goal: tuple[int, int],
    gt_path,
    gt_len: float,
    res: float,
    k_max: int,
    title_fontsize: int = 16,
) -> None:
    _configure_ax(ax)
    ax.imshow(obs_to_rgb(merged_obs), origin="upper", interpolation="none")

    if merged_topo.number_of_nodes() > 0:
        _draw_topo_graph(ax, merged_topo, edge_color=COLOR_TOPO_EDGE_INTRA, res=res)

        tlen, sn, gn = topo_path_length(merged_topo, start, goal, res)
        if sn is not None and gn is not None:
            try:
                sp = nx.shortest_path(merged_topo, sn, gn, weight="weight")
                _draw_path(ax, sp, merged_topo, res=res)
            except nx.NetworkXNoPath:
                tlen = float("inf")
        else:
            tlen = float("inf")
    else:
        tlen = float("inf")

    ax.scatter(start[1], start[0], color=COLOR_START, s=120, marker="o",
               edgecolors="white", linewidths=1.5, zorder=7)
    ax.scatter(goal[1], goal[0], color=COLOR_GOAL, s=120, marker="X",
               edgecolors="white", linewidths=1.5, zorder=7)
    _draw_gt_path(ax, gt_path, lw=2.0, res=res)

    ratio = tlen / gt_len if gt_len > 0 and tlen < float("inf") else float("inf")
    if ratio < float("inf"):
        title = f"Merged k=0..{k_max}  ratio={ratio:.3f}"
    else:
        title = f"Merged k=0..{k_max}  unreachable"

    ax.set_title(title, color="white", fontsize=title_fontsize, pad=4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ENVS = [
    ("duplex_office", "Office",  list(range(5))),
    ("octa_maze",     "Maze",    list(range(5))),
    ("tunnel",        "Tunnel",  list(range(5))),   # only first 5 of 10
]

TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 20


def main() -> None:
    _setting_font(fontsize=14, titlesize=TITLE_FONTSIZE, legend_fontsize=13)

    n_rows = len(ENVS)
    n_cols = 6  # 5 sessions + 1 merged

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 6, n_rows * 6),
        facecolor=BG_COLOR,
    )
    fig.subplots_adjust(wspace=0.03, hspace=0.12)

    for row_i, (env_name, env_label, sessions) in enumerate(ENVS):
        print(f"Loading {env_name} ...")
        data = load_env(OUTPUT_ROOT / env_name / "data", sessions)

        for col_j, k in enumerate(sessions):
            draw_session_panel(
                axes[row_i, col_j],
                obs=data["all_obs"][col_j],
                poses=data["all_poses"][col_j],
                G=data["subgraphs"][col_j],
                start=data["start"],
                goal=data["goal"],
                res=data["res"],
                session_idx=k,
                title_fontsize=TITLE_FONTSIZE,
            )

        draw_merged_panel(
            axes[row_i, n_cols - 1],
            merged_obs=data["merged_obs"],
            merged_topo=data["merged_topo"],
            subgraphs=data["subgraphs"],
            start=data["start"],
            goal=data["goal"],
            gt_path=data["gt_path"],
            gt_len=data["gt_len"],
            res=data["res"],
            k_max=max(sessions),
            title_fontsize=TITLE_FONTSIZE,
        )

        # Row label on leftmost axis
        axes[row_i, 0].set_ylabel(
            env_label,
            fontsize=LABEL_FONTSIZE,
            fontweight="bold",
            color="white",
            labelpad=8,
        )

    # Save
    out_pdf = OUTPUT_ROOT / "paper_figure_exploration.pdf"
    out_png = OUTPUT_ROOT / "paper_figure_exploration.png"
    fig.savefig(out_pdf, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()
