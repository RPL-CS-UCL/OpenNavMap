import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

import plot_paper_figure as ppf


def test_load_env_reads_dynamic_obstacle_metadata() -> None:
    data = ppf.load_env(
        ppf.OUTPUT_ROOT / "duplex_office_daychange" / "data",
        list(range(5)),
    )

    assert data["obstacle_block_cells"] == (50, 25, 55, 100)
    assert data["day_change_session"] == 2


def test_output_stems_distinguish_normal_and_dynamic_change() -> None:
    assert ppf.NORMAL_OUTPUT_STEM == "paper_figure_exploration_normal"
    assert ppf.DYNAMIC_OUTPUT_STEM == "paper_figure_exploration_dynamic_change"
