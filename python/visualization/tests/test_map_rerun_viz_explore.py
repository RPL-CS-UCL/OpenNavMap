"""T3.5 unit tests: P3 explore-layer additions to map_rerun_viz (frontiers + decision path)."""
from dataclasses import dataclass

import numpy as np

import visualization.map_rerun_viz as map_rerun_viz


class FakeRerun:
    """Records rr.log/set_time_seconds calls instead of writing to a real recording."""

    def __init__(self) -> None:
        self.logged = []
        self.times = []

    def set_time_seconds(self, timeline, value) -> None:
        self.times.append((timeline, value))

    @staticmethod
    def Points3D(positions, colors=None, radii=None, labels=None):
        return ("points3d", {"positions": list(positions), "colors": colors, "radii": radii, "labels": labels})

    @staticmethod
    def LineStrips3D(strips, radii=None, colors=None):
        return ("linestrips3d", {"strips": list(strips), "radii": radii, "colors": colors})

    def log(self, entity_path, archetype, **kwargs) -> None:
        self.logged.append((entity_path, archetype))


@dataclass
class _Frontier:
    frontier_id: int
    position: tuple
    region_size: int


def test_log_frontier_candidates_logs_points_with_labels(monkeypatch) -> None:
    fake = FakeRerun()
    monkeypatch.setattr(map_rerun_viz, "rr", fake)
    frontiers = [_Frontier(0, (1.0, 2.0, 0.0), 8), _Frontier(1, (3.0, 4.0, 0.0), 20)]

    map_rerun_viz.log_frontier_candidates(frontiers, timestamp=1.5)

    assert fake.times == [(map_rerun_viz._TIMELINE, 1.5)]
    assert len(fake.logged) == 1
    entity_path, (kind, kwargs) = fake.logged[0]
    assert entity_path == "map/explore/frontiers"
    assert kind == "points3d"
    assert [tuple(p) for p in kwargs["positions"]] == [(1.0, 2.0, 0.0), (3.0, 4.0, 0.0)]
    assert kwargs["labels"] == ["id=0 size=8", "id=1 size=20"]


def test_log_frontier_candidates_empty_list_does_not_log(monkeypatch) -> None:
    fake = FakeRerun()
    monkeypatch.setattr(map_rerun_viz, "rr", fake)

    map_rerun_viz.log_frontier_candidates([], timestamp=1.0)

    assert fake.logged == []
    assert fake.times == []


def test_log_decision_path_colors_segments_by_action(monkeypatch) -> None:
    fake = FakeRerun()
    monkeypatch.setattr(map_rerun_viz, "rr", fake)
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    actions = ["go_to_frontier", "go_to_frontier", "go_to_target"]

    map_rerun_viz.log_decision_path(positions, actions, timestamp=2.0)

    assert fake.times == [(map_rerun_viz._TIMELINE, 2.0)]
    assert len(fake.logged) == 1
    entity_path, (kind, kwargs) = fake.logged[0]
    assert entity_path == "map/explore/decision_path"
    assert kind == "linestrips3d"
    assert len(kwargs["strips"]) == 2
    colors = kwargs["colors"]
    assert list(colors[0]) == map_rerun_viz._PATH_EXPLORE_COLOR
    assert list(colors[1]) == map_rerun_viz._PATH_CONVERGE_COLOR


def test_log_decision_path_too_few_positions_does_not_log(monkeypatch) -> None:
    fake = FakeRerun()
    monkeypatch.setattr(map_rerun_viz, "rr", fake)

    map_rerun_viz.log_decision_path([(0.0, 0.0, 0.0)], ["go_to_frontier"], timestamp=1.0)

    assert fake.logged == []


def test_visualize_map_appends_explore_layers_when_data_given(monkeypatch, tmp_path) -> None:
    logged_frontiers = []
    logged_path = []
    monkeypatch.setattr(map_rerun_viz, "log_frontier_candidates", lambda f, t: logged_frontiers.append((f, t)))
    monkeypatch.setattr(map_rerun_viz, "log_decision_path", lambda p, a, t: logged_path.append((p, a, t)))
    monkeypatch.setattr(map_rerun_viz, "log_world_frame_axes", lambda **kw: None)
    monkeypatch.setattr(map_rerun_viz, "log_map_objects", lambda m: None)
    monkeypatch.setattr(map_rerun_viz, "log_object_visibility_edges", lambda m: None)

    fake = FakeRerun()
    fake.init = lambda *a, **kw: None
    fake.send_blueprint = lambda *a, **kw: None
    fake.save = lambda *a, **kw: None
    monkeypatch.setattr(map_rerun_viz, "rr", fake)
    monkeypatch.setattr(map_rerun_viz.rrb, "Blueprint", lambda *a, **kw: None)

    class _Manager:
        covis = None
        graphs = {}

    frontiers = [_Frontier(0, (1.0, 0.0, 0.0), 5)]
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    actions = ["go_to_frontier", "go_to_frontier"]

    map_rerun_viz.visualize_map(
        _Manager(), str(tmp_path / "out.rrd"), frontiers=frontiers, decision_positions=positions, decision_actions=actions
    )

    assert logged_frontiers == [(frontiers, 0.0)]
    assert logged_path == [(positions, actions, 0.0)]


def test_visualize_map_skips_explore_layers_when_no_data(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(map_rerun_viz, "log_frontier_candidates", lambda f, t: calls.append("frontiers"))
    monkeypatch.setattr(map_rerun_viz, "log_decision_path", lambda p, a, t: calls.append("path"))
    monkeypatch.setattr(map_rerun_viz, "log_world_frame_axes", lambda **kw: None)
    monkeypatch.setattr(map_rerun_viz, "log_map_objects", lambda m: None)
    monkeypatch.setattr(map_rerun_viz, "log_object_visibility_edges", lambda m: None)

    fake = FakeRerun()
    fake.init = lambda *a, **kw: None
    fake.send_blueprint = lambda *a, **kw: None
    fake.save = lambda *a, **kw: None
    monkeypatch.setattr(map_rerun_viz, "rr", fake)
    monkeypatch.setattr(map_rerun_viz.rrb, "Blueprint", lambda *a, **kw: None)

    class _Manager:
        covis = None
        graphs = {}

    map_rerun_viz.visualize_map(_Manager(), str(tmp_path / "out.rrd"))

    assert calls == []
