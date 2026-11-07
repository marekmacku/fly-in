"""Unit tests for fly_in2.pathfinding."""

from fly_in2.model import Graph
from fly_in2.parser import parse_map
from fly_in2.pathfinding import dijkstra, shortest_path


def _graph(text: str) -> Graph:
    return parse_map(text)


COSTS_MAP = """
nb_drones: 1
start_hub: s 0 0
end_hub: e 5 0
hub: r 2 0 [zone=restricted]
hub: wall 3 0 [zone=blocked]
hub: detour 1 1
connection: s-r
connection: r-e
connection: s-wall
connection: wall-e
connection: s-detour
connection: detour-e
"""

TIE_BREAK_MAP = """
nb_drones: 1
start_hub: s 0 0
end_hub: e 4 0
hub: n1 1 1
hub: n2 2 1
hub: p1 1 -1 [zone=priority]
hub: p2 2 -1 [zone=priority]
connection: s-n1
connection: n1-n2
connection: n2-e
connection: s-p1
connection: p1-p2
connection: p2-e
"""

UNREACHABLE_MAP = """
nb_drones: 1
start_hub: s 0 0
end_hub: e 9 9
hub: wall 1 0 [zone=blocked]
connection: s-wall
connection: wall-e
"""


def test_distances_are_exact_integer_costs() -> None:
    """Entering a restricted zone costs 2; normal and priority cost 1."""
    result = dijkstra(_graph(COSTS_MAP), "s")
    assert result.distances["s"] == 0
    assert result.distances["r"] == 2      # restricted entry
    assert result.distances["detour"] == 1
    assert result.distances["e"] == 2      # s -> detour -> e


def test_blocked_zones_are_never_entered() -> None:
    """A blocked zone is absent from distances even if connected."""
    result = dijkstra(_graph(COSTS_MAP), "s")
    assert "wall" not in result.distances


def test_unreachable_target_is_absent_not_infinite() -> None:
    """Unreachable zones are simply missing; no crash, no INF value."""
    result = dijkstra(_graph(UNREACHABLE_MAP), "s")
    assert "e" not in result.distances
    assert result.reconstruct("e") is None
    assert shortest_path(_graph(UNREACHABLE_MAP), "s", "e") is None


def test_shortest_path_reconstruction() -> None:
    """reconstruct walks previous links back to the source."""
    path = shortest_path(_graph(COSTS_MAP), "s", "e")
    assert path == ["s", "detour", "e"]


def test_equal_cost_tie_prefers_priority_zones() -> None:
    """Among equal-cost routes, the one through priority zones wins."""
    path = shortest_path(_graph(TIE_BREAK_MAP), "s", "e")
    assert path == ["s", "p1", "p2", "e"]


def test_tie_break_never_corrupts_distances() -> None:
    """The tie-break picks the route, not the cost: both routes cost 3."""
    result = dijkstra(_graph(TIE_BREAK_MAP), "s")
    assert result.distances["n2"] == 2
    assert result.distances["p2"] == 2
    assert result.distances["e"] == 3


def test_unknown_source_returns_empty_result() -> None:
    """A source that is not a zone yields no distances, no crash."""
    result = dijkstra(_graph(COSTS_MAP), "ghost")
    assert result.distances == {}
    assert result.reconstruct("e") is None
