"""Tests for the Fly-in project."""

from __future__ import annotations

from pathlib import Path

import pytest

from fly_in.errors import ParseError
from fly_in.parser import parse_map, parse_map_file
from fly_in.pathfinding import dijkstra, shortest_path
from fly_in.model import ZoneType
from fly_in.planner import plan
from fly_in.simulator import Simulator
from fly_in.validator import assert_valid

MAPS_ROOT = Path(__file__).resolve().parent.parent / "maps_download"


# --- Parser tests ---------------------------------------------------------

def test_parse_linear_path() -> None:
    graph = parse_map_file(str(MAPS_ROOT / "easy" / "01_linear_path.txt"))
    assert graph.num_drones == 2
    assert graph.start is not None
    assert graph.end is not None
    assert graph.start.name == "start"
    assert graph.end.name == "goal"
    assert len(graph.zones) == 4
    assert len(graph.connections) == 3


def test_parse_priority_puzzle() -> None:
    path = MAPS_ROOT / "medium" / "03_priority_puzzle.txt"
    graph = parse_map_file(str(path))
    slow = graph.zones["slow_path1"]
    assert slow.zone_type is ZoneType.RESTRICTED
    fast = graph.zones["fast_junction"]
    assert fast.zone_type is ZoneType.PRIORITY
    assert fast.max_drones == 2


def test_parse_error_duplicate_connection() -> None:
    text = (
        "nb_drones: 1\n"
        "start_hub: a 0 0\n"
        "end_hub: b 1 0\n"
        "connection: a-b\n"
        "connection: b-a\n"
    )
    with pytest.raises(ParseError):
        parse_map(text)


def test_parse_error_invalid_zone_type() -> None:
    text = (
        "nb_drones: 1\n"
        "start_hub: a 0 0\n"
        "hub: c 1 0 [zone=fast]\n"
        "end_hub: b 2 0\n"
        "connection: a-c\n"
        "connection: c-b\n"
    )
    with pytest.raises(ParseError):
        parse_map(text)


def test_parse_error_missing_nb_drones() -> None:
    text = "start_hub: a 0 0\nend_hub: b 1 0\nconnection: a-b\n"
    with pytest.raises(ParseError):
        parse_map(text)


def test_parse_error_negative_capacity() -> None:
    text = (
        "nb_drones: 1\n"
        "start_hub: a 0 0\n"
        "hub: c 1 0 [max_drones=0]\n"
        "end_hub: b 2 0\n"
        "connection: a-c\n"
        "connection: c-b\n"
    )
    with pytest.raises(ParseError):
        parse_map(text)


def test_parse_error_unknown_definition() -> None:
    text = (
        "nb_drones: 1\n"
        "start_hub: a 0 0\n"
        "end_hub: b 1 0\n"
        "connection: a-b\n"
        "weird: something\n"
    )
    with pytest.raises(ParseError):
        parse_map(text)


def test_parse_error_dash_in_name() -> None:
    text = (
        "nb_drones: 1\n"
        "start_hub: a-b 0 0\n"
        "end_hub: c 1 0\n"
        "connection: a-b-c\n"
    )
    with pytest.raises(ParseError):
        parse_map(text)


def test_parse_error_missing_end() -> None:
    text = "nb_drones: 1\nstart_hub: a 0 0\nhub: b 1 0\nconnection: a-b\n"
    with pytest.raises(ParseError):
        parse_map(text)


def test_parse_error_nb_drones_not_first() -> None:
    text = (
        "start_hub: a 0 0\n"
        "nb_drones: 1\n"
        "end_hub: b 1 0\n"
        "connection: a-b\n"
    )
    with pytest.raises(ParseError):
        parse_map(text)


# --- Pathfinding tests ----------------------------------------------------

def test_dijkstra_simple_graph() -> None:
    graph = parse_map_file(str(MAPS_ROOT / "easy" / "01_linear_path.txt"))
    result = dijkstra(graph, "start")
    assert result.distances["goal"] == 3.0
    path = shortest_path(graph, "start", "goal")
    assert path == ["start", "waypoint1", "waypoint2", "goal"]


def test_dijkstra_avoids_blocked() -> None:
    text = (
        "nb_drones: 1\n"
        "start_hub: a 0 0\n"
        "hub: block 1 0 [zone=blocked]\n"
        "hub: detour 0 1\n"
        "end_hub: b 1 1\n"
        "connection: a-block\n"
        "connection: a-detour\n"
        "connection: block-b\n"
        "connection: detour-b\n"
    )
    graph = parse_map(text)
    path = shortest_path(graph, "a", "b")
    assert path == ["a", "detour", "b"]


# --- Simulation validity tests -------------------------------------------

def _all_maps() -> list[Path]:
    return sorted(MAPS_ROOT.rglob("*.txt"))


@pytest.mark.parametrize("map_path", _all_maps())
def test_simulation_valid_for_all_maps(map_path: Path) -> None:
    graph = parse_map_file(str(map_path))
    drones = plan(graph)
    assert_valid(graph, drones)


@pytest.mark.parametrize("map_path", _all_maps())
def test_simulation_meets_target(map_path: Path) -> None:
    graph = parse_map_file(str(map_path))
    drones = plan(graph)
    result = Simulator(graph, drones).run()
    target = _target_for(map_path)
    assert result.makespan <= target, (
        f"{map_path.name}: {result.makespan} > target {target}"
    )


def _target_for(path: Path) -> int:
    name = path.name
    easy = {
        "01_linear_path.txt": 6,
        "02_simple_fork.txt": 8,
        "03_basic_capacity.txt": 6,
    }
    medium = {
        "01_dead_end_trap.txt": 12,
        "02_circular_loop.txt": 15,
        "03_priority_puzzle.txt": 12,
    }
    hard = {
        "01_maze_nightmare.txt": 30,
        "02_capacity_hell.txt": 35,
        "03_ultimate_challenge.txt": 45,
    }
    if name in easy:
        return easy[name]
    if name in medium:
        return medium[name]
    if name in hard:
        return hard[name]
    return 45  # challenger reference record


def test_restricted_transit_costs_two_turns() -> None:
    text = (
        "nb_drones: 1\n"
        "start_hub: a 0 0\n"
        "hub: c 1 0 [zone=restricted]\n"
        "end_hub: b 2 0\n"
        "connection: a-c\n"
        "connection: c-b\n"
    )
    graph = parse_map(text)
    drones = plan(graph)
    assert_valid(graph, drones)
    result = Simulator(graph, drones).run()
    tokens = [line for line in result.turns if line]
    # Turn 1: in transit on connection a-c.
    assert "D1-a-c" in tokens[0]
    # Turn 2: arrives at c.
    assert "D1-c" in tokens[1]
    # Turn 3: moves to b (end).
    assert "D1-b" in tokens[2]


def test_drone_label_format() -> None:
    from fly_in.simulator import label_for

    assert label_for(1) == "D1"
    assert label_for(12) == "D12"
