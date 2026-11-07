"""Unit and integration tests for fly_in2.planner."""

from pathlib import Path

import pytest

from fly_in2.errors import SimulationError
from fly_in2.model import Drone
from fly_in2.parser import parse_map
from fly_in2.planner import plan

MAPS = Path(__file__).resolve().parent.parent / "maps_download"


def _makespan(drones: list[Drone]) -> int:
    return max(
        (rec.turn for d in drones for rec in d.trajectory
         if rec.token is not None),
        default=0,
    )


def _trajectory(
    drones: list[Drone], drone_id: int
) -> list[tuple[int, str | None, str | None]]:
    return [
        (rec.turn, rec.token, rec.location)
        for d in drones if d.drone_id == drone_id
        for rec in d.trajectory
    ]


LINEAR = """
nb_drones: 2
start_hub: s 0 0
hub: w1 1 0
hub: w2 2 0
end_hub: e 3 0
connection: s-w1
connection: w1-w2
connection: w2-e
"""

RESTRICTED = """
nb_drones: 1
start_hub: s 0 0
hub: r 1 0 [zone=restricted]
end_hub: e 2 0
connection: s-r
connection: r-e
"""

PRIORITY_ROUTES = """
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

UNREACHABLE = """
nb_drones: 1
start_hub: s 0 0
end_hub: e 9 9
hub: wall 1 0 [zone=blocked]
connection: s-wall
connection: wall-e
"""


def test_linear_path_trajectories() -> None:
    """D1 walks straight through; D2 waits one turn then follows."""
    drones = plan(parse_map(LINEAR))
    assert _trajectory(drones, 1) == [
        (1, "w1", "w1"), (2, "w2", "w2"), (3, "e", "e"),
    ]
    assert _trajectory(drones, 2) == [
        (1, None, "s"), (2, "w1", "w1"), (3, "w2", "w2"), (4, "e", "e"),
    ]
    assert _makespan(drones) == 4


def test_restricted_transit_pattern() -> None:
    """A restricted move emits link-in-transit then forced arrival."""
    drones = plan(parse_map(RESTRICTED))
    assert _trajectory(drones, 1) == [
        (1, "s-r", None),   # in transit on the connection
        (2, "r", "r"),      # forced arrival the next turn
        (3, "e", "e"),      # can only act again from turn 3
    ]


def test_priority_route_preferred() -> None:
    """Among equal-time routes the drone flies the priority lane."""
    drones = plan(parse_map(PRIORITY_ROUTES))
    tokens = [rec.token for rec in drones[0].trajectory]
    assert tokens == ["p1", "p2", "e"]


def test_zone_capacity_is_respected() -> None:
    """No (zone, turn) ever holds more drones than its capacity."""
    graph = parse_map((MAPS / "hard" / "02_capacity_hell.txt").read_text())
    drones = plan(graph)
    occupancy: dict[tuple[str, int], int] = {}
    for drone in drones:
        for rec in drone.trajectory:
            if rec.location is None:
                continue  # in transit on a connection
            occupancy[(rec.location, rec.turn)] = (
                occupancy.get((rec.location, rec.turn), 0) + 1
            )
    for (zone_name, turn), count in occupancy.items():
        capacity = graph.zones[zone_name].capacity
        assert capacity is None or count <= capacity, (
            f"{zone_name} at turn {turn}: {count} > {capacity}"
        )


def test_unreachable_end_raises() -> None:
    with pytest.raises(SimulationError, match="unreachable"):
        plan(parse_map(UNREACHABLE))


BENCHMARKS = [
    ("easy/01_linear_path.txt", 6, 4),
    ("easy/02_simple_fork.txt", 8, 4),
    ("easy/03_basic_capacity.txt", 6, 4),
    ("medium/01_dead_end_trap.txt", 12, 8),
    ("medium/02_circular_loop.txt", 15, 10),
    ("medium/03_priority_puzzle.txt", 12, 6),
    ("hard/01_maze_nightmare.txt", 30, 13),
    ("hard/02_capacity_hell.txt", 35, 16),
    ("hard/03_ultimate_challenge.txt", 45, 26),
    ("challenger/01_the_impossible_dream.txt", 44, 43),
]


def test_benchmark_targets() -> None:
    """Every provided map meets its turn target (challenger < 45)."""
    for relative, target, expected in BENCHMARKS:
        graph = parse_map((MAPS / relative).read_text())
        assert _makespan(plan(graph)) == expected, relative
        assert expected <= target, relative
