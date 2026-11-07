"""Tests for fly_in2.validator: the independent replay oracle."""

from pathlib import Path

from fly_in2.model import Drone, TurnRecord
from fly_in2.parser import parse_map
from fly_in2.planner import plan
from fly_in2.validator import assert_valid, validate

ROOT = Path(__file__).resolve().parent.parent
PROVIDED = sorted((ROOT / "maps_download").glob("*/*.txt"))
CUSTOM = sorted((ROOT / "maps_custom" / "valid").glob("*.txt"))

CORRIDOR = """
nb_drones: 2
start_hub: s 0 0
hub: m 1 0
end_hub: e 2 0
connection: s-m
connection: m-e
"""

BLOCKED_MAP = """
nb_drones: 1
start_hub: s 0 0
end_hub: e 2 0
hub: wall 1 0 [zone=blocked]
connection: s-wall
connection: wall-e
"""

RESTRICTED_MAP = """
nb_drones: 1
start_hub: s 0 0
hub: r 1 0 [zone=restricted]
end_hub: e 2 0
connection: s-r
connection: r-e
"""


def test_all_provided_and_custom_maps_validate() -> None:
    """Every planned map passes the oracle (19 maps)."""
    for path in PROVIDED + CUSTOM:
        graph = parse_map(path.read_text())
        assert_valid(graph, plan(graph))


def test_same_turn_swap_is_legal() -> None:
    """A drone leaving a zone frees its capacity for the same turn."""
    graph = parse_map(CORRIDOR)
    drones = [
        Drone(drone_id=1, trajectory=[
            TurnRecord(1, "m", "m"),
            TurnRecord(2, "e", "e"),   # leaves m during turn 2
        ]),
        Drone(drone_id=2, trajectory=[
            TurnRecord(1, None, "s"),  # waits
            TurnRecord(2, "m", "m"),   # enters m the same turn D1 left
            TurnRecord(3, "e", "e"),
        ]),
    ]
    assert validate(graph, drones).valid


def test_zone_capacity_overflow_is_caught() -> None:
    graph = parse_map(CORRIDOR)
    drones = [
        Drone(drone_id=1, trajectory=[TurnRecord(1, "m", "m")]),
        Drone(drone_id=2, trajectory=[TurnRecord(1, "m", "m")]),
    ]
    result = validate(graph, drones)
    assert not result.valid
    assert any(
        "overflows zone m capacity 1" in v.message
        for v in result.violations
    )


def test_link_capacity_overflow_is_caught() -> None:
    graph = parse_map(CORRIDOR)
    drones = [
        Drone(drone_id=1, trajectory=[TurnRecord(1, "m", "m")]),
        Drone(drone_id=2, trajectory=[TurnRecord(1, "m", "m")]),
    ]
    result = validate(graph, drones)
    messages = [v.message for v in result.violations]
    assert any("overflows link" in m for m in messages)


def test_blocked_entry_is_caught() -> None:
    graph = parse_map(BLOCKED_MAP)
    drones = [
        Drone(drone_id=1, trajectory=[
            TurnRecord(1, "wall", "wall"),
        ]),
    ]
    result = validate(graph, drones)
    assert not result.valid
    assert "enters blocked zone wall" in result.violations[0].message


def test_missing_connection_is_caught() -> None:
    graph = parse_map(CORRIDOR)  # s and e are NOT connected
    drones = [
        Drone(drone_id=1, trajectory=[TurnRecord(1, "e", "e")]),
    ]
    result = validate(graph, drones)
    assert not result.valid
    assert any("missing connection" in v.message for v in result.violations)


def test_wait_while_in_transit_is_caught() -> None:
    graph = parse_map(RESTRICTED_MAP)
    drones = [
        Drone(drone_id=1, trajectory=[
            TurnRecord(1, "s-r", None),   # enters transit
            TurnRecord(2, None, None),    # illegal: waits mid-air
        ]),
    ]
    result = validate(graph, drones)
    assert not result.valid
    assert any(
        "waits while in transit" in v.message for v in result.violations
    )


def test_wrong_transit_arrival_is_caught() -> None:
    """The drone committed to r but arrives elsewhere next turn."""
    graph = parse_map(RESTRICTED_MAP)
    drones = [
        Drone(drone_id=1, trajectory=[
            TurnRecord(1, "s-r", None),
            TurnRecord(2, "e", "e"),
        ]),
    ]
    result = validate(graph, drones)
    assert not result.valid
    assert any(
        "committed to r" in v.message for v in result.violations
    )


def test_undelivered_drone_is_caught() -> None:
    graph = parse_map(CORRIDOR)
    drones = [
        Drone(drone_id=1, trajectory=[
            TurnRecord(1, "m", "m"),
        ]),
    ]
    result = validate(graph, drones)
    assert not result.valid
    assert any(
        "never reached end" in v.message for v in result.violations
    )
