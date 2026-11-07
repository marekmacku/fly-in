"""Unit tests for fly_in2.simulator."""

from fly_in2.model import Drone, TurnRecord
from fly_in2.parser import parse_map
from fly_in2.planner import plan
from fly_in2.simulator import Simulator

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


def test_linear_playback() -> None:
    """Tokens merge per turn; waiting drones are omitted."""
    result = Simulator(plan(parse_map(LINEAR))).run()
    assert result.turns == [
        "D1-w1",
        "D1-w2 D2-w1",
        "D1-e D2-w2",
        "D2-e",
    ]
    assert result.makespan == 4


def test_restricted_transit_line() -> None:
    """In-transit drones print the src-dst connection name."""
    result = Simulator(plan(parse_map(RESTRICTED))).run()
    assert result.turns == ["D1-s-r", "D1-r", "D1-e"]
    assert result.makespan == 3


def test_turn_where_nobody_moves_is_empty() -> None:
    """A turn with only waits yields an empty line, not a gap."""
    drone = Drone(drone_id=1, trajectory=[
        TurnRecord(1, None, "s"),   # waits during turn 1
        TurnRecord(2, "e", "e"),    # moves during turn 2
    ])
    result = Simulator([drone]).run()
    assert result.turns == ["", "D1-e"]
    assert result.makespan == 2


def test_tokens_sorted_by_drone_id() -> None:
    """Within a line, drones appear in ascending id order."""
    drones = [
        Drone(drone_id=2, trajectory=[TurnRecord(1, "a", "a")]),
        Drone(drone_id=1, trajectory=[TurnRecord(1, "b", "b")]),
    ]
    result = Simulator(drones).run()
    assert result.turns == ["D1-b D2-a"]
