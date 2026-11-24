"""Unit tests for the pure animation helpers in fly_in2.gui (they need no pygame)."""

from functools import partial
from typing import Callable

import pytest

from fly_in2.gui import (
    _drone_points,
    _flights,
    _glide,
    _transit_edges,
)
from fly_in2.parser import parse_map
from fly_in2.planner import plan
from fly_in2.simulator import Simulator
from fly_in2.visualizer import positions_per_turn

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

H = 1e-6


def slope(curve: Callable[[float], float], a: float) -> float:
    """Speed of an eased curve over the tiny step from `a` towards the inside of [0, 1]."""
    b = a + H if a < 0.5 else a - H
    return (curve(b) - curve(a)) / (b - a)


def flights_for(text: str) -> list[dict[int, tuple[str, str]]]:
    """Plan a map and return which link each drone flies on each turn."""
    graph = parse_map(text)
    drones = plan(graph)
    n_turns = len(Simulator(drones).run().turns)
    positions = positions_per_turn(graph, drones, n_turns)
    return _flights(positions, _transit_edges(drones), "s", n_turns)


# --- easing --------------------------------------------------------------------------------


def test_glide_between_zones_starts_and_ends_at_rest() -> None:
    """A zone-to-zone leg runs 0 -> 1 and has zero speed at both zones."""
    leg = partial(_glide, from_mid=False, to_mid=False)
    assert leg(0.0) == pytest.approx(0.0)
    assert leg(1.0) == pytest.approx(1.0)
    assert slope(leg, 0.0) == pytest.approx(0.0, abs=1e-4)
    assert slope(leg, 1.0) == pytest.approx(0.0, abs=1e-4)


def test_restricted_crossing_starts_and_ends_at_rest() -> None:
    """A restricted crossing accelerates out of its zone and decelerates into the next one."""
    departing = partial(_glide, from_mid=False, to_mid=True)
    arriving = partial(_glide, from_mid=True, to_mid=False)
    assert slope(departing, 0.0) == pytest.approx(0.0, abs=1e-4)
    assert slope(arriving, 1.0) == pytest.approx(0.0, abs=1e-4)


def test_restricted_crossing_meets_at_the_midpoint() -> None:
    """The departing half ends exactly where the arriving half begins: the link midpoint."""
    assert _glide(1.0, from_mid=False, to_mid=True) == pytest.approx(1.0)
    assert _glide(0.0, from_mid=True, to_mid=False) == pytest.approx(0.0)


def test_restricted_crossing_does_not_stop_at_the_midpoint() -> None:
    """Speed is continuous and non-zero where the two halves of a restricted crossing meet."""
    into_mid = slope(partial(_glide, from_mid=False, to_mid=True), 1.0)
    out_of_mid = slope(partial(_glide, from_mid=True, to_mid=False), 0.0)
    assert into_mid == pytest.approx(out_of_mid, rel=1e-3)
    assert into_mid > 0.5


# --- drone positions -----------------------------------------------------------------------

# One drone: start (0, 0), restricted midpoint (50, 0) after turn 1, far zone (100, 0) after 2.
FRAMES = [{1: (0, 0)}, {1: (50, 0)}, {1: (100, 0)}]
TRANSIT = {1: {1: ("a", "b")}}


def test_drone_points_sit_on_the_keyframes_at_whole_turns() -> None:
    """At the end of turn k a drone is exactly on keyframe k, restricted midpoint included."""
    assert _drone_points(FRAMES, TRANSIT, 0.0) == {1: (0, 0)}
    assert _drone_points(FRAMES, TRANSIT, 1.0) == {1: (50, 0)}
    assert _drone_points(FRAMES, TRANSIT, 2.0) == {1: (100, 0)}


def test_drone_points_move_between_keyframes_during_a_turn() -> None:
    """Halfway through a turn the drone is strictly between that turn's two keyframes."""
    assert 0 < _drone_points(FRAMES, TRANSIT, 0.5)[1][0] < 50
    assert 50 < _drone_points(FRAMES, TRANSIT, 1.5)[1][0] < 100


# --- flights -------------------------------------------------------------------------------


def test_flights_list_only_drones_that_move() -> None:
    """Waiting and already delivered drones fly no link on that turn."""
    assert flights_for(LINEAR) == [
        {},
        {1: ("s", "w1")},                       # D2 waits on start
        {1: ("w1", "w2"), 2: ("s", "w1")},
        {1: ("w2", "e"), 2: ("w1", "w2")},
        {2: ("w2", "e")},                       # D1 already delivered
    ]


def test_restricted_crossing_flies_its_link_on_both_turns() -> None:
    """A restricted move shows the same link on its departure turn and its arrival turn."""
    assert flights_for(RESTRICTED) == [
        {},
        {1: ("s", "r")},
        {1: ("s", "r")},
        {1: ("r", "e")},
    ]
