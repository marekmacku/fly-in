"""Challenge-map tests: hand-computed expectations for new maps.

Each map in maps_custom/challenge targets one behaviour; most expected
makespans are hand-derived from the capacity arithmetic, so a planner
regression shows up as a mismatch here.
"""

from pathlib import Path

from fly_in2.model import Drone
from fly_in2.parser import parse_map
from fly_in2.planner import plan
from fly_in2.validator import assert_valid

CHALLENGES = Path(__file__).resolve().parent.parent / "maps_custom" / "challenge"


def _makespan(drones: list[Drone]) -> int:
    return max(
        (rec.turn for d in drones for rec in d.trajectory
         if rec.token is not None),
        default=0,
    )


def _tokens(drones: list[Drone]) -> list[str]:
    return [
        rec.token for d in drones for rec in d.trajectory
        if rec.token is not None
    ]


def _plan_map(name: str) -> tuple[list[Drone], object]:
    graph = parse_map((CHALLENGES / name).read_text())
    drones = plan(graph)
    assert_valid(graph, drones)
    return drones, graph


# name -> exact expected makespan (hand-computed)
EXACT = {
    "m01_restricted_corridor.txt": 5,     # 2+2+1 stacked transits
    "m02_narrow_bridge.txt": 9,           # 4 hops + 5 queue turns
    "m03_cost_beats_preference.txt": 3,   # crack beats priority lane
    "m04_dead_end_bait.txt": 5,           # 3 hops + 2 queue turns
    "m05_hub_star.txt": 3,                # 2 hops, 4-wide links
    "m07_link_bottleneck.txt": 6,         # 3 hops + 3 link-queue turns
    "m09_transit_conflict.txt": 4,        # wait out the transit
    "m10_wall_detour.txt": 5,             # 3-turn crack, queued x3
}


def test_exact_makespans() -> None:
    for name, expected in EXACT.items():
        drones, _graph = _plan_map(name)
        assert _makespan(drones) == expected, name


def test_mesh_grid_scales_well() -> None:
    """Six drones through a 4x4 mesh with two holes: near-optimal."""
    drones, _graph = _plan_map("m06_mesh_grid.txt")
    assert 6 <= _makespan(drones) <= 12


def test_big_world_plan_is_valid_and_bounded() -> None:
    """20 drones, 48 zones, walled column.

    Result is 31 turns: the theoretical split-route optimum is around
    22, but greedy prioritized planning piles every drone onto the same
    corridor (each drone's individually fastest route is the same one),
    giving the single-corridor pipeline bound of base + n - 1 =
    12 + 19. Known, explainable limitation - see README.
    """
    drones, _graph = _plan_map("m08_big_world.txt")
    assert _makespan(drones) <= 32


def test_cost_beats_preference_uses_the_crack() -> None:
    """The drone flies the restricted crack, not the priority lane."""
    drones, _graph = _plan_map("m03_cost_beats_preference.txt")
    assert _tokens(drones) == ["s-crack", "crack", "e"]


def test_dead_end_bait_is_ignored() -> None:
    """Priority bait in dead ends lures nobody."""
    drones, _graph = _plan_map("m04_dead_end_bait.txt")
    assert all("bait" not in t for t in _tokens(drones))


def test_transit_conflict_wait_pattern() -> None:
    """D2 waits a turn, transits, and lands as D1 vacates the zone."""
    drones, _graph = _plan_map("m09_transit_conflict.txt")
    drone2 = next(d for d in drones if d.drone_id == 2)
    turns = [(rec.turn, rec.token) for rec in drone2.trajectory]
    assert turns == [(1, None), (2, "s-r"), (3, "r"), (4, "e")]
