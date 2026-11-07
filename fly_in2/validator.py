"""Validator: the independent oracle that replays a finished plan.

Replays every trajectory turn by turn and re-checks each rule from
scratch: start and delivery, blocked zones, existing connections, the
two-turn restricted transit pattern, and every zone and link capacity
limit (start/end hubs are unlimited).

The occupancy bookkeeping here deliberately duplicates the planner's
reservation-table logic instead of sharing it: an oracle that reuses
the code it is checking cannot catch bugs in that code. Both derive
the same rule - a drone leaving a zone does not occupy it that turn -
independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

from fly_in2.errors import SimulationError
from fly_in2.model import Drone, Graph, TurnRecord, ZoneType

ActionKind = Literal["wait", "move", "transit", "arrival"]


class _OccCounter:
    """Occupancy counts per (zone-or-link, turn)."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str | tuple[str, str], int], int] = {}

    def add(self, key: str | tuple[str, str], turn: int) -> None:
        """Record one occupant for ``key`` during ``turn``."""
        self._counts[(key, turn)] = self._counts.get((key, turn), 0) + 1

    def count(self, key: str | tuple[str, str], turn: int) -> int:
        """Number of occupants for ``key`` during ``turn``."""
        return self._counts.get((key, turn), 0)


class _Violation(NamedTuple):
    """A single rule violation detected during replay."""

    turn: int
    message: str


class _Action(NamedTuple):
    """What one drone does during one turn of the replay."""

    did: int
    kind: ActionKind
    src: str | None
    dest: str | None
    edge: tuple[str, str] | None


@dataclass
class ValidationResult:
    """Outcome of a replay: validity flag and every violation found."""

    valid: bool
    violations: list[_Violation]


def _edge_key(a: str, b: str) -> tuple[str, str]:
    """Canonical sorted endpoint pair (same form as Connection.key)."""
    return (a, b) if a <= b else (b, a)


def _check_zone(
    counter: _OccCounter, zone: str, turn: int,
    cap: int | None, did: int, violations: list[_Violation],
) -> None:
    """Add one occupant and flag an overflow if capacity is exceeded."""
    counter.add(zone, turn)
    if cap is not None and counter.count(zone, turn) > cap:
        violations.append(
            _Violation(
                turn, f"D{did} overflows zone {zone} capacity {cap}",
            )
        )


def _check_link(
    counter: _OccCounter, edge: tuple[str, str], turn: int,
    cap: int, did: int, violations: list[_Violation],
) -> None:
    """Add one traversal and flag an overflow if capacity is exceeded."""
    counter.add(edge, turn)
    if counter.count(edge, turn) > cap:
        violations.append(
            _Violation(
                turn,
                f"D{did} overflows link {edge} capacity {cap}",
            )
        )


def validate(graph: Graph, drones: list[Drone]) -> ValidationResult:
    """Replay all trajectories and report every rule violation."""
    assert graph.start is not None and graph.end is not None
    start_name = graph.start.name
    end_name = graph.end.name
    zones = graph.zones

    positions: dict[int, str | None] = {
        d.drone_id: start_name for d in drones
    }
    committed: dict[int, str | None] = {d.drone_id: None for d in drones}
    by_drone: dict[int, dict[int, TurnRecord]] = {
        d.drone_id: {rec.turn: rec for rec in d.trajectory} for d in drones
    }
    delivered: set[int] = set()

    zone_occ = _OccCounter()
    link_occ = _OccCounter()
    violations: list[_Violation] = []
    makespan = max(
        (rec.turn for d in drones for rec in d.trajectory), default=0
    )

    for turn in range(1, makespan + 1):
        # Pass 1: classify each active drone's action, or flag it.
        actions: list[_Action] = []
        for drone in drones:
            did = drone.drone_id
            if did in delivered:
                continue
            rec = by_drone[did].get(turn)
            pos = positions[did]
            if rec is None or rec.token is None:
                if pos is None:  # in transit: waiting is impossible
                    violations.append(
                        _Violation(
                            turn, f"D{did} waits while in transit",
                        )
                    )
                    continue
                actions.append(_Action(did, "wait", pos, None, None))
                continue
            token = rec.token
            if "-" in token:
                src, _, dst = token.partition("-")
                if pos != src:
                    violations.append(
                        _Violation(
                            turn, f"D{did} transit {token} from {pos}",
                        )
                    )
                    continue
                if src not in zones or dst not in zones:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} transit references unknown zone",
                        )
                    )
                    continue
                edge = _edge_key(src, dst)
                if edge not in graph.connections:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} transit on missing connection {token}",
                        )
                    )
                    continue
                if zones[dst].zone_type is not ZoneType.RESTRICTED:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} transit to non-restricted {dst}",
                        )
                    )
                    continue
                actions.append(_Action(did, "transit", src, dst, edge))
            elif committed[did] is not None:
                # Forced arrival from a restricted transit.
                if token != committed[did]:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} arrives at {token} but committed "
                            f"to {committed[did]}",
                        )
                    )
                    continue
                actions.append(_Action(did, "arrival", None, token, None))
            else:
                if pos is None:
                    violations.append(
                        _Violation(
                            turn, f"D{did} moves while in transit",
                        )
                    )
                    continue
                if token not in zones:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} moves to unknown zone {token}",
                        )
                    )
                    continue
                edge = _edge_key(pos, token)
                if edge not in graph.connections:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} uses missing connection "
                            f"{pos}-{token}",
                        )
                    )
                    continue
                if zones[token].zone_type.movement_cost is None:
                    violations.append(
                        _Violation(
                            turn, f"D{did} enters blocked zone {token}",
                        )
                    )
                    continue
                actions.append(_Action(did, "move", pos, token, edge))

        # Pass 2: apply capacity rules, then update drone state.
        for act in actions:
            did = act.did
            if act.kind == "wait":
                assert act.src is not None
                _check_zone(
                    zone_occ, act.src, turn,
                    zones[act.src].capacity, did, violations,
                )
            elif act.kind == "move":
                assert act.dest is not None and act.edge is not None
                _check_link(
                    link_occ, act.edge, turn,
                    graph.connections[act.edge].max_link_capacity,
                    did, violations,
                )
                _check_zone(
                    zone_occ, act.dest, turn,
                    zones[act.dest].capacity, did, violations,
                )
                positions[did] = act.dest
                if act.dest == end_name:
                    delivered.add(did)
            elif act.kind == "transit":
                assert act.dest is not None and act.edge is not None
                _check_link(
                    link_occ, act.edge, turn,
                    graph.connections[act.edge].max_link_capacity,
                    did, violations,
                )
                committed[did] = act.dest
                positions[did] = None
            else:  # arrival
                assert act.dest is not None
                _check_zone(
                    zone_occ, act.dest, turn,
                    zones[act.dest].capacity, did, violations,
                )
                positions[did] = act.dest
                committed[did] = None
                if act.dest == end_name:
                    delivered.add(did)

    for drone in drones:
        if drone.drone_id not in delivered:
            violations.append(
                _Violation(
                    makespan, f"D{drone.drone_id} never reached end",
                )
            )
    return ValidationResult(valid=not violations, violations=violations)


def assert_valid(graph: Graph, drones: list[Drone]) -> None:
    """Raise SimulationError if the plan violates any rule.

    Raises:
        SimulationError: with every violation, turn-stamped.
    """
    result = validate(graph, drones)
    if not result.valid:
        messages = "; ".join(
            f"turn {v.turn}: {v.message}" for v in result.violations
        )
        raise SimulationError(f"invalid simulation plan: {messages}")
