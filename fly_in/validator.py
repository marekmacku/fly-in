"""Validator: replays planned trajectories and checks every rule.

This is the correctness oracle used by tests and during defense. It verifies:

* drones start at the start zone and reach the end zone;
* no drone enters a blocked zone;
* every move uses an existing connection;
* restricted transits commit to a 2-turn pattern (enter connection, then
  arrive next turn) and the destination is restricted;
* zone occupancy never exceeds ``max_drones`` (start/end are unlimited);
* link occupancy never exceeds ``max_link_capacity``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from fly_in.errors import SimulationError
from fly_in.model import Drone, Graph, TurnRecord, ZoneType

INF_CAP = -1


class _OccCounter:
    """Small helper counting occupants per (key, turn)."""

    def __init__(self) -> None:
        self._counts: dict[tuple[object, int], int] = {}

    def add(self, key: object, turn: int) -> None:
        """Record one occupant for ``key`` at ``turn``."""
        self._counts[(key, turn)] = self._counts.get((key, turn), 0) + 1

    def count(self, key: object, turn: int) -> int:
        """Return the number of occupants for ``key`` at ``turn``."""
        return self._counts.get((key, turn), 0)


class _Violation(NamedTuple):
    """A single rule violation detected during replay."""

    turn: int
    message: str


class _Action(NamedTuple):
    did: int
    kind: str
    src: str
    dst: str | None
    edge: tuple[str, str] | None


@dataclass
class ValidationResult:
    """Result of replaying a plan: validity flag and list of violations."""

    valid: bool
    violations: list[_Violation]


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _records_by_turn(drone: Drone) -> dict[int, TurnRecord]:
    return {rec.turn: rec for rec in drone.trajectory}


def validate(graph: Graph, drones: list[Drone]) -> ValidationResult:
    """Replay all trajectories and report any rule violation."""
    assert graph.start is not None and graph.end is not None
    start_name = graph.start.name
    end_name = graph.end.name
    zones = graph.zones

    positions: dict[int, str | None] = {
        d.drone_id: start_name for d in drones
    }
    in_transit: dict[int, str | None] = {d.drone_id: None for d in drones}
    by_drone = {d.drone_id: _records_by_turn(d) for d in drones}
    delivered: set[int] = set()

    zone_occ = _OccCounter()
    link_occ = _OccCounter()
    violations: list[_Violation] = []

    makespan = max(
        (rec.turn for d in drones for rec in d.trajectory),
        default=0,
    )

    for turn in range(1, makespan + 1):
        # First pass: determine each drone's action and source/dest.
        actions: list[_Action] = []
        for drone in drones:
            did = drone.drone_id
            if did in delivered:
                continue
            rec = by_drone[did].get(turn)
            pos = positions[did]
            if rec is None:
                # No action: drone waits at its current zone.
                if pos is None:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} is in transit with no arrival record",
                        )
                    )
                    continue
                actions.append(_Action(did, "wait", pos, None, None))
                continue
            token = rec.token
            if token is None:
                if pos is None:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} waits while in transit",
                        )
                    )
                    continue
                actions.append(_Action(did, "wait", pos, None, None))
            elif "-" in token:
                # Transit onto a connection toward a restricted zone.
                src, _, dst = token.partition("-")
                if pos != src:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} transit {token} from {pos}",
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
            else:
                # Move-in or arrival.
                dest = token
                if dest not in zones:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} moves to unknown zone {dest}",
                        )
                    )
                    continue
                if in_transit[did] is not None:
                    # Arrival from a restricted transit.
                    if dest != in_transit[did]:
                        violations.append(
                            _Violation(
                                turn,
                                f"D{did} arrives at {dest} but committed "
                                f"to {in_transit[did]}",
                            )
                        )
                        continue
                    actions.append(_Action(did, "arrival", dest, None, None))
                else:
                    if pos is None:
                        violations.append(
                            _Violation(
                                turn,
                                f"D{did} moves while in transit",
                            )
                        )
                        continue
                    edge = _edge_key(pos, dest)
                    if edge not in graph.connections:
                        violations.append(
                            _Violation(
                                turn,
                                f"D{did} uses missing connection "
                                f"{pos}-{dest}",
                            )
                        )
                        continue
                    if zones[dest].zone_type is ZoneType.BLOCKED:
                        violations.append(
                            _Violation(
                                turn,
                                f"D{did} enters blocked zone {dest}",
                            )
                        )
                        continue
                    actions.append(_Action(did, "move", pos, dest, edge))

        # Second pass: apply capacity checks and update state.
        for action in actions:
            did = action.did
            kind = action.kind
            src = action.src
            act_dst = action.dst
            act_edge = action.edge
            if kind == "wait":
                zone_occ.add(src, turn)
                cap = zones[src].capacity
                if cap != INF_CAP and zone_occ.count(src, turn) > cap:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} overflows zone {src} capacity {cap}",
                        )
                    )
            elif kind == "move":
                assert act_dst is not None and act_edge is not None
                link_occ.add(act_edge, turn)
                lcap = graph.connections[act_edge].max_link_capacity
                if link_occ.count(act_edge, turn) > lcap:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} overflows link {act_edge} capacity {lcap}",
                        )
                    )
                zone_occ.add(act_dst, turn)
                cap = zones[act_dst].capacity
                if cap != INF_CAP and zone_occ.count(act_dst, turn) > cap:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} overflows zone {act_dst} capacity {cap}",
                        )
                    )
                positions[did] = act_dst
                if act_dst == end_name:
                    delivered.add(did)
            elif kind == "transit":
                assert act_dst is not None and act_edge is not None
                link_occ.add(act_edge, turn)
                lcap = graph.connections[act_edge].max_link_capacity
                if link_occ.count(act_edge, turn) > lcap:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} overflows link {act_edge} capacity {lcap}",
                        )
                    )
                in_transit[did] = act_dst
                positions[did] = None
            elif kind == "arrival":
                assert act_dst is None and act_edge is None
                dest = src  # arrival destination stored in src slot
                zone_occ.add(dest, turn)
                cap = zones[dest].capacity
                if cap != INF_CAP and zone_occ.count(dest, turn) > cap:
                    violations.append(
                        _Violation(
                            turn,
                            f"D{did} overflows zone {dest} capacity {cap}",
                        )
                    )
                positions[did] = dest
                in_transit[did] = None
                if dest == end_name:
                    delivered.add(did)

    for drone in drones:
        if drone.drone_id not in delivered:
            violations.append(
                _Violation(makespan, f"D{drone.drone_id} never reached end")
            )

    return ValidationResult(valid=not violations, violations=violations)


def assert_valid(graph: Graph, drones: list[Drone]) -> None:
    """Raise SimulationError if the plan violates any rule."""
    result = validate(graph, drones)
    if not result.valid:
        messages = "; ".join(
            f"turn {v.turn}: {v.message}" for v in result.violations
        )
        raise SimulationError(f"invalid simulation plan: {messages}")
