"""Time-expanded prioritized planning with a reservation table.

Each drone is planned one at a time against the reservations of all previously
planned drones. For a single drone we run Dijkstra over a time-expanded graph
whose states are ``(zone, ready_turn)`` pairs. Three actions are available:

* ``WAIT``     at zone v:    (v, t) -> (v, t+1), occupies v at turn t.
* ``MOVE``     to neighbour u (normal/priority): (v, t) -> (u, t+1),
  occupies link(v,u) at turn t and u at turn t (move-in); v is freed at t.
* ``TRANSIT``  to restricted u: (v, t) -> (u, t+2), occupies link(v,u) at
  turn t (in transit) and u at turn t+1 (arrival). The drone is busy for two
  turns and may only act again at turn t+2, matching the spec rule that a
  restricted move costs two turns and the drone MUST arrive the next turn.

Occupancy rule (spec): drones moving out of a zone free its capacity for that
same turn, so a move-out never reserves the source zone. Reservations are only
created for waiting and move-in/arrival turns.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import NamedTuple, cast

from fly_in.errors import SimulationError
from fly_in.model import Drone, Graph, TurnRecord, Zone, ZoneType
from fly_in.pathfinding import INF, dijkstra

START_TURN = 1
MAX_HORIZON = 2000

Action = str  # "wait" | "move" | "transit"
State = tuple[str, int]


class _Reservation(NamedTuple):
    kind: str  # "zone" | "link"
    key: object
    turn: int


@dataclass
class _ReservationTable:
    """Counts of drone occupants per (zone, turn) and (link, turn)."""

    zone_occ: dict[tuple[str, int], int] = field(default_factory=dict)
    link_occ: dict[tuple[tuple[str, str], int], int] = field(default_factory=dict)

    def zone_count(self, zone: str, turn: int) -> int:
        """Number of drones occupying ``zone`` at ``turn``."""
        return self.zone_occ.get((zone, turn), 0)

    def link_count(self, edge: tuple[str, str], turn: int) -> int:
        """Number of drones traversing ``edge`` at ``turn``."""
        return self.link_occ.get((edge, turn), 0)

    def commit(self, reservations: list[_Reservation]) -> None:
        """Apply all reservations of one drone to the global table."""
        for res in reservations:
            if res.kind == "zone":
                zkey: tuple[str, int] = (str(res.key), res.turn)
                self.zone_occ[zkey] = self.zone_occ.get(zkey, 0) + 1
            else:
                assert isinstance(res.key, tuple)
                edge = cast("tuple[str, str]", res.key)
                lkey: tuple[tuple[str, str], int] = (edge, res.turn)
                self.link_occ[lkey] = self.link_occ.get(lkey, 0) + 1


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _zone_capacity(zone: Zone) -> int:
    return zone.capacity


def _can_occupy_zone(
    table: _ReservationTable, zone: Zone, turn: int
) -> bool:
    cap = _zone_capacity(zone)
    if cap < 0:  # unlimited (start/end)
        return True
    return table.zone_count(zone.name, turn) + 1 <= cap


def _can_use_link(
    table: _ReservationTable, edge: tuple[str, str], turn: int, cap: int
) -> bool:
    return table.link_count(edge, turn) + 1 <= cap


@dataclass
class _SearchNode:
    state: State
    parent: State | None
    action: Action
    destination: str  # neighbour name (== state[0] for wait)


def _plan_drone(
    graph: Graph,
    table: _ReservationTable,
    drone_id: int,
    horizon: int,
) -> list[TurnRecord]:
    """Plan a conflict-free trajectory for a single drone."""
    assert graph.start is not None and graph.end is not None
    start_name = graph.start.name
    end_name = graph.end.name
    start_state: State = (start_name, START_TURN)

    nodes: dict[State, _SearchNode] = {
        start_state: _SearchNode(start_state, None, "start", start_name)
    }
    best_ready: dict[State, int] = {start_state: START_TURN}
    heap: list[tuple[int, int, State]] = [(START_TURN, 0, start_state)]
    counter = 1
    goal_state: State | None = None
    finalized: set[State] = set()

    while heap:
        ready, _, state = heapq.heappop(heap)
        if state in finalized:
            continue
        finalized.add(state)
        zone_name, t = state
        if zone_name == end_name:
            goal_state = state
            break
        if t >= horizon:
            continue
        zone = graph.zones[zone_name]

        # WAIT
        wait_state: State = (zone_name, t + 1)
        if wait_state not in finalized and _can_occupy_zone(
            table, zone, t
        ):
            if ready + 1 < best_ready.get(wait_state, MAX_HORIZON + 1):
                best_ready[wait_state] = ready + 1
                nodes[wait_state] = _SearchNode(
                    wait_state, state, "wait", zone_name
                )
                heapq.heappush(heap, (ready + 1, counter, wait_state))
                counter += 1

        # MOVE / TRANSIT to neighbours
        for neighbour, conn in graph.neighbors(zone_name):
            dest = graph.zones[neighbour]
            if dest.zone_type is ZoneType.BLOCKED:
                continue
            edge = _edge_key(zone_name, neighbour)
            if dest.zone_type is ZoneType.RESTRICTED:
                new_t = t + 2
                arrival = t + 1
                if new_t > horizon:
                    continue
                if not _can_use_link(table, edge, t, conn.max_link_capacity):
                    continue
                if not _can_occupy_zone(table, dest, arrival):
                    continue
                new_state: State = (neighbour, new_t)
                if new_state in finalized:
                    continue
                if new_t < best_ready.get(new_state, MAX_HORIZON + 1):
                    best_ready[new_state] = new_t
                    nodes[new_state] = _SearchNode(
                        new_state, state, "transit", neighbour
                    )
                    heapq.heappush(heap, (new_t, counter, new_state))
                    counter += 1
            else:
                new_t = t + 1
                if new_t > horizon:
                    continue
                if not _can_use_link(table, edge, t, conn.max_link_capacity):
                    continue
                if not _can_occupy_zone(table, dest, t):
                    continue
                new_state = (neighbour, new_t)
                if new_state in finalized:
                    continue
                if new_t < best_ready.get(new_state, MAX_HORIZON + 1):
                    best_ready[new_state] = new_t
                    nodes[new_state] = _SearchNode(
                        new_state, state, "move", neighbour
                    )
                    heapq.heappush(heap, (new_t, counter, new_state))
                    counter += 1

    if goal_state is None:
        raise SimulationError(
            f"drone D{drone_id}: no path to end within horizon {horizon}"
        )

    path_nodes: list[_SearchNode] = []
    cursor: State | None = goal_state
    while cursor is not None:
        node = nodes[cursor]
        path_nodes.append(node)
        cursor = node.parent
    path_nodes.reverse()

    records: list[TurnRecord] = []
    reservations: list[_Reservation] = []
    prev_zone = start_name
    for node in path_nodes[1:]:
        t = node.state[1]
        if node.action == "wait":
            records.append(TurnRecord(turn=t - 1, token=None, location=prev_zone))
            reservations.append(
                _Reservation("zone", prev_zone, t - 1)
            )
        elif node.action == "move":
            dest_name = node.destination
            edge = _edge_key(prev_zone, dest_name)
            records.append(
                TurnRecord(turn=t - 1, token=dest_name, location=dest_name)
            )
            reservations.append(_Reservation("link", edge, t - 1))
            reservations.append(_Reservation("zone", dest_name, t - 1))
            prev_zone = dest_name
        elif node.action == "transit":
            dest_name = node.destination
            edge = _edge_key(prev_zone, dest_name)
            records.append(
                TurnRecord(
                    turn=t - 2,
                    token=f"{prev_zone}-{dest_name}",
                    location=None,
                )
            )
            records.append(
                TurnRecord(turn=t - 1, token=dest_name, location=dest_name)
            )
            reservations.append(_Reservation("link", edge, t - 2))
            reservations.append(_Reservation("zone", dest_name, t - 1))
            prev_zone = dest_name

    table.commit(reservations)
    return records


def _makespan(drones: list[Drone]) -> int:
    """Return the last simulation turn across all drone trajectories."""
    last = 0
    for drone in drones:
        for rec in drone.trajectory:
            if rec.turn > last:
                last = rec.turn
    return last


def _plan_for_order(
    graph: Graph, order: list[int], horizon: int
) -> list[Drone] | None:
    """Plan every drone in the given order; return drones or None on failure."""
    table = _ReservationTable()
    drones: list[Drone] = []
    for drone_id in order:
        try:
            records = _plan_drone(graph, table, drone_id, horizon)
        except SimulationError:
            return None
        drones.append(Drone(drone_id=drone_id, trajectory=records))
    return drones


def _base_cost(graph: Graph) -> int:
    """Static shortest path cost (in turns) from start to end."""
    assert graph.start is not None and graph.end is not None
    result = dijkstra(graph, graph.start.name)
    dist = result.distances.get(graph.end.name, INF)
    if dist == INF:
        raise SimulationError("end zone is unreachable from start")
    return int(dist)


def _candidate_orders(n: int) -> list[list[int]]:
    """Generate a handful of drone orderings to try."""
    base = list(range(1, n + 1))
    orders: list[list[int]] = [base, list(reversed(base))]
    rng = random.Random(424242)
    seen = {"", }
    for _ in range(8):
        shuffled = base[:]
        rng.shuffle(shuffled)
        sig = ",".join(map(str, shuffled))
        if sig in seen:
            continue
        seen.add(sig)
        orders.append(shuffled)
    return orders


def plan(graph: Graph) -> list[Drone]:
    """Plan trajectories for all drones, trying several orderings."""
    graph.validate()
    base = _base_cost(graph)
    n = graph.num_drones
    horizon = min(MAX_HORIZON, base * 2 + n * (base + 1) + 64)
    best: list[Drone] | None = None
    best_score = -1
    for order in _candidate_orders(n):
        result = _plan_for_order(graph, order, horizon)
        if result is None:
            continue
        score = _makespan(result)
        if best is None or score < best_score:
            best = result
            best_score = score
    if best is None:
        # Fallback: plan in ascending order, surfacing any error.
        ascending = list(range(1, n + 1))
        return _plan_for_order(graph, ascending, horizon) or []
    return best
