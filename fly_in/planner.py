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
from dataclasses import dataclass, field
from typing import NamedTuple, cast

from fly_in.errors import SimulationError
from fly_in.model import Drone, Graph, TurnRecord, Zone, ZoneType
from fly_in.pathfinding import INF, dijkstra

START_TURN = 1
# Bound on (zones x turns) states a single drone's search may materialize;
# the per-map turn cap is derived from it so memory stays bounded on maps
# of any size. Well above n * base, the provable worst-case plan length.
MAX_STATES = 5_000_000
FLOOR_HORIZON = 2048

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
            if ready + 1 < best_ready.get(wait_state, MAX_STATES + 1):
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
                if new_t < best_ready.get(new_state, MAX_STATES + 1):
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
                if new_t < best_ready.get(new_state, MAX_STATES + 1):
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


def _plan_all(graph: Graph, horizon: int) -> list[Drone]:
    """Plan every drone in ascending id order against one shared table.

    All drones are interchangeable (same start zone, same goal), so the
    planning order cannot change the outcome; a single pass suffices.
    """
    table = _ReservationTable()
    drones: list[Drone] = []
    for drone_id in range(1, graph.num_drones + 1):
        records = _plan_drone(graph, table, drone_id, horizon)
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


def plan(graph: Graph) -> list[Drone]:
    """Plan trajectories for all drones in a single ascending pass."""
    graph.validate()
    base = _base_cost(graph)
    n = graph.num_drones
    cap = max(FLOOR_HORIZON, MAX_STATES // max(1, len(graph.zones)))
    horizon = min(cap, base * 2 + n * (base + 1) + 64)
    return _plan_all(graph, horizon)
