"""Time-expanded prioritized planning with a reservation table.

Each drone is planned one at a time (ascending id order) against the
reservations of all previously planned drones. For a single drone we
run Dijkstra over a time-expanded graph whose states are
``(zone, ready_turn)`` pairs; the ready turn is the next turn the
drone may act in. Three actions are available:

* ``WAIT`` at zone v: (v, t) -> (v, t+1); occupies v during turn t.
  The start zone has unlimited room, so a wait there reserves nothing
  and jumps straight to the next turn the drone could leave in: one
  search step instead of one per idle turn, which keeps planning
  linear in the fleet size rather than quadratic.
* ``MOVE`` to neighbour u (normal/priority): (v, t) -> (u, t+1); uses
  link(v, u) during turn t and occupies u from turn t (move-in). The
  source v is not reserved at t: drones moving out free capacity for
  that same turn (spec rule).
* ``TRANSIT`` to restricted u: (v, t) -> (u, t+2); occupies link(v, u)
  during turn t (in transit) and u from turn t+1. Arrival is forced
  the next turn, per the spec's two-turn restricted move.

Among states with the same ready turn, the search prefers routes that
enter fewer non-priority zones (the spec's "priority zones should be
prioritized"), mirroring the tie-break in pathfinding.dijkstra; it can
never influence the arrival turn itself.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Literal

from fly_in2.errors import SimulationError
from fly_in2.model import (
    Connection,
    Drone,
    Graph,
    TurnRecord,
    Zone,
    ZoneType,
)
from fly_in2.pathfinding import dijkstra

START_TURN = 1
# Bound on (zones x turns) states a single drone's search may
# materialize; the per-map horizon is derived from it so memory stays
# bounded on maps of any size. Well above n * base, the provable
# worst-case plan length.
MAX_STATES = 5_000_000
FLOOR_HORIZON = 2048

Action = Literal["start", "wait", "move", "transit"]
State = tuple[str, int]
ZoneRes = tuple[str, int]
LinkRes = tuple[tuple[str, str], int]


def _edge_key(a: str, b: str) -> tuple[str, str]:
    """Canonical sorted endpoint pair (same form as Connection.key)."""
    return (a, b) if a <= b else (b, a)


@dataclass
class _ReservationTable:
    """Occupancy counts per (zone, turn) and per (link, turn)."""

    zone_occ: dict[ZoneRes, int] = field(default_factory=dict)
    link_occ: dict[LinkRes, int] = field(default_factory=dict)
    # Per start neighbour: every turn below this one is known to have no
    # room to leave the start that way (see _next_departure).
    depart_from: dict[str, int] = field(default_factory=dict)

    def zone_count(self, zone: str, turn: int) -> int:
        """Drones occupying ``zone`` during ``turn``."""
        return self.zone_occ.get((zone, turn), 0)

    def link_count(self, edge: tuple[str, str], turn: int) -> int:
        """Drones traversing ``edge`` during ``turn``."""
        return self.link_occ.get((edge, turn), 0)

    def commit(
        self, zone_res: list[ZoneRes], link_res: list[LinkRes]
    ) -> None:
        """Apply every reservation of one finished trajectory."""
        for zone_key in zone_res:
            self.zone_occ[zone_key] = self.zone_occ.get(zone_key, 0) + 1
        for link_key in link_res:
            self.link_occ[link_key] = self.link_occ.get(link_key, 0) + 1


def _can_occupy_zone(
    table: _ReservationTable, zone: Zone, turn: int
) -> bool:
    """True if one more drone may occupy ``zone`` during ``turn``."""
    if zone.capacity is None:  # start/end hubs: unlimited
        return True
    return table.zone_count(zone.name, turn) + 1 <= zone.capacity


def _can_use_link(
    table: _ReservationTable, edge: tuple[str, str], turn: int, cap: int
) -> bool:
    """True if one more drone may traverse the link during ``turn``."""
    return table.link_count(edge, turn) + 1 <= cap


def _step(
    graph: Graph,
    table: _ReservationTable,
    src: str,
    neighbour: str,
    conn: Connection,
    turn: int,
    horizon: int,
) -> tuple[int, Action] | None:
    """Moving `src` -> `neighbour` during `turn`, if allowed.

    Returns the ready turn on arrival and the action ("move", or
    "transit" into a restricted zone), or None when the neighbour is
    blocked, the link or zone is full, or it ends past the horizon.
    """
    dest = graph.zones[neighbour]
    if dest.zone_type.movement_cost is None:  # blocked
        return None
    edge = _edge_key(src, neighbour)
    if dest.zone_type is ZoneType.RESTRICTED:
        if (
            turn + 2 > horizon
            or not _can_use_link(table, edge, turn, conn.max_link_capacity)
            or not _can_occupy_zone(table, dest, turn + 1)
        ):
            return None
        return turn + 2, "transit"
    if (
        turn + 1 > horizon
        or not _can_use_link(table, edge, turn, conn.max_link_capacity)
        or not _can_occupy_zone(table, dest, turn)
    ):
        return None
    return turn + 1, "move"


def _next_departure(
    graph: Graph,
    table: _ReservationTable,
    start: str,
    after: int,
    horizon: int,
) -> int | None:
    """First turn >= `after`, below the horizon, to leave the start in.

    Reservations only ever grow, so a turn found full for a way out
    stays full: `table.depart_from` keeps, per neighbour, the turn
    below which everything is known full, and later drones skip that
    stretch instead of re-checking it turn by turn.
    """
    found: int | None = None
    for neighbour, conn in graph.neighbors(start):
        if graph.zones[neighbour].zone_type.movement_cost is None:
            continue  # blocked: never a way out
        # Grow the known-full stretch first: each turn is crossed once
        # per neighbour over the whole plan, not once per drone.
        known = table.depart_from.get(neighbour, START_TURN)
        while known < horizon and not _step(
            graph, table, start, neighbour, conn, known, horizon
        ):
            known += 1
        table.depart_from[neighbour] = known
        turn = max(after, known)
        limit = horizon if found is None else found
        while turn < limit and not _step(
            graph, table, start, neighbour, conn, turn, horizon
        ):
            turn += 1
        if turn < limit:
            found = turn
    return found


@dataclass
class _SearchNode:
    """One hop of a drone's route: how a state was reached."""

    state: State
    parent: State | None
    action: Action
    destination: str


def _plan_drone(
    graph: Graph,
    table: _ReservationTable,
    drone_id: int,
    horizon: int,
) -> list[TurnRecord]:
    """Plan one conflict-free trajectory for a single drone.

    Raises:
        SimulationError: if no trajectory reaches the end zone within
            the horizon (possible only on adversarial maps: a drone
            can always wait in the unlimited-capacity start zone).
    """
    assert graph.start is not None and graph.end is not None
    start_name = graph.start.name
    end_name = graph.end.name
    start_state: State = (start_name, START_TURN)

    nodes: dict[State, _SearchNode] = {
        start_state: _SearchNode(start_state, None, "start", start_name)
    }
    # Search key per state: (ready turn, non-priority zone entries).
    best: dict[State, tuple[int, int]] = {start_state: (START_TURN, 0)}
    heap: list[tuple[int, int, int, State]] = [
        (START_TURN, 0, 0, start_state)
    ]
    counter = 1
    goal_state: State | None = None
    finalized: set[State] = set()

    def relax(
        new_state: State,
        new_key: tuple[int, int],
        action: Action,
        destination: str,
        parent: State,
    ) -> None:
        """Record a route to ``new_state`` if it beats the incumbent."""
        nonlocal counter
        if new_state in finalized:
            return
        incumbent = best.get(new_state)
        if incumbent is not None and (
            new_key[0] > incumbent[0]
            or (new_key[0] == incumbent[0] and new_key[1] >= incumbent[1])
        ):
            return
        best[new_state] = new_key
        nodes[new_state] = _SearchNode(
            new_state, parent, action, destination
        )
        heapq.heappush(heap, (new_key[0], new_key[1], counter, new_state))
        counter += 1

    while heap:
        # First two heap fields (ready turn, npc) mirror state keys;
        # unpack the state itself for the turn and route history.
        _, npc, _, state = heapq.heappop(heap)
        if state in finalized:
            continue
        finalized.add(state)
        zone_name, turn = state
        if zone_name == end_name:
            goal_state = state
            break
        if turn >= horizon:
            continue
        zone = graph.zones[zone_name]

        # WAIT: stay put during this turn (occupies the zone). On the
        # unlimited start, skip the idle turns in which no way out has
        # room: from those the only option is to keep waiting.
        if zone_name == start_name:
            leave = _next_departure(
                graph, table, start_name, turn + 1, horizon
            )
            if leave is not None:
                relax(
                    (zone_name, leave), (leave, npc),
                    "wait", zone_name, state,
                )
        elif _can_occupy_zone(table, zone, turn):
            relax(
                (zone_name, turn + 1), (turn + 1, npc),
                "wait", zone_name, state,
            )

        for neighbour, conn in graph.neighbors(zone_name):
            step = _step(
                graph, table, zone_name, neighbour, conn, turn, horizon
            )
            if step is None:
                continue
            new_turn, action = step
            dest = graph.zones[neighbour]
            entry_npc = npc + (
                0 if dest.zone_type is ZoneType.PRIORITY else 1
            )
            relax(
                (neighbour, new_turn), (new_turn, entry_npc),
                action, neighbour, state,
            )

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
    zone_res: list[ZoneRes] = []
    link_res: list[LinkRes] = []
    prev_zone = start_name
    for node in path_nodes[1:]:
        ready = node.state[1]
        if node.action == "wait":
            # On the start a wait needs no record (the output omits
            # waits) and no reservation (the start is unlimited).
            if prev_zone != start_name:
                records.append(TurnRecord(ready - 1, None, prev_zone))
                zone_res.append((prev_zone, ready - 1))
        elif node.action == "move":
            dest_name = node.destination
            edge = _edge_key(prev_zone, dest_name)
            records.append(TurnRecord(ready - 1, dest_name, dest_name))
            link_res.append((edge, ready - 1))
            zone_res.append((dest_name, ready - 1))
            prev_zone = dest_name
        elif node.action == "transit":
            dest_name = node.destination
            edge = _edge_key(prev_zone, dest_name)
            records.append(
                TurnRecord(ready - 2, f"{prev_zone}-{dest_name}", None)
            )
            records.append(TurnRecord(ready - 1, dest_name, dest_name))
            link_res.append((edge, ready - 2))
            zone_res.append((dest_name, ready - 1))
            prev_zone = dest_name

    table.commit(zone_res, link_res)
    return records


def _plan_all(graph: Graph, horizon: int) -> list[Drone]:
    """Plan every drone in ascending id order against one shared table.

    All drones are interchangeable (same start zone, same goal), so
    the planning order cannot change the outcome; a single pass
    suffices.
    """
    table = _ReservationTable()
    drones: list[Drone] = []
    for drone_id in range(1, graph.num_drones + 1):
        records = _plan_drone(graph, table, drone_id, horizon)
        drones.append(Drone(drone_id=drone_id, trajectory=records))
    return drones


def _base_cost(graph: Graph) -> int:
    """Static shortest-path cost in turns from start to end.

    Raises:
        SimulationError: if the end zone is unreachable.
    """
    assert graph.start is not None and graph.end is not None
    result = dijkstra(graph, graph.start.name)
    if graph.end.name not in result.distances:
        raise SimulationError("end zone is unreachable from start")
    return result.distances[graph.end.name]


def plan(graph: Graph) -> list[Drone]:
    """Plan trajectories for all drones in a single ascending pass.

    Raises:
        SimulationError: if the end zone is unreachable, the fleet is
            too large to route within the bounded horizon, or a drone
            cannot be routed within the search horizon.
    """
    graph.validate()
    base = _base_cost(graph)
    n = graph.num_drones
    cap = max(FLOOR_HORIZON, MAX_STATES // max(1, len(graph.zones)))
    horizon = min(cap, base * 2 + n * (base + 1) + 64)
    # Worst case, the last drone waits for all n-1 predecessors to
    # clear the network, needing about base + n turns. If even that
    # does not fit the bounded horizon, no plan can exist - fail fast
    # instead of looping over an impossible fleet.
    if base + n > horizon:
        raise SimulationError(
            f"too many drones ({n}): routing the fleet needs a "
            f"horizon of at least {base + n} turns, bounded here to "
            f"{horizon} by the per-map state budget"
        )
    return _plan_all(graph, horizon)
