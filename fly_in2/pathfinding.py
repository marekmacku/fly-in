"""Static pathfinding: Dijkstra implemented without graph libraries."""

from __future__ import annotations

import heapq
from typing import NamedTuple

from fly_in2.model import Graph, ZoneType


class ShortestPathResult(NamedTuple):
    """Single-source shortest path outcome.

    ``distances`` holds exact integer movement costs and contains only
    reachable zones; a zone missing from it is unreachable from the
    source. ``previous`` maps each reached zone to its predecessor on
    the best path (None for the source itself).
    """

    distances: dict[str, int]
    previous: dict[str, str | None]

    def reconstruct(self, target: str) -> list[str] | None:
        """Return the zone sequence from source to target, or None.

        None means the target is unreachable.
        """
        if target not in self.distances:
            return None
        path: list[str] = []
        cursor: str | None = target
        while cursor is not None:
            path.append(cursor)
            cursor = self.previous[cursor]
        path.reverse()
        return path


def dijkstra(graph: Graph, source: str) -> ShortestPathResult:
    """Run Dijkstra from ``source``, weighted by zone entry cost.

    Edge weight is the destination zone's movement_cost (1 or 2;
    blocked zones return None and are never entered). Distances are
    exact integers.

    To honour the spec's "priority zones should be prioritized" rule,
    equal-cost ties are broken toward paths stepping through fewer
    non-priority zones: each state carries a secondary non_priority
    count used only as a heap tie-breaker, never added to distances.
    """
    distances: dict[str, int] = {}
    non_priority: dict[str, int] = {}
    previous: dict[str, str | None] = {}
    if source not in graph.zones:
        return ShortestPathResult(distances, previous)
    distances[source] = 0
    non_priority[source] = 0
    previous[source] = None
    heap: list[tuple[int, int, int, str]] = [(0, 0, 0, source)]
    counter = 0
    while heap:
        dist, npc, _, zone_name = heapq.heappop(heap)
        # Skip stale entries: a worse cost, or an equal cost that has
        # since been re-relaxed with a better tie-break.
        if (dist > distances[zone_name]
                or (dist == distances[zone_name]
                    and npc > non_priority[zone_name])):
            continue
        for neighbour, _conn in graph.neighbors(zone_name):
            zone = graph.zones[neighbour]
            cost = zone.zone_type.movement_cost
            if cost is None:  # blocked: cannot be entered
                continue
            candidate_dist = dist + cost
            candidate_npc = npc + (
                0 if zone.zone_type is ZoneType.PRIORITY else 1
            )
            better = (
                neighbour not in distances
                or candidate_dist < distances[neighbour]
                or (
                    candidate_dist == distances[neighbour]
                    and candidate_npc < non_priority[neighbour]
                )
            )
            if better:
                distances[neighbour] = candidate_dist
                non_priority[neighbour] = candidate_npc
                previous[neighbour] = zone_name
                counter += 1
                heapq.heappush(
                    heap, (candidate_dist, candidate_npc, counter, neighbour)
                )
    return ShortestPathResult(distances, previous)


def shortest_path(graph: Graph, source: str, target: str) -> list[str] | None:
    """Return the best zone sequence from source to target, or None.

    None means the target is unreachable from the source.
    """
    if source not in graph.zones or target not in graph.zones:
        return None
    return dijkstra(graph, source).reconstruct(target)
