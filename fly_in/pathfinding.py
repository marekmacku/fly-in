"""Pathfinding primitives implemented without graph libraries."""

from __future__ import annotations

import heapq
from typing import NamedTuple

from fly_in.model import Graph, Zone, ZoneType

INF = float("inf")


class ShortestPathResult(NamedTuple):
    """Outcome of a single-source shortest path search."""

    distances: dict[str, float]
    previous: dict[str, str | None]

    def reconstruct(self, target: str) -> list[str] | None:
        """Reconstruct the zone path to ``target`` or None if unreachable."""
        if target not in self.distances:
            return None
        path: list[str] = []
        cursor: str | None = target
        while cursor is not None:
            path.append(cursor)
            cursor = self.previous[cursor]
        path.reverse()
        return path


def _edge_cost(destination: Zone) -> float:
    """Turn cost to enter ``destination`` (blocked => unreachable)."""
    if destination.zone_type is ZoneType.BLOCKED:
        return INF
    return float(destination.zone_type.movement_cost)


def dijkstra(graph: Graph, source: str) -> ShortestPathResult:
    """Run Dijkstra from ``source`` weighted by zone entry cost.

    Blocked zones are never entered. Distances are exact movement costs. To
    honour the spec's "priority zones should be prioritized" rule, equal-cost
    ties are broken toward paths that step through more priority zones: each
    state tracks a secondary ``non_priority`` count that is only used as a
    heap tie-breaker, never added to the distance.
    """
    distances: dict[str, float] = {name: INF for name in graph.zones}
    non_priority: dict[str, int] = {name: 0 for name in graph.zones}
    previous: dict[str, str | None] = {name: None for name in graph.zones}
    if source not in graph.zones:
        return ShortestPathResult(distances, previous)
    distances[source] = 0.0
    heap: list[tuple[float, int, int, str]] = [(0.0, 0, 0, source)]
    counter = 0
    while heap:
        dist, _npc, _, name = heapq.heappop(heap)
        if dist > distances[name]:
            continue
        for neighbour, _conn in graph.neighbors(name):
            zone = graph.zones[neighbour]
            if zone.zone_type is ZoneType.BLOCKED:
                continue
            step = _edge_cost(zone)
            if step == INF:
                continue
            candidate_dist = dist + step
            candidate_npc = non_priority[name] + (
                0 if zone.zone_type is ZoneType.PRIORITY else 1
            )
            better = candidate_dist < distances[neighbour] or (
                candidate_dist == distances[neighbour]
                and candidate_npc < non_priority[neighbour]
            )
            if better:
                distances[neighbour] = candidate_dist
                non_priority[neighbour] = candidate_npc
                previous[neighbour] = name
                counter += 1
                heapq.heappush(
                    heap, (candidate_dist, candidate_npc, counter, neighbour)
                )
    return ShortestPathResult(distances, previous)


def shortest_path(graph: Graph, source: str, target: str) -> list[str] | None:
    """Return the shortest zone sequence from source to target or None."""
    if source not in graph.zones or target not in graph.zones:
        return None
    result = dijkstra(graph, source)
    if result.distances.get(target, INF) == INF:
        return None
    return result.reconstruct(target)
