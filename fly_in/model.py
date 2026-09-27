"""Core data model: zones, connections, graph and drones."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class ZoneType(Enum):
    """Movement-cost classification for a zone."""

    NORMAL = "normal"
    BLOCKED = "blocked"
    RESTRICTED = "restricted"
    PRIORITY = "priority"

    @staticmethod
    def from_str(value: str) -> "ZoneType":
        """Convert a raw string into a ZoneType or raise ValueError."""
        try:
            return ZoneType(value)
        except ValueError:
            raise ValueError(f"invalid zone type '{value}'") from None

    @property
    def movement_cost(self) -> int:
        """Number of turns required to enter this zone."""
        if self is ZoneType.RESTRICTED:
            return 2
        if self is ZoneType.BLOCKED:
            return -1
        return 1


@dataclass(frozen=True)
class Zone:
    """A single zone (vertex) of the routing graph."""

    name: str
    x: int
    y: int
    zone_type: ZoneType = ZoneType.NORMAL
    color: str | None = None
    max_drones: int = 1
    is_start: bool = False
    is_end: bool = False

    @property
    def capacity(self) -> int:
        """Effective occupancy capacity (start/end are unlimited)."""
        if self.is_start or self.is_end:
            return -1
        return self.max_drones


@dataclass(frozen=True)
class Connection:
    """A bidirectional edge between two zones."""

    src: str
    dst: str
    max_link_capacity: int = 1

    def key(self) -> tuple[str, str]:
        """Canonical (sorted) key so a-b and b-a collide."""
        a, b = self.src, self.dst
        return (a, b) if a <= b else (b, a)


@dataclass
class Drone:
    """A drone to be routed from start to end."""

    drone_id: int
    trajectory: list["TurnRecord"] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Human-readable identifier used in simulation output."""
        return f"D{self.drone_id}"


@dataclass
class TurnRecord:
    """What a drone does during a single simulation turn.

    ``token`` is the output string placed after ``D<id>-`` (a zone name for a
    move/arrival, a ``src-dst`` connection name while in transit, or ``None``
    when the drone waits and produces no output for that turn).
    """

    turn: int
    token: str | None
    location: str | None


@dataclass
class Graph:
    """Adjacency-list graph built from parsed zones and connections."""

    zones: dict[str, Zone] = field(default_factory=dict)
    connections: dict[tuple[str, str], Connection] = field(default_factory=dict)
    adjacency: dict[str, list[tuple[str, Connection]]] = field(
        default_factory=dict
    )
    start: Zone | None = None
    end: Zone | None = None
    num_drones: int = 0

    def add_zone(self, zone: Zone) -> None:
        """Register a zone, rejecting duplicates."""
        if zone.name in self.zones:
            raise ValueError(f"duplicate zone name '{zone.name}'")
        self.zones[zone.name] = zone
        self.adjacency.setdefault(zone.name, [])
        if zone.is_start:
            if self.start is not None:
                raise ValueError("multiple start_hub zones")
            self.start = zone
        if zone.is_end:
            if self.end is not None:
                raise ValueError("multiple end_hub zones")
            self.end = zone

    def add_connection(self, conn: Connection) -> None:
        """Register a bidirectional connection, rejecting duplicates."""
        if conn.src not in self.zones:
            raise ValueError(f"connection references unknown zone '{conn.src}'")
        if conn.dst not in self.zones:
            raise ValueError(f"connection references unknown zone '{conn.dst}'")
        if conn.src == conn.dst:
            raise ValueError("a zone cannot connect to itself")
        key = conn.key()
        if key in self.connections:
            raise ValueError(f"duplicate connection '{conn.src}-{conn.dst}'")
        self.connections[key] = conn
        self.adjacency[conn.src].append((conn.dst, conn))
        self.adjacency[conn.dst].append((conn.src, conn))

    def neighbors(self, name: str) -> Iterator[tuple[str, Connection]]:
        """Yield (neighbor_name, connection) pairs for a zone."""
        return iter(self.adjacency.get(name, ()))

    def validate(self) -> None:
        """Ensure the graph has exactly one start and one end zone."""
        if self.start is None:
            raise ValueError("missing start_hub zone")
        if self.end is None:
            raise ValueError("missing end_hub zone")
        if self.num_drones <= 0:
            raise ValueError("nb_drones must be a positive integer")
