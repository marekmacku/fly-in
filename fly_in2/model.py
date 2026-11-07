"""Core data model: zones, connections, the graph, drones and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class ZoneType(Enum):
    """Movement classification of a zone, from the map's ``zone=`` tag."""

    NORMAL = "normal"
    BLOCKED = "blocked"
    RESTRICTED = "restricted"
    PRIORITY = "priority"

    @staticmethod
    def from_str(value: str) -> "ZoneType":
        """Convert a raw map string into a ZoneType.

        Raises:
            ValueError: if the string is not one of the four zone types.
        """
        try:
            return ZoneType(value)
        except ValueError:
            raise ValueError(f"invalid zone type '{value}'") from None

    @property
    def movement_cost(self) -> int | None:
        """Turns needed to enter this zone, or None if it cannot be."""
        if self is ZoneType.BLOCKED:
            return None
        if self is ZoneType.RESTRICTED:
            return 2
        return 1


@dataclass(frozen=True)
class Zone:
    """A zone (graph vertex); frozen so instances can be shared safely."""

    name: str
    x: int
    y: int
    zone_type: ZoneType = ZoneType.NORMAL
    color: str | None = None
    max_drones: int = 1
    is_start: bool = False
    is_end: bool = False

    @property
    def capacity(self) -> int | None:
        """Simultaneous drones allowed, or None when unlimited.

        The spec gives start/end hubs no occupancy limit, so any
        max_drones tag on them is deliberately ignored here.
        """
        if self.is_start or self.is_end:
            return None
        return self.max_drones


@dataclass(frozen=True)
class Connection:
    """A bidirectional edge between two zones, referenced by name.

    Names (not Zone objects) keep the model cycle-free and mirror the
    file format; the Graph resolves them when the edge is registered.
    """

    src: str
    dst: str
    max_link_capacity: int = 1

    def key(self) -> tuple[str, str]:
        """Canonical sorted endpoint pair so a-b and b-a collide."""
        a, b = self.src, self.dst
        return (a, b) if a <= b else (b, a)


@dataclass
class TurnRecord:
    """What one drone does during one turn.

    ``token`` is the output string placed after ``D<id>-`` (a zone
    name, or a ``src-dst`` name while in transit); None means the
    drone waited and is omitted from that turn's output line.
    ``location`` is where the drone ends the turn (None only while in
    transit); the visualizer and validator use it, it is never printed.
    """

    turn: int
    token: str | None
    location: str | None


@dataclass
class Drone:
    """A drone routed from the start zone to the end zone."""

    drone_id: int
    trajectory: list[TurnRecord] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Identifier used in simulation output, e.g. 'D3'."""
        return f"D{self.drone_id}"


@dataclass
class SimulationResult:
    """What the simulator produces: one line per turn plus the makespan."""

    turns: list[str] = field(default_factory=list)
    makespan: int = 0


@dataclass
class Graph:
    """Adjacency-list graph assembled from zones and connections.

    add_zone and add_connection enforce the locally checkable rules
    (unique names, known endpoints, no duplicate edges); validate()
    enforces the global ones (exactly one start and one end hub, a
    positive drone count). The parser turns every ValueError into a
    ParseError stamped with the offending line number.
    """

    zones: dict[str, Zone] = field(default_factory=dict)
    connections: dict[tuple[str, str], Connection] = field(
        default_factory=dict
    )
    adjacency: dict[str, list[tuple[str, Connection]]] = field(
        default_factory=dict
    )
    start: Zone | None = None
    end: Zone | None = None
    num_drones: int = 0

    def add_zone(self, zone: Zone) -> None:
        """Register a zone.

        Raises:
            ValueError: on a duplicate name or a second start/end hub.
        """
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
        """Register a bidirectional edge between two known zones.

        Raises:
            ValueError: if an endpoint is unknown, the edge is a
                self-loop, or the same edge was already added.
        """
        for endpoint in (conn.src, conn.dst):
            if endpoint not in self.zones:
                raise ValueError(
                    f"connection references unknown zone '{endpoint}'"
                )
        if conn.src == conn.dst:
            raise ValueError("a zone cannot connect to itself")
        key = conn.key()
        if key in self.connections:
            raise ValueError(
                f"duplicate connection '{conn.src}-{conn.dst}'"
            )
        self.connections[key] = conn
        self.adjacency[conn.src].append((conn.dst, conn))
        self.adjacency[conn.dst].append((conn.src, conn))

    def neighbors(self, name: str) -> Iterator[tuple[str, Connection]]:
        """Yield (neighbor_name, connection) pairs for a zone."""
        return iter(self.adjacency.get(name, ()))

    def validate(self) -> None:
        """Check the global invariants the line-by-line rules cannot see.

        Raises:
            ValueError: if a start or end hub is missing, or the
                drone count is not a positive integer.
        """
        if self.start is None:
            raise ValueError("missing start_hub zone")
        if self.end is None:
            raise ValueError("missing end_hub zone")
        if self.num_drones <= 0:
            raise ValueError("nb_drones must be a positive integer")
