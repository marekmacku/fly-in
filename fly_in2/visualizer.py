"""Colored terminal visualization of the simulation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from fly_in2.model import (
    Drone,
    Graph,
    SimulationResult,
    TurnRecord,
    Zone,
    ZoneType,
    drone_label,
)

RESET = "\033[0m"
BOLD = "\033[1m"

_COLOR_TABLE: dict[str, str] = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
    "orange": "\033[38;5;208m",
    "purple": "\033[38;5;129m",
    "pink": "\033[38;5;213m",
    "brown": "\033[38;5;130m",
    "lime": "\033[38;5;154m",
    "gold": "\033[38;5;220m",
    "maroon": "\033[38;5;88m",
    "crimson": "\033[38;5;160m",
    "violet": "\033[38;5;177m",
    "rainbow": "\033[38;5;201m",
    "darkred": "\033[38;5;52m",
    "black": "\033[90m",
    "white": "\033[97m",
}

_DRONE_COLOR = "\033[36m"


def color_for(name: str | None) -> str:
    """Return an ANSI escape for a color name (empty => no color)."""
    if not name:
        return ""
    return _COLOR_TABLE.get(name.lower(), "")


def _wrap(text: str, color: str) -> str:
    """Wrap ``text`` in an ANSI color, or return it unchanged."""
    if not color:
        return text
    return f"{color}{text}{RESET}"


def _zone_symbol(zone: Zone) -> str:
    """Single-character grid symbol for a zone."""
    if zone.is_start:
        return "S"
    if zone.is_end:
        return "E"
    if zone.zone_type is ZoneType.BLOCKED:
        return "#"
    if zone.zone_type is ZoneType.RESTRICTED:
        return "R"
    if zone.zone_type is ZoneType.PRIORITY:
        return "P"
    return "O"


def _colorize_line(line: str) -> str:
    """Colorize a plain ``D<id>-<token>`` movement line."""
    if not line:
        return ""
    out: list[str] = []
    for token in line.split(" "):
        if "-" in token:
            label, _, rest = token.partition("-")
            out.append(
                f"{_DRONE_COLOR}{label}{RESET}-{BOLD}{rest}{RESET}"
            )
        else:
            out.append(token)
    return " ".join(out)


@dataclass
class _Cell:
    """One grid cell: the zone sitting there plus its drones."""

    zone: Zone | None
    drones: list[int]


def positions_per_turn(
    graph: Graph, drones: list[Drone], n_turns: int
) -> list[dict[int, str | None]]:
    """Each drone's zone per turn (None while in transit).

    Index ``t`` (1-based) maps every active drone to the zone it ends
    turn ``t`` in; delivered drones stay parked at the end zone.
    """
    assert graph.start is not None and graph.end is not None
    positions: list[dict[int, str | None]] = [
        {} for _ in range(n_turns + 2)
    ]
    for drone in drones:
        current: str | None = graph.start.name
        records_by_turn: dict[int, TurnRecord] = {
            rec.turn: rec for rec in drone.trajectory
        }
        delivery = 0
        for record in drone.trajectory:
            if record.token is not None and record.turn > delivery:
                delivery = record.turn
        for turn in range(1, n_turns + 1):
            if delivery > 0 and turn > delivery:
                positions[turn][drone.drone_id] = graph.end.name
                continue
            rec = records_by_turn.get(turn)
            if rec is None:
                positions[turn][drone.drone_id] = current
            elif rec.location is not None:
                current = rec.location
                positions[turn][drone.drone_id] = current
            else:
                positions[turn][drone.drone_id] = None
    return positions


class Visualizer:
    """Renders colored per-turn state and movement lines."""

    def __init__(self, graph: Graph, stream: TextIO | None = None) -> None:
        self._graph = graph
        self._stream = stream if stream is not None else sys.stdout

    def _grid_bounds(self) -> tuple[int, int, int, int]:
        """Return (min_x, max_x, min_y, max_y) over all zones."""
        xs = [z.x for z in self._graph.zones.values()]
        ys = [z.y for z in self._graph.zones.values()]
        return min(xs), max(xs), min(ys), max(ys)

    def _build_grid(
        self, positions: dict[int, str | None]
    ) -> dict[tuple[int, int], _Cell]:
        """Place zones and drones on the coordinate grid."""
        cells: dict[tuple[int, int], _Cell] = {
            (z.x, z.y): _Cell(zone=z, drones=[])
            for z in self._graph.zones.values()
        }
        for drone_id, loc in positions.items():
            if loc is None:
                continue  # in transit: not on the grid this turn
            zone = self._graph.zones.get(loc)
            if zone is None:
                continue
            cell = cells.get((zone.x, zone.y))
            if cell is not None:
                cell.drones.append(drone_id)
        return cells

    def _render_grid(self, positions: dict[int, str | None]) -> str:
        """Draw the grid for one turn (top row = highest y)."""
        if not self._graph.zones:
            return ""
        min_x, max_x, min_y, max_y = self._grid_bounds()
        cells = self._build_grid(positions)
        lines: list[str] = []
        for y in range(max_y, min_y - 1, -1):
            row: list[str] = []
            for x in range(min_x, max_x + 1):
                cell = cells.get((x, y))
                if cell is None or cell.zone is None:
                    row.append("   .  ")
                    continue
                symbol = _zone_symbol(cell.zone)
                color = color_for(cell.zone.color)
                if cell.drones:
                    labels = ",".join(
                        drone_label(d) for d in sorted(cell.drones)
                    )
                    row.append(
                        _wrap(f"[{symbol}{labels}]", _DRONE_COLOR)
                    )
                else:
                    row.append(_wrap(f" {symbol}   ", color))
            lines.append("".join(row))
        return "\n".join(lines)

    def render_legend(self) -> str:
        """Build a compact legend of zone names and colors."""
        parts: list[str] = []
        for zone in self._graph.zones.values():
            color = color_for(zone.color)
            parts.append(_wrap(zone.name, color or BOLD))
        return "Zones: " + ", ".join(parts)

    def render(
        self,
        result: SimulationResult,
        drones: list[Drone],
        show_grid: bool,
        colored: bool,
    ) -> None:
        """Render the full, already-validated simulation."""
        out = self._stream
        positions = positions_per_turn(
            self._graph, drones, len(result.turns)
        )
        if show_grid:
            out.write(self.render_legend() + "\n\n")
        for index, line in enumerate(result.turns):
            turn = index + 1
            if show_grid:
                header = (
                    f"{BOLD}--- Turn {turn} ---{RESET}"
                    if colored
                    else f"--- Turn {turn} ---"
                )
                out.write(header + "\n")
                out.write(self._render_grid(positions[turn]) + "\n")
            if colored:
                out.write(
                    f"{BOLD}Turn {turn}{RESET}: "
                    + _colorize_line(line)
                    + "\n"
                )
            else:
                out.write(f"Turn {turn}: {line}\n")
        if colored:
            out.write(f"\n{BOLD}Total turns: {result.makespan}{RESET}\n")
        else:
            out.write(f"\nTotal turns: {result.makespan}\n")
