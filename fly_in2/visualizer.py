"""Colored terminal visualization of the simulation."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Iterable, TextIO

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

# Coordinate gaps up to this many units are drawn true to the file.
_GAP_KEEP = 4
# No gap, however absurd, is drawn longer than this.
_GAP_MAX = 8


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
    """One grid cell: the zone sitting there plus its drones.

    `stacked` marks a cell several zones share; it then holds the
    drones of all of them and draws as `*`.
    """

    zone: Zone | None
    drones: list[int]
    stacked: bool = False


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


def _squash(values: Iterable[int]) -> dict[int, int]:
    """Drawn position of each distinct coordinate value, from 0.

    Gaps up to _GAP_KEEP stay as in the file; a longer gap g is drawn
    _GAP_KEEP + log2(g / _GAP_KEEP) long, capped at _GAP_MAX: 8 takes 5
    units, 16 takes 6, 32 takes 7 and anything from 64 up takes 8. Order
    is kept, so the picture still reads left to right and bottom to top,
    but an absurd coordinate can no longer push a zone off screen or blow
    up the grid.
    """
    drawn: dict[int, int] = {}
    pos = 0
    prev: int | None = None
    for value in sorted(set(values)):
        if prev is not None:
            gap = value - prev
            if gap > _GAP_KEEP:
                gap = min(
                    _GAP_MAX,
                    # log2 of the int itself: gap / _GAP_KEEP would
                    # overflow a float once the gap passes ~1e308.
                    round(
                        _GAP_KEEP + math.log2(gap) - math.log2(_GAP_KEEP)
                    ),
                )
            pos += gap
        drawn[value] = pos
        prev = value
    return drawn


def display_positions(graph: Graph) -> dict[str, tuple[int, int]]:
    """Where each zone is drawn: its (x, y), long gaps squashed.

    Coordinates never affect movement, only drawing, so both the
    terminal grid and the GUI place zones from this.
    """
    xs = _squash(z.x for z in graph.zones.values())
    ys = _squash(z.y for z in graph.zones.values())
    return {name: (xs[z.x], ys[z.y]) for name, z in graph.zones.items()}


class Visualizer:
    """Renders colored per-turn state and movement lines."""

    def __init__(self, graph: Graph, stream: TextIO | None = None) -> None:
        self._graph = graph
        self._stream = stream if stream is not None else sys.stdout
        self._at = display_positions(graph)

    def _grid_bounds(self) -> tuple[int, int, int, int]:
        """Return (min_x, max_x, min_y, max_y) over the drawn cells."""
        xs = [x for x, _ in self._at.values()]
        ys = [y for _, y in self._at.values()]
        return min(xs), max(xs), min(ys), max(ys)

    def _build_grid(
        self, positions: dict[int, str | None]
    ) -> dict[tuple[int, int], _Cell]:
        """Place zones and drones on the coordinate grid."""
        cells: dict[tuple[int, int], _Cell] = {}
        for z in self._graph.zones.values():
            if self._at[z.name] in cells:
                cells[self._at[z.name]].stacked = True
            else:
                cells[self._at[z.name]] = _Cell(zone=z, drones=[])
        for drone_id, loc in positions.items():
            if loc is None:
                continue  # in transit: not on the grid this turn
            at = self._at.get(loc)
            if at is None:
                continue
            cell = cells.get(at)
            if cell is not None:
                cell.drones.append(drone_id)
        return cells

    def _render_grid(
        self, positions: dict[int, str | None], colored: bool
    ) -> str:
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
                symbol = "*" if cell.stacked else _zone_symbol(cell.zone)
                color = color_for(cell.zone.color)
                if cell.stacked or not colored:
                    color = ""
                if cell.drones:
                    labels = ",".join(
                        drone_label(d) for d in sorted(cell.drones)
                    )
                    row.append(_wrap(
                        f"[{symbol}{labels}]",
                        _DRONE_COLOR if colored else "",
                    ))
                else:
                    row.append(_wrap(f" {symbol}   ", color))
            lines.append("".join(row))
        return "\n".join(lines)

    def render_legend(self, colored: bool) -> str:
        """Build a compact legend of zone names and colors.

        Zones sharing a position get an extra line, since the grid
        can only draw them as one `*` cell.
        """
        parts: list[str] = []
        for zone in self._graph.zones.values():
            color = color_for(zone.color) or BOLD
            parts.append(_wrap(zone.name, color if colored else ""))
        legend = "Zones: " + ", ".join(parts)
        for names in self._graph.stacked_zones():
            legend += "\n* shared by: " + ", ".join(names)
        return legend

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
            out.write(self.render_legend(colored) + "\n\n")
        for index, line in enumerate(result.turns):
            turn = index + 1
            if show_grid:
                header = (
                    f"{BOLD}--- Turn {turn} ---{RESET}"
                    if colored
                    else f"--- Turn {turn} ---"
                )
                out.write(header + "\n")
                out.write(
                    self._render_grid(positions[turn], colored) + "\n"
                )
            if colored:
                out.write(
                    f"{BOLD}Turn {turn}{RESET}: "
                    + _colorize_line(line)
                    + "\n"
                )
            elif show_grid:
                out.write(f"Turn {turn}: {line}\n")
            else:
                out.write(line + "\n")   # plain: the spec's bare format
        if colored:
            out.write(f"\n{BOLD}Total turns: {result.makespan}{RESET}\n")
        elif show_grid:
            out.write(f"\nTotal turns: {result.makespan}\n")
