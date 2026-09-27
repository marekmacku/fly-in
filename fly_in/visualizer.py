"""Colored terminal visualization of the simulation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from fly_in.model import Graph, Zone, ZoneType
from fly_in.simulator import SimulationResult, label_for

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
    if not color:
        return text
    return f"{color}{text}{RESET}"


@dataclass
class _Cell:
    zone: Zone | None
    drones: list[int]


class Visualizer:
    """Renders colored per-turn state and movement lines."""

    def __init__(self, graph: Graph, stream: TextIO | None = None) -> None:
        self._graph = graph
        self._stream = stream if stream is not None else sys.stdout

    def _grid_size(self) -> tuple[int, int, int, int]:
        xs = [z.x for z in self._graph.zones.values()]
        ys = [z.y for z in self._graph.zones.values()]
        return min(xs), max(xs), min(ys), max(ys)

    def _build_grid(
        self, positions: dict[int, str | None]
    ) -> dict[tuple[int, int], _Cell]:
        cells: dict[tuple[int, int], _Cell] = {
            (z.x, z.y): _Cell(zone=z, drones=[])
            for z in self._graph.zones.values()
        }
        for drone_id, loc in positions.items():
            if loc is None:
                continue
            zone = self._graph.zones.get(loc)
            if zone is None:
                continue
            cell = cells.get((zone.x, zone.y))
            if cell is not None:
                cell.drones.append(drone_id)
        return cells

    def _render_grid(self, positions: dict[int, str | None]) -> str:
        if not self._graph.zones:
            return ""
        min_x, max_x, min_y, max_y = self._grid_size()
        cells = self._build_grid(positions)
        lines: list[str] = []
        for y in range(max_y, min_y - 1, -1):
            row: list[str] = []
            for x in range(min_x, max_x + 1):
                cell = cells.get((x, y))
                if cell is None or cell.zone is None:
                    row.append("   .  ")
                    continue
                zone = cell.zone
                symbol = _zone_symbol(zone)
                color = color_for(zone.color)
                if cell.drones:
                    labels = ",".join(
                        label_for(d) for d in sorted(cell.drones)
                    )
                    row.append(
                        _wrap(f"[{symbol}{labels}]", _DRONE_COLOR)
                    )
                else:
                    row.append(_wrap(f" {symbol}   ", color))
            lines.append("".join(row))
        return "\n".join(lines)

    def render_legend(self) -> str:
        """Build a compact legend of zone types and colors."""
        parts: list[str] = []
        for zone in self._graph.zones.values():
            color = color_for(zone.color)
            parts.append(_wrap(zone.name, color or BOLD))
        return "Zones: " + ", ".join(parts)

    def render(
        self,
        result: SimulationResult,
        positions_per_turn: list[dict[int, str | None]],
        show_grid: bool,
        colored: bool,
    ) -> None:
        """Render the full simulation to the configured stream."""
        out = self._stream
        if show_grid:
            out.write(self.render_legend() + "\n\n")
        for index, line in enumerate(result.turns):
            turn = index + 1
            positions = (
                positions_per_turn[turn]
                if turn < len(positions_per_turn)
                else {}
            )
            if show_grid:
                header = (
                    f"{BOLD}--- Turn {turn} ---{RESET}"
                    if colored
                    else f"--- Turn {turn} ---"
                )
                out.write(header + "\n")
                out.write(self._render_grid(positions) + "\n")
            if colored:
                out.write(
                    f"{BOLD}Turn {turn}{RESET}: "
                    + _colorize_line(line)
                    + "\n"
                )
            else:
                out.write(f"Turn {turn}: {line}\n")
        if colored:
            out.write(
                f"\n{BOLD}Total turns: {result.makespan}{RESET}\n"
            )
        else:
            out.write(f"\nTotal turns: {result.makespan}\n")


def _zone_symbol(zone: Zone) -> str:
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
            out.append(f"{_DRONE_COLOR}{label}{RESET}-"
                       f"{BOLD}{rest}{RESET}")
        else:
            out.append(token)
    return " ".join(out)
