"""Animated pygame visualization of the simulation (optional --gui).

pygame is imported lazily inside launch(), so the program runs fine
without it; the CLI falls back to the colored terminal renderer.

Only existing zones are drawn (not every grid cell), so huge or
absurd coordinate spans degrade gracefully instead of hanging.
"""

from __future__ import annotations

from typing import Any

from fly_in2.model import Drone, Graph, SimulationResult, Zone, ZoneType
from fly_in2.visualizer import positions_per_turn

_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (200, 60, 60), "green": (60, 170, 80),
    "yellow": (220, 190, 60), "blue": (70, 110, 220),
    "magenta": (190, 70, 190), "cyan": (70, 190, 190),
    "gray": (120, 120, 125), "orange": (235, 140, 50),
    "purple": (140, 80, 200), "pink": (230, 130, 170),
    "brown": (140, 90, 50), "lime": (140, 220, 70),
    "gold": (230, 190, 60), "maroon": (130, 45, 45),
    "crimson": (190, 30, 60), "violet": (150, 90, 220),
    "rainbow": (200, 60, 220), "darkred": (100, 30, 30),
    "black": (35, 35, 40), "white": (235, 235, 235),
}

_TYPE_RGB: dict[ZoneType, tuple[int, int, int]] = {
    ZoneType.NORMAL: (95, 105, 125),
    ZoneType.PRIORITY: (70, 130, 230),
    ZoneType.RESTRICTED: (200, 110, 50),
    ZoneType.BLOCKED: (45, 45, 50),
}

_TYPE_LETTER: dict[ZoneType, str] = {
    ZoneType.NORMAL: "O",
    ZoneType.PRIORITY: "P",
    ZoneType.RESTRICTED: "R",
    ZoneType.BLOCKED: "#",
}

_BG = (18, 18, 24)
_FG = (230, 230, 235)
_LINK = (70, 70, 85)
_DRONE_RGB = (250, 220, 90)
_DRONE_BORDER = (25, 25, 30)
_MARGIN = 40
_HEADER = 46
_MAX_W, _MAX_H = 1920, 1080


def _zone_rgb(zone: Zone) -> tuple[int, int, int]:
    """Fill color: the zone's color tag, or a per-type fallback."""
    if zone.color is not None and zone.color.lower() in _COLOR_RGB:
        return _COLOR_RGB[zone.color.lower()]
    if zone.is_start:
        return (80, 180, 90)
    if zone.is_end:
        return (230, 90, 90)
    return _TYPE_RGB[zone.zone_type]


def _spans(graph: Graph) -> tuple[int, int]:
    """Width and height of the map in zone-coordinate units."""
    xs = [z.x for z in graph.zones.values()]
    ys = [z.y for z in graph.zones.values()]
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def _cell_size(graph: Graph) -> int:
    """Cell side in pixels, scaled to fit the window budget."""
    span_x, span_y = _spans(graph)
    cell = min(
        80,
        (_MAX_W - 2 * _MARGIN) // max(1, span_x),
        (_MAX_H - 2 * _MARGIN - _HEADER) // max(1, span_y),
    )
    return max(1, cell)


def _zone_pixels(
    graph: Graph, cell: int
) -> dict[str, tuple[int, int]]:
    """Pixel centre of each zone (y flipped: high y is up)."""
    xs = [z.x for z in graph.zones.values()]
    ys = [z.y for z in graph.zones.values()]
    min_x, max_y = min(xs), max(ys)
    return {
        name: (
            _MARGIN + (z.x - min_x) * cell + cell // 2,
            _MARGIN + (max_y - z.y) * cell + cell // 2,
        )
        for name, z in graph.zones.items()
    }


def _transit_edges(
    drones: list[Drone],
) -> dict[int, dict[int, tuple[str, str]]]:
    """Per turn: drone id -> (src, dst) of the link it is flying over."""
    transit: dict[int, dict[int, tuple[str, str]]] = {}
    for drone in drones:
        for rec in drone.trajectory:
            if rec.location is None and rec.token is not None:
                src, _, dst = rec.token.partition("-")
                transit.setdefault(rec.turn, {})[drone.drone_id] = (
                    src, dst,
                )
    return transit


def _drone_label(zone: Zone) -> str:
    """One-letter grid symbol, S/E winning over the type letter."""
    if zone.is_start:
        return "S"
    if zone.is_end:
        return "E"
    return _TYPE_LETTER[zone.zone_type]


def _draw_frame(
    pygame: Any,
    screen: Any,
    fonts: tuple[Any, Any],
    graph: Graph,
    pixels: dict[str, tuple[int, int]],
    cell: int,
    turn: int,
    last_turn: int,
    makespan: int,
    positions: dict[int, str | None],
    transit: dict[int, tuple[str, str]],
) -> None:
    """Render one turn: links, zones, drones and the header."""
    screen.fill(_BG)
    for conn in graph.connections.values():
        pygame.draw.line(
            screen, _LINK, pixels[conn.src], pixels[conn.dst], 2
        )
    half = max(1, cell // 2 - 2)
    for name, zone in graph.zones.items():
        cx, cy = pixels[name]
        pygame.draw.rect(
            screen, _zone_rgb(zone),
            pygame.Rect(cx - half, cy - half, 2 * half, 2 * half),
            border_radius=4,
        )
        pygame.draw.rect(
            screen, _BG,
            pygame.Rect(cx - half, cy - half, 2 * half, 2 * half),
            1, border_radius=4,
        )
        if cell >= 22:
            tag = f"{_drone_label(zone)} {name}"
            label = fonts[1].render(tag, True, _BG)
            screen.blit(
                label, label.get_rect(midtop=(cx, cy - half + 1))
            )
    radius = max(5, cell // 5)
    for drone_id, loc in positions.items():
        if loc is None:
            edge = transit.get(drone_id)
            if edge is None:
                continue  # in transit with no recorded link: skip
            a, b = pixels[edge[0]], pixels[edge[1]]
            cx, cy = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        else:
            cx, cy = pixels[loc]
        pygame.draw.circle(
            screen, _DRONE_RGB, (cx, cy), radius + 2
        )
        pygame.draw.circle(
            screen, _DRONE_BORDER, (cx, cy), radius + 2, 2
        )
        if cell >= 26:
            tag = f"D{drone_id}"
            label = fonts[1].render(tag, True, _DRONE_BORDER)
            screen.blit(label, label.get_rect(center=(cx, cy)))
    if turn >= last_turn:
        header = (
            f"Turn {turn}/{last_turn} - done, total turns: {makespan}"
            "  |  q: quit"
        )
    else:
        header = (
            f"Turn {turn}/{last_turn}"
            "  |  space: pause  right: step  q: quit"
        )
    screen.blit(fonts[0].render(header, True, _FG), (_MARGIN, 12))


def launch(
    graph: Graph,
    result: SimulationResult,
    drones: list[Drone],
    frame_delay_ms: int = 350,
    auto_exit: bool = False,
) -> bool:
    """Play the simulation in a window; False if pygame is unavailable.

    auto_exit closes after the final turn instead of waiting for a
    key press (used by the headless smoke test).
    """
    try:
        import pygame
    except ImportError:
        return False

    cell = _cell_size(graph)
    span_x, span_y = _spans(graph)
    win_w = min(_MAX_W, span_x * cell + 2 * _MARGIN)
    win_h = min(_MAX_H, span_y * cell + 2 * _MARGIN + _HEADER)
    pixels = _zone_pixels(graph, cell)
    positions = positions_per_turn(graph, drones, len(result.turns))
    transit = _transit_edges(drones)
    last_turn = max(1, len(result.turns))
    turn = 1
    paused = False

    pygame.init()
    try:
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Fly-in")
        fonts = (
            pygame.font.Font(None, 28),
            pygame.font.Font(None, 20),
        )
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return True
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_RIGHT and not paused:
                        turn = min(last_turn, turn + 1)
            _draw_frame(
                pygame, screen, fonts, graph, pixels, cell, turn,
                last_turn, result.makespan,
                positions[turn], transit.get(turn, {}),
            )
            pygame.display.flip()
            if auto_exit and turn >= last_turn:
                return True
            pygame.time.wait(frame_delay_ms if not paused else 80)
            if not paused and turn < last_turn:
                turn += 1
    finally:
        pygame.quit()
