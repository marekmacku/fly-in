"""Animated pygame visualization of the simulation (optional --gui).

The window scales itself to the map (and can be resized live with the
mouse) and renders through pygame's SCALED mode so it stays crisp on
high-DPI displays. The animation starts at turn 0 - the whole fleet
parked on the start zone - and drones glide between their per-turn
positions; a drone flying a restricted move spends one full turn
travelling along its connection before arriving.

Controls: space play/pause, left/right arrow step one turn,
r restart, q or escape quit.

pygame is imported lazily inside launch(), so the program runs fine
without it; the CLI falls back to the colored terminal renderer.
Only existing zones are drawn, so absurd coordinate spans degrade
gracefully instead of hanging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from fly_in2.model import Drone, Graph, SimulationResult, Zone, ZoneType
from fly_in2.visualizer import positions_per_turn

_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (204, 66, 66), "green": (74, 176, 94),
    "yellow": (222, 194, 62), "blue": (76, 116, 222),
    "magenta": (192, 74, 192), "cyan": (74, 192, 192),
    "gray": (122, 124, 130), "orange": (236, 142, 52),
    "purple": (144, 82, 202), "pink": (232, 132, 172),
    "brown": (142, 92, 52), "lime": (142, 222, 72),
    "gold": (232, 192, 62), "maroon": (132, 46, 46),
    "crimson": (192, 32, 62), "violet": (152, 92, 222),
    "rainbow": (202, 62, 222), "darkred": (102, 32, 32),
    "black": (37, 37, 42), "white": (236, 236, 240),
}

_TYPE_RGB: dict[ZoneType, tuple[int, int, int]] = {
    ZoneType.NORMAL: (92, 102, 124),
    ZoneType.PRIORITY: (66, 128, 226),
    ZoneType.RESTRICTED: (198, 112, 52),
    ZoneType.BLOCKED: (44, 46, 52),
}

_BG = (16, 17, 24)
_PANEL = (23, 24, 33)
_DOT = (30, 32, 44)
_LINK = (66, 68, 84)
_FG = (236, 236, 242)
_MUTED = (136, 140, 156)
_DRONE_RGB = (250, 216, 92)
_DRONE_DIM = (122, 124, 134)
_DRONE_BORDER = (28, 28, 34)
_ACCENT = (94, 158, 236)

_HEADER = 46
_FOOTER = 38
_PAD = 48          # breathing room on every side
_MAX_CELL = 110
_MIN_WIN = (560, 420)
_HARD_MAX = (1920, 1200)
_GOLDEN = 2.399963  # golden angle: even fan-out for co-located drones


def _zone_rgb(zone: Zone) -> tuple[int, int, int]:
    """Fill color: the zone's color tag, or a per-type fallback."""
    if zone.color is not None and zone.color.lower() in _COLOR_RGB:
        return _COLOR_RGB[zone.color.lower()]
    if zone.is_start:
        return (84, 182, 96)
    if zone.is_end:
        return (228, 92, 92)
    return _TYPE_RGB[zone.zone_type]


def _lighten(rgb: tuple[int, int, int], k: int) -> tuple[int, int, int]:
    """A lighter version of a color, for circle borders."""
    return (
        min(255, rgb[0] + k),
        min(255, rgb[1] + k),
        min(255, rgb[2] + k),
    )


def _spans(graph: Graph) -> tuple[int, int]:
    """Width and height of the map in zone-coordinate units."""
    xs = [z.x for z in graph.zones.values()]
    ys = [z.y for z in graph.zones.values()]
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


@dataclass
class _Layout:
    """Everything geometry: window, cell size, zone pixel centres."""

    win: tuple[int, int]
    cell: int
    pixels: dict[str, tuple[int, int]]


def _layout(
    pygame: Any, graph: Graph, forced: tuple[int, int] | None = None
) -> _Layout:
    """Fit the map into the screen (or the given window size).

    The cell is the grid pitch; zone circles are drawn smaller than
    the cell so connections between neighbours stay visible.
    """
    span_x, span_y = _spans(graph)
    if forced is not None:
        budget_w, budget_h = forced
    else:
        info = pygame.display.Info()
        budget_w = min(1700, max(_MIN_WIN[0], int(info.current_w * 0.85)))
        budget_h = min(1100, max(_MIN_WIN[1], int(info.current_h * 0.85)))
    budget_w = min(budget_w, _HARD_MAX[0])
    budget_h = min(budget_h, _HARD_MAX[1])
    content_w = max(1, budget_w - 2 * _PAD)
    content_h = max(1, budget_h - _HEADER - _FOOTER - 2 * _PAD)
    cell = min(
        _MAX_CELL,
        content_w // max(1, span_x),
        content_h // max(1, span_y),
    )
    cell = max(8, cell)
    win_w = max(
        _MIN_WIN[0], min(budget_w, span_x * cell + 2 * _PAD),
    )
    win_h = max(
        _MIN_WIN[1],
        min(budget_h, span_y * cell + _HEADER + _FOOTER + 2 * _PAD),
    )
    xs = [z.x for z in graph.zones.values()]
    ys = [z.y for z in graph.zones.values()]
    min_x, max_y = min(xs), max(ys)
    x0 = (win_w - span_x * cell) // 2
    y0 = _HEADER + (win_h - _HEADER - _FOOTER - span_y * cell) // 2
    return _Layout(
        win=(win_w, win_h),
        cell=cell,
        pixels={
            name: (
                x0 + (z.x - min_x) * cell + cell // 2,
                y0 + (max_y - z.y) * cell + cell // 2,
            )
            for name, z in graph.zones.items()
        },
    )


def _zone_radius(cell: int) -> int:
    """Circle radius: deliberately smaller than the cell pitch."""
    return max(6, cell // 2 - max(7, cell // 6))


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


def _delivery_turns(drones: list[Drone]) -> dict[int, int]:
    """Turn on which each drone reached the end zone."""
    delivered: dict[int, int] = {}
    for drone in drones:
        for rec in reversed(drone.trajectory):
            if rec.token is not None:
                delivered[drone.drone_id] = rec.turn
                break
    return delivered


def _spread(
    raw: dict[int, tuple[int, int]], cell: int
) -> dict[int, tuple[int, int]]:
    """Fan out drones that share one point on a golden-angle spiral."""
    groups: dict[tuple[int, int], list[int]] = {}
    for did, point in raw.items():
        groups.setdefault(point, []).append(did)
    frame: dict[int, tuple[int, int]] = {}
    for point, ids in groups.items():
        if len(ids) == 1:
            frame[ids[0]] = point
            continue
        spread = min(max(10, cell // 2), 6 + 4 * len(ids))
        for k, did in enumerate(sorted(ids)):
            dist = 5 + (k * spread) // max(1, len(ids))
            frame[did] = (
                point[0] + int(dist * math.cos(k * _GOLDEN)),
                point[1] + int(dist * math.sin(k * _GOLDEN)),
            )
    return frame


def _keyframes(
    graph: Graph,
    drones: list[Drone],
    n_turns: int,
    layout: _Layout,
) -> list[dict[int, tuple[int, int]]]:
    """Drone pixel positions per turn, starting at turn 0.

    Frame 0 is the whole fleet on the start zone. A drone in restricted
    transit sits midway along its connection that turn, so it crosses
    the link over two animation frames.
    """
    positions = positions_per_turn(graph, drones, n_turns)
    transit = _transit_edges(drones)
    assert graph.start is not None
    start_pt = layout.pixels[graph.start.name]
    frames = [_spread(
        {d.drone_id: start_pt for d in drones}, layout.cell
    )]
    for turn in range(1, n_turns + 1):
        raw: dict[int, tuple[int, int]] = {}
        for did, loc in positions[turn].items():
            if loc is None:
                edge = transit.get(turn, {}).get(did)
                if edge is None:
                    continue
                a, b = layout.pixels[edge[0]], layout.pixels[edge[1]]
                raw[did] = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            else:
                raw[did] = layout.pixels[loc]
        frames.append(_spread(raw, layout.cell))
    return frames


def _lerp(
    a: tuple[int, int], b: tuple[int, int], frac: float
) -> tuple[int, int]:
    """Point between a and b at fraction frac."""
    return (
        int(a[0] + (b[0] - a[0]) * frac),
        int(a[1] + (b[1] - a[1]) * frac),
    )


def _draw_world(
    pygame: Any,
    screen: Any,
    fonts: tuple[Any, Any, Any],
    graph: Graph,
    layout: _Layout,
    frame_a: dict[int, tuple[int, int]],
    frame_b: dict[int, tuple[int, int]],
    frac: float,
    deliveries: dict[int, int],
    turn: int,
) -> None:
    """Render one frame: lattice, links, zone circles, gliding drones."""
    screen.fill(_BG)
    cell = layout.cell
    span_x, span_y = _spans(graph)
    if span_x * span_y <= 4000:
        x0 = (layout.win[0] - span_x * cell) // 2
        y0 = _HEADER + (
            layout.win[1] - _HEADER - _FOOTER - span_y * cell
        ) // 2
        for gx in range(span_x + 1):
            for gy in range(span_y + 1):
                pygame.draw.circle(
                    screen, _DOT, (x0 + gx * cell, y0 + gy * cell), 1,
                )
    for conn in graph.connections.values():
        pygame.draw.line(
            screen, _LINK, layout.pixels[conn.src],
            layout.pixels[conn.dst], max(2, cell // 30),
        )
    radius = _zone_radius(cell)
    for name, zone in graph.zones.items():
        cx, cy = layout.pixels[name]
        fill = _zone_rgb(zone)
        pygame.draw.circle(screen, fill, (cx, cy), radius)
        if zone.zone_type is ZoneType.BLOCKED:
            pygame.draw.circle(
                screen, _lighten(fill, 30), (cx, cy), radius, 2
            )
            d = int(radius * 0.5)
            pygame.draw.line(
                screen, _MUTED, (cx - d, cy - d), (cx + d, cy + d), 2
            )
            pygame.draw.line(
                screen, _MUTED, (cx - d, cy + d), (cx + d, cy - d), 2
            )
        else:
            pygame.draw.circle(
                screen, _lighten(fill, 45), (cx, cy), radius, 2
            )
            if zone.is_start or zone.is_end:
                pygame.draw.circle(
                    screen, _FG, (cx, cy), radius + 4, 2
                )
        if cell >= 30:
            color = _MUTED if zone.zone_type is ZoneType.BLOCKED else _FG
            label = fonts[1].render(name, True, color)
            screen.blit(
                label,
                label.get_rect(midtop=(cx, cy + radius + 3)),
            )
    drone_r = max(6, min(13, cell // 7))
    for did, point in frame_a.items():
        target = frame_b.get(did, point)
        x, y = _lerp(point, target, frac)
        dim = deliveries.get(did, 0) <= turn
        color = _DRONE_DIM if dim else _DRONE_RGB
        pygame.draw.circle(screen, color, (x, y), drone_r)
        pygame.draw.circle(screen, _DRONE_BORDER, (x, y), drone_r, 2)
        if cell >= 44:
            tag = fonts[2].render(f"D{did}", True, _DRONE_BORDER)
            screen.blit(tag, tag.get_rect(center=(x, y)))


def _draw_chrome(
    pygame: Any,
    screen: Any,
    fonts: tuple[Any, Any, Any],
    layout: _Layout,
    turn: int,
    last_turn: int,
    makespan: int,
    paused: bool,
) -> None:
    """Header with status, footer with controls, legend, progress bar."""
    win_w, win_h = layout.win
    pygame.draw.rect(screen, _PANEL, (0, 0, win_w, _HEADER))
    pygame.draw.rect(screen, _PANEL, (0, win_h - _FOOTER, win_w, _FOOTER))
    if turn >= last_turn:
        status = f"DONE  total turns: {makespan}"
    elif turn == 0:
        status = "PAUSED" if paused else "PLAYING"
    else:
        status = "PAUSED" if paused else "PLAYING"
    title = fonts[0].render(
        f"Fly-in   Turn {turn}/{last_turn}   {status}", True, _FG
    )
    screen.blit(title, (_PAD, (_HEADER - title.get_height()) // 2))
    hint = fonts[2].render(
        "space play/pause   left/right step   r restart   q quit",
        True, _MUTED,
    )
    screen.blit(
        hint,
        (_PAD, win_h - _FOOTER + (_FOOTER - hint.get_height()) // 2),
    )
    legend_items = [
        ("S", (84, 182, 96)), ("E", (228, 92, 92)),
        ("P", _TYPE_RGB[ZoneType.PRIORITY]),
        ("R", _TYPE_RGB[ZoneType.RESTRICTED]),
        ("#", _MUTED), ("O", _TYPE_RGB[ZoneType.NORMAL]),
    ]
    lx = win_w - _PAD
    for letter, color in reversed(legend_items):
        glyph = fonts[2].render(letter, True, _FG)
        screen.blit(glyph, glyph.get_rect(midright=(lx, win_h - 19)))
        lx -= glyph.get_width() + 6
        pygame.draw.circle(
            screen, color, (lx - 4, win_h - _FOOTER // 2), 5
        )
        lx -= 16
    bar_x0, bar_x1 = _PAD, win_w - _PAD
    bar_y = win_h - 9
    pygame.draw.line(screen, _DOT, (bar_x0, bar_y), (bar_x1, bar_y), 3)
    frac = turn / last_turn if last_turn else 1.0
    head = bar_x0 + int((bar_x1 - bar_x0) * frac)
    pygame.draw.line(screen, _ACCENT, (bar_x0, bar_y), (head, bar_y), 3)


def _make_fonts(pygame: Any, cell: int) -> tuple[Any, Any, Any]:
    """Header, zone-name and tag fonts sized for the current cell."""
    return (
        pygame.font.Font(None, 30),
        pygame.font.Font(None, max(18, min(28, cell // 3))),
        pygame.font.Font(None, max(16, min(24, cell // 4))),
    )


def _open_window(pygame: Any, win: tuple[int, int]) -> Any:
    """Open the window, preferring the crisp SCALED mode."""
    try:
        return pygame.display.set_mode(
            win, pygame.SCALED | pygame.RESIZABLE
        )
    except pygame.error:
        return pygame.display.set_mode(win, pygame.RESIZABLE)


def launch(
    graph: Graph,
    result: SimulationResult,
    drones: list[Drone],
    frame_delay_ms: int = 500,
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

    last_turn = max(1, len(result.turns))
    deliveries = _delivery_turns(drones)

    pygame.init()
    try:
        layout = _layout(pygame, graph)
        screen = _open_window(pygame, layout.win)
        pygame.display.set_caption(f"Fly-in - {len(drones)} drones")
        fonts = _make_fonts(pygame, layout.cell)
        frames = _keyframes(graph, drones, len(result.turns), layout)
        turn = 0
        paused = False
        anim_start = pygame.time.get_ticks()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                if event.type == pygame.VIDEORESIZE:
                    layout = _layout(pygame, graph, (event.w, event.h))
                    screen = _open_window(pygame, layout.win)
                    frames = _keyframes(
                        graph, drones, len(result.turns), layout
                    )
                    fonts = _make_fonts(pygame, layout.cell)
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return True
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        anim_start = pygame.time.get_ticks()
                    if event.key == pygame.K_RIGHT:
                        paused = True
                        turn = min(last_turn, turn + 1)
                    if event.key == pygame.K_LEFT:
                        paused = True
                        turn = max(0, turn - 1)
                    if event.key == pygame.K_r:
                        turn = 0
                        paused = False
                        anim_start = pygame.time.get_ticks()
            now = pygame.time.get_ticks()
            frac = (
                1.0
                if paused or turn >= last_turn
                else min(1.0, (now - anim_start) / frame_delay_ms)
            )
            if not paused and turn < last_turn and frac >= 1.0:
                turn += 1
                anim_start = now
                frac = 0.0
            _draw_world(
                pygame, screen, fonts, graph, layout,
                frames[turn],
                frames[min(turn + 1, last_turn)],
                frac, deliveries, turn,
            )
            _draw_chrome(
                pygame, screen, fonts, layout, turn, last_turn,
                result.makespan, paused,
            )
            pygame.display.flip()
            if auto_exit and turn >= last_turn:
                pygame.time.wait(250)
                return True
            pygame.time.wait(15)
    finally:
        pygame.quit()
