"""Animated pygame visualization of the simulation (optional --gui).

Zones have a fixed size (pitch 130 points); the window sizes itself
to the map and the pitch only shrinks when the map cannot fit the
screen. Rendering goes through pygame's Window API with
allow_high_dpi, so the surface is at the display's native resolution
(no stretched, blurry upscaling). The animation starts at turn 0 -
the whole fleet parked on the start zone - and drones glide between
their per-turn positions; a drone flying a restricted move spends one
full turn travelling along its connection before arriving.

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
    # Reference palette sampled from the subject's example figures.
    "red": (232, 48, 32), "green": (48, 120, 32),
    "yellow": (248, 248, 80), "blue": (0, 0, 240),
    "magenta": (216, 64, 216), "cyan": (112, 248, 248),
    "gray": (152, 152, 152), "orange": (240, 168, 56),
    "purple": (148, 84, 204), "pink": (234, 132, 172),
    "brown": (146, 94, 54), "lime": (142, 222, 72),
    "gold": (232, 192, 62), "maroon": (132, 46, 46),
    "crimson": (192, 32, 62), "violet": (154, 94, 224),
    "rainbow": (202, 62, 222), "darkred": (102, 32, 32),
    "black": (40, 40, 46), "white": (250, 250, 252),
}

_TYPE_RGB: dict[ZoneType, tuple[int, int, int]] = {
    ZoneType.NORMAL: (176, 178, 186),
    ZoneType.PRIORITY: (0, 0, 240),
    ZoneType.RESTRICTED: (240, 168, 56),
    ZoneType.BLOCKED: (74, 76, 82),
}

# Light theme, matching the subject's example figures: near-white
# background, saturated zone circles, gray links, dark text.
_BG = (248, 248, 248)
_PANEL = (238, 238, 241)
_DOT = (224, 226, 230)
_LINK = (198, 200, 208)
_FG = (40, 42, 48)
_MUTED = (124, 128, 140)
_DRONE_RGB = (28, 28, 36)
_DRONE_DIM = (168, 170, 178)
_DRONE_BORDER = (250, 250, 252)
_DRONE_TEXT = (250, 250, 252)
_ACCENT = (0, 0, 240)

# Static sizing, in points; multiplied by the display's DPI scale.
_PITCH = 130          # grid pitch between zone centres
_PAD = 48             # breathing room on every side
_HEADER = 46          # status bar height
_FOOTER = 38          # controls bar height
_MIN_WIN = (560, 420)
_HARD_MAX = (1920, 1200)
_GOLDEN = 2.399963    # golden angle: even fan-out for co-located drones


def _zone_rgb(zone: Zone) -> tuple[int, int, int]:
    """Fill color: the zone's color tag, or a per-type fallback."""
    if zone.color is not None and zone.color.lower() in _COLOR_RGB:
        return _COLOR_RGB[zone.color.lower()]
    if zone.is_start:
        return (48, 120, 32)
    if zone.is_end:
        return (232, 48, 32)
    return _TYPE_RGB[zone.zone_type]


def _shade(rgb: tuple[int, int, int], k: int) -> tuple[int, int, int]:
    """Darken (negative k) or lighten a color."""
    return (
        max(0, min(255, rgb[0] + k)),
        max(0, min(255, rgb[1] + k)),
        max(0, min(255, rgb[2] + k)),
    )


def _lum(rgb: tuple[int, int, int]) -> int:
    """Perceived luminance, 0..255."""
    return (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) // 1000


def _edge(fill: tuple[int, int, int]) -> tuple[int, int, int]:
    """A crisp border color for a circle: darker on bright fills."""
    return _shade(fill, -45) if _lum(fill) > 140 else _shade(fill, 55)


def _text_on(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Readable text color on top of the given fill."""
    if _lum(rgb) > 140:
        return (22, 24, 30)
    return (246, 246, 250)


def _spans(graph: Graph) -> tuple[int, int]:
    """Width and height of the map in zone-coordinate units."""
    xs = [z.x for z in graph.zones.values()]
    ys = [z.y for z in graph.zones.values()]
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


@dataclass
class _Layout:
    """Geometry in physical pixels: pitch, chrome, zone centres."""

    px: tuple[int, int]
    scale: int
    pitch: int
    pad: int
    header: int
    footer: int
    pixels: dict[str, tuple[int, int]]


def _layout(graph: Graph, px: tuple[int, int], scale: int) -> _Layout:
    """Place zones on a static-pitch grid, centered in the surface.

    The pitch is _PITCH points unless the map cannot fit, in which
    case it shrinks just enough. It never grows: small maps get a
    small window, not giant zones.
    """
    span_x, span_y = _spans(graph)
    pad, header, footer = _PAD * scale, _HEADER * scale, _FOOTER * scale
    pitch = _PITCH * scale
    fit = min(
        max(1, (px[0] - 2 * pad) // max(1, span_x)),
        max(1, (px[1] - header - footer - 2 * pad) // max(1, span_y)),
    )
    pitch = max(10, min(pitch, fit))
    xs = [z.x for z in graph.zones.values()]
    ys = [z.y for z in graph.zones.values()]
    min_x, max_y = min(xs), max(ys)
    x0 = (px[0] - span_x * pitch) // 2
    y0 = header + (px[1] - header - footer - span_y * pitch) // 2
    return _Layout(
        px=px,
        scale=scale,
        pitch=pitch,
        pad=pad,
        header=header,
        footer=footer,
        pixels={
            name: (
                x0 + (z.x - min_x) * pitch + pitch // 2,
                y0 + (max_y - z.y) * pitch + pitch // 2,
            )
            for name, z in graph.zones.items()
        },
    )


def _zone_radius(pitch: int) -> int:
    """Circle radius: deliberately smaller than the pitch."""
    return max(6, int(pitch * 0.44))


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
    raw: dict[int, tuple[int, int]], pitch: int
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
        spread = min(max(10, pitch // 2), 6 + 4 * len(ids))
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
        {d.drone_id: start_pt for d in drones}, layout.pitch
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
        frames.append(_spread(raw, layout.pitch))
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
    scale = layout.scale
    span_x, span_y = _spans(graph)
    pitch = layout.pitch
    radius = _zone_radius(pitch)
    if span_x * span_y <= 4000:
        x0 = (layout.px[0] - span_x * pitch) // 2
        y0 = layout.header + (
            layout.px[1] - layout.header - layout.footer - span_y * pitch
        ) // 2
        for gx in range(span_x + 1):
            for gy in range(span_y + 1):
                pygame.draw.circle(
                    screen, _DOT, (x0 + gx * pitch, y0 + gy * pitch),
                    scale,
                )
    for conn in graph.connections.values():
        pygame.draw.line(
            screen, _LINK, layout.pixels[conn.src],
            layout.pixels[conn.dst], max(2 * scale, pitch // 40),
        )
    for name, zone in graph.zones.items():
        cx, cy = layout.pixels[name]
        fill = _zone_rgb(zone)
        # Start and end hubs get visibly larger circles.
        zone_r = radius
        if zone.is_start or zone.is_end:
            zone_r = min(pitch // 2 - 4 * scale, int(radius * 1.3))
        pygame.draw.circle(screen, fill, (cx, cy), zone_r)
        edge = _edge(fill)
        if zone.zone_type is ZoneType.BLOCKED:
            pygame.draw.circle(screen, edge, (cx, cy), zone_r, 2 * scale)
            d = int(zone_r * 0.5)
            pygame.draw.line(
                screen, _shade(fill, 120), (cx - d, cy - d),
                (cx + d, cy + d), 3 * scale,
            )
            pygame.draw.line(
                screen, _shade(fill, 120), (cx - d, cy + d),
                (cx + d, cy - d), 3 * scale,
            )
            if pitch >= 40 * scale:
                label = fonts[1].render(name, True, _MUTED)
                screen.blit(
                    label,
                    label.get_rect(midtop=(cx, cy + zone_r + 3 * scale)),
                )
            continue
        pygame.draw.circle(screen, edge, (cx, cy), zone_r, 2 * scale)
        if zone.is_start or zone.is_end:
            pygame.draw.circle(
                screen, edge, (cx, cy), zone_r + 4 * scale, 2 * scale
            )
        if pitch >= 40 * scale:
            # Name inside the circle, shrinking to fit; below if wide.
            color = _text_on(fill)
            label = fonts[1].render(name, True, color)
            if label.get_width() > zone_r * 1.9:
                label = fonts[2].render(name, True, color)
            if label.get_width() > zone_r * 1.9:
                screen.blit(
                    label,
                    label.get_rect(midtop=(cx, cy + zone_r + 3 * scale)),
                )
            else:
                screen.blit(label, label.get_rect(center=(cx, cy)))
    drone_r = max(6 * scale, min(14 * scale, pitch // 9))
    for did, point in frame_a.items():
        target = frame_b.get(did, point)
        x, y = _lerp(point, target, frac)
        dim = deliveries.get(did, 0) <= turn
        color = _DRONE_DIM if dim else _DRONE_RGB
        pygame.draw.circle(screen, color, (x, y), drone_r)
        pygame.draw.circle(screen, _DRONE_BORDER, (x, y), drone_r, 2)
        if pitch >= 52 * scale:
            tag_color = _FG if dim else _DRONE_TEXT
            tag = fonts[2].render(f"D{did}", True, tag_color)
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
    win_w, win_h = layout.px
    scale = layout.scale
    pygame.draw.rect(screen, _PANEL, (0, 0, win_w, layout.header))
    pygame.draw.rect(
        screen, _PANEL, (0, win_h - layout.footer, win_w, layout.footer)
    )
    if turn >= last_turn:
        status = f"DONE  total turns: {makespan}"
    else:
        status = "PAUSED" if paused else "PLAYING"
    title = fonts[0].render(
        f"Fly-in   Turn {turn}/{last_turn}   {status}", True, _FG
    )
    screen.blit(
        title, (layout.pad, (layout.header - title.get_height()) // 2)
    )
    hint = fonts[2].render(
        "space play/pause   left/right step   r restart   q quit",
        True, _MUTED,
    )
    screen.blit(
        hint,
        (layout.pad,
         win_h - layout.footer + (layout.footer - hint.get_height()) // 2),
    )
    legend_items = [
        ("S", (48, 120, 32)), ("E", (232, 48, 32)),
        ("P", _TYPE_RGB[ZoneType.PRIORITY]),
        ("R", _TYPE_RGB[ZoneType.RESTRICTED]),
        ("#", _MUTED), ("O", _TYPE_RGB[ZoneType.NORMAL]),
    ]
    lx = win_w - layout.pad
    mid_y = win_h - layout.footer // 2
    for letter, color in reversed(legend_items):
        glyph = fonts[2].render(letter, True, _FG)
        screen.blit(glyph, glyph.get_rect(midright=(lx, mid_y)))
        lx -= glyph.get_width() + 6 * scale
        pygame.draw.circle(screen, color, (lx - 4 * scale, mid_y), 5 * scale)
        lx -= 16 * scale
    bar_x0, bar_x1 = layout.pad, win_w - layout.pad
    bar_y = win_h - 9 * scale
    pygame.draw.line(
        screen, _DOT, (bar_x0, bar_y), (bar_x1, bar_y), 3 * scale
    )
    frac = turn / last_turn if last_turn else 1.0
    head = bar_x0 + int((bar_x1 - bar_x0) * frac)
    pygame.draw.line(
        screen, _ACCENT, (bar_x0, bar_y), (head, bar_y), 3 * scale
    )


def _make_fonts(pygame: Any, scale: int) -> tuple[Any, Any, Any]:
    """Header, zone-name and tag fonts at the display's resolution."""
    return (
        pygame.font.Font(None, 30 * scale),
        pygame.font.Font(None, 24 * scale),
        pygame.font.Font(None, 17 * scale),
    )


@dataclass
class _Target:
    """The drawing surface, plus how to present it.

    ``window`` is None on the classic set_mode fallback path.
    """

    window: Any | None
    screen: Any

    @property
    def pts(self) -> tuple[int, int]:
        """Window size in points."""
        if self.window is not None:
            w, h = self.window.size
        else:
            w, h = self.screen.get_size()
        return (int(w), int(h))

    @property
    def px(self) -> tuple[int, int]:
        """Surface size in physical pixels."""
        w, h = self.screen.get_size()
        return (int(w), int(h))

    @property
    def scale(self) -> int:
        """Physical pixels per point (2 on Retina, 1 elsewhere)."""
        if self.window is not None:
            return max(1, self.px[0] // max(1, self.pts[0]))
        return 1

    def flip(self) -> None:
        """Present the frame."""
        if self.window is not None:
            self.window.flip()
        else:
            import pygame  # noqa: PLC0415 - lazy like everywhere here
            pygame.display.flip()


def _make_target(
    pygame: Any, title: str, pts: tuple[int, int]
) -> _Target:
    """Open a window, preferring native-resolution rendering."""
    for kwargs in (
        {"size": pts, "resizable": True, "allow_high_dpi": True},
        {"size": pts, "resizable": True},
    ):
        try:
            window = pygame.Window(title, **kwargs)
            return _Target(window=window, screen=window.get_surface())
        except (TypeError, pygame.error):
            continue
    screen = pygame.display.set_mode(pts, pygame.RESIZABLE)
    return _Target(window=None, screen=screen)


def _desired_pts(pygame: Any, graph: Graph) -> tuple[int, int]:
    """Window size in points: the map at static pitch, screen-clamped."""
    span_x, span_y = _spans(graph)
    try:
        info = pygame.display.Info()
        dw, dh = int(info.current_w), int(info.current_h)
    except pygame.error:
        dw, dh = _HARD_MAX
    content_w = span_x * _PITCH + 2 * _PAD
    content_h = span_y * _PITCH + _HEADER + _FOOTER + 2 * _PAD
    w = max(
        _MIN_WIN[0],
        min(content_w, int(dw * 0.9), _HARD_MAX[0]),
    )
    h = max(
        _MIN_WIN[1],
        min(content_h, int(dh * 0.9), _HARD_MAX[1]),
    )
    return (w, h)


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
    title = f"Fly-in - {len(drones)} drones"

    pygame.init()
    try:
        target = _make_target(pygame, title, _desired_pts(pygame, graph))
        layout = _layout(graph, target.px, target.scale)
        fonts = _make_fonts(pygame, layout.scale)
        frames = _keyframes(graph, drones, len(result.turns), layout)
        turn = 0
        paused = False
        anim_start = pygame.time.get_ticks()

        def rebuild() -> None:
            """Relayout after a window resize (pitch stays static)."""
            nonlocal layout, fonts, frames, target
            if target.window is not None:
                target = _Target(
                    window=target.window,
                    screen=target.window.get_surface(),
                )
            layout = _layout(graph, target.px, target.scale)
            fonts = _make_fonts(pygame, layout.scale)
            frames = _keyframes(graph, drones, len(result.turns), layout)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                if event.type == pygame.VIDEORESIZE:
                    if target.window is None:
                        target = _Target(
                            window=None,
                            screen=pygame.display.set_mode(
                                (event.w, event.h), pygame.RESIZABLE
                            ),
                        )
                    rebuild()
                if event.type == pygame.WINDOWSIZECHANGED:
                    rebuild()
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
                pygame, target.screen, fonts, graph, layout,
                frames[turn],
                frames[min(turn + 1, last_turn)],
                frac, deliveries, turn,
            )
            _draw_chrome(
                pygame, target.screen, fonts, layout, turn, last_turn,
                result.makespan, paused,
            )
            target.flip()
            if auto_exit and turn >= last_turn:
                pygame.time.wait(250)
                return True
            pygame.time.wait(15)
    finally:
        pygame.quit()
