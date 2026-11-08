"""Animated pygame visualization of the simulation (optional --gui).

Zones have a fixed base size (pitch 60 points) on every map, small
or large; every map opens in the same default-size window, which the
user can resize (the size persists across map switches in a tour),
and the pitch only shrinks when a map cannot fit the window.
Rendering goes through pygame's Window API with
allow_high_dpi, so the surface is at the display's native resolution
(no stretched, blurry upscaling). The animation starts at turn 0 -
the whole fleet parked on the start zone - and drones glide between
their per-turn positions; a drone flying a restricted move spends one
full turn travelling along its connection before arriving. The
footer shows the simulation's own output line for the current turn,
and zones with capacity above one carry a badge. Label sizes derive
from the pitch, so text always fits instead of disappearing.

Controls: space play/pause, left/right arrow step one turn, n/p next
or previous map (tour mode), r restart, q or escape quit.

pygame is imported lazily inside launch(), so the program runs fine
without it; the CLI falls back to the colored terminal renderer.
Only existing zones are drawn, so absurd coordinate spans degrade
gracefully instead of hanging.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fly_in2.errors import FlyInError
from fly_in2.model import Drone, Graph, SimulationResult, Zone, ZoneType
from fly_in2.parser import parse_map_file
from fly_in2.planner import plan
from fly_in2.simulator import Simulator
from fly_in2.validator import validate
from fly_in2.visualizer import positions_per_turn

SessionEnd = Literal["quit", "next", "prev"]

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

# Dark theme: black background, saturated zone circles, light text.
_BG = (10, 10, 14)
_PANEL = (18, 19, 26)
_DOT = (28, 30, 40)
_LINK = (96, 98, 112)
_FG = (236, 236, 242)
_MUTED = (150, 154, 168)
_DRONE_RGB = (246, 246, 250)
_DRONE_DIM = (112, 114, 124)
_DRONE_BORDER = (20, 20, 26)
_DRONE_TEXT = (28, 28, 36)
_ACCENT = (94, 158, 236)

# Static sizing, in points; multiplied by the display's DPI scale.
_PITCH = 60           # base grid pitch everywhere (challenger-like)
_DEFAULT_WIN = (1560, 980)   # one window size for every map
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


def _edge(fill: tuple[int, int, int]) -> tuple[int, int, int]:
    """A lighter border color: reads well on the black background."""
    return _shade(fill, 55)


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


def _origin(
    layout: _Layout, span_x: int, span_y: int
) -> tuple[int, int]:
    """Top-left pixel of the zone grid within the surface."""
    x0 = (layout.px[0] - span_x * layout.pitch) // 2
    y0 = layout.header + (
        layout.px[1] - layout.header - layout.footer - span_y * layout.pitch
    ) // 2
    return x0, y0


def _mix(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    """Linear blend between two colors."""
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def _draw_zone(
    pygame: Any,
    layer: Any,
    fonts: tuple[Any, Any, Any, Any],
    zone: Zone,
    cx: int,
    cy: int,
    zone_r: int,
    pitch: int,
    scale: int,
) -> None:
    """One zone: shaded ball, type decoration, name, capacity badge."""
    fill = _zone_rgb(zone)
    # Start and end hubs glow: faint rings stepping outward.
    if zone.is_start or zone.is_end:
        pygame.draw.circle(
            layer, _shade(fill, 45), (cx, cy),
            zone_r + 6 * scale, 1 * scale,
        )
        pygame.draw.circle(
            layer, _shade(fill, 25), (cx, cy),
            zone_r + 10 * scale, 1 * scale,
        )
    # Shaded ball: dark rim to light centre reads as a sphere.
    rim = _shade(fill, -35)
    core = _shade(fill, 30)
    for i in range(6):
        t = i / 5
        rad = max(1, int(zone_r * (1.0 - 0.45 * t)))
        pygame.draw.circle(layer, _mix(rim, core, t), (cx, cy), rad)
    # Type decorations.
    if zone.zone_type is ZoneType.RESTRICTED:
        # Hazard stripes: 45-degree chords clipped to the circle.
        u = math.sqrt(0.5)
        step = max(5 * scale, zone_r // 4)
        for off in range(-zone_r + step, zone_r, step):
            half = math.sqrt(max(0.0, zone_r * zone_r - off * off))
            ax = int(cx - u * off - u * half)
            ay = int(cy + u * off - u * half)
            bx = int(cx - u * off + u * half)
            by = int(cy + u * off + u * half)
            pygame.draw.line(
                layer, _shade(fill, -55), (ax, ay), (bx, by), 2 * scale
            )
    elif zone.zone_type is ZoneType.PRIORITY:
        # Bright inner ring marks the preferred lane.
        pygame.draw.circle(
            layer, _shade(fill, 80), (cx, cy),
            max(2, int(zone_r * 0.55)), max(2 * scale, zone_r // 12),
        )
    if zone.zone_type is ZoneType.BLOCKED:
        d = int(zone_r * 0.5)
        pygame.draw.line(
            layer, _shade(fill, 130), (cx - d, cy - d),
            (cx + d, cy + d), 3 * scale,
        )
        pygame.draw.line(
            layer, _shade(fill, 130), (cx - d, cy + d),
            (cx + d, cy - d), 3 * scale,
        )
    if zone.is_start:
        # Bullseye: launch point.
        pygame.draw.circle(
            layer, _shade(fill, 100), (cx, cy),
            max(2 * scale, zone_r // 5),
        )
        pygame.draw.circle(
            layer, _shade(fill, 100), (cx, cy),
            max(3 * scale, zone_r // 2), 2 * scale,
        )
    elif zone.is_end and zone.zone_type is not ZoneType.BLOCKED:
        # Target rings: the goal.
        pygame.draw.circle(
            layer, _shade(fill, 100), (cx, cy), int(zone_r * 0.6),
            2 * scale,
        )
        pygame.draw.circle(
            layer, _shade(fill, 100), (cx, cy), int(zone_r * 0.3),
            2 * scale,
        )
    pygame.draw.circle(layer, _edge(fill), (cx, cy), zone_r, 2 * scale)
    # Name below the circle: name font, then smaller font with the
    # full name, then truncation - never inside, where drones live.
    color = _MUTED if zone.zone_type is ZoneType.BLOCKED else _FG
    limit = pitch * 0.95
    shown = zone.name
    label = fonts[2].render(shown, True, color)
    if label.get_width() > limit:
        label = fonts[3].render(shown, True, color)
        while label.get_width() > limit and len(shown) > 4:
            shown = shown[:-3] + ".."
            label = fonts[3].render(shown, True, color)
    layer.blit(
        label, label.get_rect(midtop=(cx, cy + zone_r + 4 * scale))
    )
    # Capacity badge for zones that hold more than one drone.
    if zone.capacity is not None and zone.capacity > 1:
        bx = cx + int(zone_r * 0.78)
        by = cy - int(zone_r * 0.78)
        br = max(9 * scale, pitch // 13)
        pygame.draw.circle(layer, _PANEL, (bx, by), br)
        pygame.draw.circle(layer, _MUTED, (bx, by), br, 1)
        badge = fonts[3].render(f"x{zone.capacity}", True, _FG)
        layer.blit(badge, badge.get_rect(center=(bx, by)))


def _make_map_layer(
    pygame: Any,
    graph: Graph,
    layout: _Layout,
    fonts: tuple[Any, Any, Any, Any],
) -> Any:
    """Static map art: lattice, links, decorated zones, names, badges.

    Everything here is constant per layout, so it renders once and is
    blitted every frame; per-frame work is only the drones. The dots
    cover the entire window, aligned with the zone centres.
    """
    layer = pygame.Surface(layout.px)
    layer.fill(_BG)
    span_x, span_y = _spans(graph)
    x0, y0 = _origin(layout, span_x, span_y)
    step = max(14 * layout.scale, layout.pitch // 5)
    radius = max(1, layout.scale)
    for kx in range(-(x0 // step), (layout.px[0] - x0) // step + 1):
        x = x0 + kx * step
        if not 0 <= x < layout.px[0]:
            continue
        for ky in range(-(y0 // step), (layout.px[1] - y0) // step + 1):
            y = y0 + ky * step
            if 0 <= y < layout.px[1]:
                pygame.draw.circle(layer, _DOT, (x, y), radius)
    for conn in graph.connections.values():
        pygame.draw.line(
            layer, _LINK, layout.pixels[conn.src],
            layout.pixels[conn.dst],
            max(2 * layout.scale, layout.pitch // 40),
        )
    base_r = _zone_radius(layout.pitch)
    for name, zone in graph.zones.items():
        cx, cy = layout.pixels[name]
        zone_r = base_r
        if zone.is_start or zone.is_end:
            zone_r = min(
                layout.pitch // 2 - 4 * layout.scale, int(base_r * 1.15)
            )
        _draw_zone(
            pygame, layer, fonts, zone, cx, cy, zone_r,
            layout.pitch, layout.scale,
        )
    return layer


def _zone_radius(pitch: int) -> int:
    """Circle radius: well below the pitch so links stay long."""
    return max(6, int(pitch * 0.30))


def _drone_radius(pitch: int, scale: int) -> int:
    """Drone marker radius, proportional to the pitch."""
    return max(7 * scale, min(15 * scale, pitch // 9))


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
    layer: Any,
    fonts: tuple[Any, Any, Any, Any],
    layout: _Layout,
    frame_a: dict[int, tuple[int, int]],
    frame_b: dict[int, tuple[int, int]],
    frac: float,
    deliveries: dict[int, int],
    turn: int,
) -> None:
    """Render one frame: blit the map layer, then the gliding drones."""
    screen.blit(layer, (0, 0))
    scale = layout.scale
    pitch = layout.pitch
    drone_r = _drone_radius(pitch, scale)
    for did, point in frame_a.items():
        target = frame_b.get(did, point)
        x, y = _lerp(point, target, frac)
        dim = deliveries.get(did, 0) <= turn
        color = _DRONE_DIM if dim else _DRONE_RGB
        pygame.draw.circle(screen, color, (x, y), drone_r)
        pygame.draw.circle(screen, _DRONE_BORDER, (x, y), drone_r, 2)
        tag = fonts[3].render(f"D{did}", True, _DRONE_TEXT)
        if tag.get_width() <= drone_r * 1.9:
            screen.blit(tag, tag.get_rect(center=(x, y)))
        else:  # too wide for the marker: hover it just above
            screen.blit(
                tag,
                tag.get_rect(midbottom=(x, y - drone_r - 2 * scale)),
            )


def _draw_chrome(
    pygame: Any,
    screen: Any,
    fonts: tuple[Any, Any, Any, Any],
    layout: _Layout,
    turn: int,
    last_turn: int,
    makespan: int,
    paused: bool,
    line: str,
    map_label: str | None,
) -> None:
    """Header (status, map name), footer (output line, legend), bar."""
    win_w, win_h = layout.px
    scale = layout.scale
    pygame.draw.rect(screen, _PANEL, (0, 0, win_w, layout.header))
    pygame.draw.rect(
        screen, _PANEL, (0, win_h - layout.footer, win_w, layout.footer)
    )
    map_w = 0
    if map_label:
        glyph = fonts[1].render(map_label, True, _MUTED)
        map_w = glyph.get_width()
        screen.blit(
            glyph,
            glyph.get_rect(midright=(win_w - layout.pad, layout.header // 2)),
        )
    if turn >= last_turn:
        status = f"DONE  total turns: {makespan}"
    else:
        status = "PAUSED" if paused else "PLAYING"
    title = fonts[0].render(
        f"Fly-in   Turn {turn}/{last_turn}   {status}"
        "   -   space pause   arrows step   n/p map   r restart   q quit",
        True, _FG,
    )
    title_w = max(0, win_w - layout.pad - map_w - 3 * layout.pad)
    screen.blit(
        title, (layout.pad, (layout.header - title.get_height()) // 2),
        pygame.Rect(0, 0, title_w, title.get_height()),
    )
    legend_items = [
        ("S", (48, 120, 32)), ("E", (232, 48, 32)),
        ("P", _TYPE_RGB[ZoneType.PRIORITY]),
        ("R", _TYPE_RGB[ZoneType.RESTRICTED]),
        ("#", _MUTED), ("O", _TYPE_RGB[ZoneType.NORMAL]),
    ]
    legend_w = sum(
        fonts[1].size(letter)[0] + 22 * scale for letter, _ in legend_items
    )
    lx = win_w - layout.pad
    mid_y = win_h - layout.footer // 2
    for letter, color in reversed(legend_items):
        glyph = fonts[1].render(letter, True, _FG)
        screen.blit(glyph, glyph.get_rect(midright=(lx, mid_y)))
        lx -= glyph.get_width() + 6 * scale
        pygame.draw.circle(screen, color, (lx - 4 * scale, mid_y), 5 * scale)
        lx -= 16 * scale
    if line:
        glyph = fonts[1].render(line, True, _FG)
        line_w = max(0, win_w - 2 * layout.pad - legend_w - 4 * scale)
        screen.blit(
            glyph,
            (layout.pad,
             win_h - layout.footer + (layout.footer - glyph.get_height()) // 2),
            pygame.Rect(0, 0, line_w, glyph.get_height()),
        )
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


def _make_fonts(pygame: Any, layout: _Layout) -> tuple[Any, Any, Any, Any]:
    """Header, chrome, zone-name and drone fonts.

    Name and drone sizes derive from the pitch, so labels keep
    fitting when a large map shrinks the grid instead of being
    gated away entirely.
    """
    scale = layout.scale
    pitch = layout.pitch
    # Vertical room under a circle before the next row's circle:
    # with r = 0.30 * pitch, the label band can use 0.40 * pitch.
    # The default font's real height is about 0.65x its nominal size
    # (measured), hence the divisor.
    name_size = int((pitch * 0.40 - 4 * scale) / 0.65)
    name_size = max(9 * scale, min(26 * scale, name_size))
    drone_r = _drone_radius(pitch, scale)
    drone_size = max(9 * scale, min(17 * scale, int(drone_r * 0.9)))
    return (
        pygame.font.Font(None, 30 * scale),   # header title
        pygame.font.Font(None, 17 * scale),   # chrome: legend, lines
        pygame.font.Font(None, name_size),    # zone names
        pygame.font.Font(None, drone_size),   # drone ids, badges
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


def _default_pts(pygame: Any) -> tuple[int, int]:
    """One default window size for every map, screen-clamped."""
    try:
        info = pygame.display.Info()
        dw, dh = int(info.current_w), int(info.current_h)
    except pygame.error:
        dw, dh = _HARD_MAX
    w = max(_MIN_WIN[0], min(_DEFAULT_WIN[0], int(dw * 0.8)))
    h = max(_MIN_WIN[1], min(_DEFAULT_WIN[1], int(dh * 0.8)))
    return (w, h)


def _run(
    pygame: Any,
    graph: Graph,
    result: SimulationResult,
    drones: list[Drone],
    map_label: str | None,
    has_prev: bool,
    has_next: bool,
    frame_delay_ms: int,
    auto_exit: bool = False,
    existing: _Target | None = None,
) -> SessionEnd:
    """Animate one map; returns 'quit', 'next' or 'prev'.

    ``existing`` reuses an already-open window (tour mode keeps one
    fixed-size window across maps); when omitted a fresh default-size
    window is opened and destroyed on exit.

    auto_exit closes after the final turn (used by the headless smoke
    test): 'next' when another map follows, else 'quit'.
    """
    last_turn = max(1, len(result.turns))
    deliveries = _delivery_turns(drones)
    title = f"Fly-in - {len(drones)} drones"

    owns_window = existing is None
    if existing is None:
        target = _make_target(pygame, title, _default_pts(pygame))
    else:
        target = existing
        if target.window is not None:
            target.window.title = title
    try:
        layout = _layout(graph, target.px, target.scale)
        fonts = _make_fonts(pygame, layout)
        frames = _keyframes(graph, drones, len(result.turns), layout)
        layer = _make_map_layer(pygame, graph, layout, fonts)
        turn = 0
        paused = False
        anim_start = pygame.time.get_ticks()

        def rebuild() -> None:
            """Relayout after a window resize (pitch stays static)."""
            nonlocal layout, fonts, frames, target, layer
            if target.window is not None:
                target = _Target(
                    window=target.window,
                    screen=target.window.get_surface(),
                )
            layout = _layout(graph, target.px, target.scale)
            fonts = _make_fonts(pygame, layout)
            frames = _keyframes(graph, drones, len(result.turns), layout)
            layer = _make_map_layer(pygame, graph, layout, fonts)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
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
                        return "quit"
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
                    if event.key == pygame.K_n and has_next:
                        return "next"
                    if event.key == pygame.K_p and has_prev:
                        return "prev"
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
                pygame, target.screen, layer, fonts, layout,
                frames[turn],
                frames[min(turn + 1, last_turn)],
                frac, deliveries, turn,
            )
            line = result.turns[turn - 1] if turn >= 1 else ""
            _draw_chrome(
                pygame, target.screen, fonts, layout, turn, last_turn,
                result.makespan, paused, line, map_label,
            )
            target.flip()
            if auto_exit and turn >= last_turn:
                pygame.time.wait(250)
                return "next" if has_next else "quit"
            pygame.time.wait(15)
    finally:
        if owns_window and target.window is not None:
            target.window.destroy()


def _plan_map(
    path: str,
) -> tuple[Graph, SimulationResult, list[Drone]] | None:
    """Parse, plan, simulate and validate one map; None on failure.

    The same runtime gate as the CLI: an invalid plan never reaches
    the screen.
    """
    try:
        graph = parse_map_file(path)
        drones = plan(graph)
        result = Simulator(drones).run()
        validate(graph, drones)
    except FlyInError as exc:
        sys.stderr.write(f"skipping {path}: {exc}\n")
        return None
    return graph, result, drones


def launch(
    graph: Graph,
    result: SimulationResult,
    drones: list[Drone],
    frame_delay_ms: int = 500,
    auto_exit: bool = False,
) -> bool:
    """Play one simulation in a window; False if pygame is unavailable."""
    try:
        import pygame
    except ImportError:
        return False
    pygame.init()
    try:
        _run(
            pygame, graph, result, drones, None, False, False,
            frame_delay_ms, auto_exit,
        )
        return True
    finally:
        pygame.quit()


def launch_tour(
    paths: list[str],
    index: int = 0,
    frame_delay_ms: int = 500,
    auto_exit: bool = False,
) -> bool:
    """Play maps in order, switching with n/p; False if no pygame.

    Every map animates in the same default-size window (which the
    user can resize; the size then persists across map switches).
    """
    try:
        import pygame
    except ImportError:
        return False
    pygame.init()
    target: _Target | None = None
    try:
        target = _make_target(pygame, "Fly-in", _default_pts(pygame))
        while 0 <= index < len(paths):
            loaded = _plan_map(paths[index])
            if loaded is None:
                index += 1
                continue
            graph, result, drones = loaded
            label = (
                f"{Path(paths[index]).name}   {index + 1}/{len(paths)}"
            )
            action = _run(
                pygame, graph, result, drones, label,
                index > 0, index < len(paths) - 1,
                frame_delay_ms, auto_exit, target,
            )
            if action == "quit":
                return True
            index += 1 if action == "next" else -1
        return True
    finally:
        if target is not None and target.window is not None:
            target.window.destroy()
        pygame.quit()
