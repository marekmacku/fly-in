"""Animated pygame visualization of the simulation (optional --gui).

Zones have a fixed base size (pitch 60 points) on every map, small
or large; every map opens in the same default-size window, which the
user can resize (the size persists across map switches in a tour),
and the pitch only shrinks when a map cannot fit the window.
Rendering goes through pygame's Window API with
allow_high_dpi, so the surface is at the display's native resolution
(no stretched, blurry upscaling). The animation starts at turn 0 -
the whole fleet parked on the start zone - and drones glide between
their per-turn positions, easing out of one zone and into the next; a
drone flying a restricted move spends one full turn travelling along
its connection before arriving, eased over the whole crossing. The
links flown during the shown turn glow. The footer shows the
simulation's own output line for the current turn, and each line is
also printed to stdout as its moves start (once per turn, in order,
in the --plain format). Zones with capacity above one carry a badge.
Label sizes derive from the pitch, so text always fits instead of
disappearing. Text uses a clean system sans when one is installed.

Controls: space play/pause, left/right arrow step one turn, up/down
arrow speed up/slow down, home/end first/last turn, n/p next or
previous map (tour mode), r restart, h reset the view, mouse wheel
zoom (anchored at the cursor), left-drag pan, ? show/hide the
controls, q or escape quit.

pygame is imported lazily inside launch(), so the program runs fine
without it; the CLI falls back to the colored terminal renderer.
Long coordinate gaps are squashed logarithmically before drawing
(visualizer.display_positions), so absurd coordinates stay on screen.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fly_in2.errors import FlyInError
from fly_in2.model import (
    Connection,
    Drone,
    Graph,
    SimulationResult,
    Zone,
    ZoneType,
)
from fly_in2.parser import parse_map_file
from fly_in2.planner import plan
from fly_in2.simulator import Simulator
from fly_in2.validator import validate
from fly_in2.visualizer import (
    Positions,
    display_positions,
    positions_per_turn,
)

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

# First installed wins; the built-in pygame font is the last resort.
_FONT_NAMES = "helveticaneue,inter,notosans,cantarell,dejavusans"
_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)   # up/down arrow steps

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
    """Width and height of the map in drawn (squashed) grid units."""
    at = display_positions(graph).values()
    xs = [x for x, _ in at]
    ys = [y for _, y in at]
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


def _ring(n: int) -> list[tuple[float, float]]:
    """Unit offsets fanning n stacked zones out on a ring, first left."""
    return [
        (math.cos(a), math.sin(a))
        for a in (math.pi + 2 * math.pi * k / n for k in range(n))
    ]


def _layout(
    graph: Graph,
    px: tuple[int, int],
    scale: int,
    zoom: float = 1.0,
    pan: tuple[int, int] = (0, 0),
) -> _Layout:
    """Place zones on the grid; zoom and pan are user view state.

    The base pitch is _PITCH points unless the map cannot fit the
    window; the user's zoom factor scales it from there, and zoomed
    content may overflow the window edges (that is what zooming is).
    It never grows past the base: small maps get a small window's
    worth of content, not giant zones.
    """
    span_x, span_y = _spans(graph)
    # Zones sharing one position fan out on a ring just wide enough
    # for their circles to touch, so every one of them stays visible.
    # The stack's grid column (and row) widens by the ring's reach on
    # each side, so the links to the zones before and after it grow
    # and the stack keeps the normal gap to them. Sizes in pitches;
    # 0.30 is _zone_radius, 1.15 the larger start/end hub size.
    stacks = graph.stacked_zones()
    stack_r = 0.30 * 1.15
    grow_x: dict[int, float] = {}
    grow_y: dict[int, float] = {}
    at = display_positions(graph)
    for names in stacks:
        sx, sy = at[names[0]]
        ring = stack_r / math.sin(math.pi / len(names))
        offsets = _ring(len(names))
        reach_x = ring * max(abs(dx) for dx, _ in offsets)
        reach_y = ring * max(abs(dy) for _, dy in offsets)
        grow_x[sx] = max(grow_x.get(sx, 0.0), reach_x)
        grow_y[sy] = max(grow_y.get(sy, 0.0), reach_y)
    width = span_x + 2 * sum(grow_x.values())
    height = span_y + 2 * sum(grow_y.values())
    pad, header, footer = _PAD * scale, _HEADER * scale, _FOOTER * scale
    fit = min(
        max(1, (px[0] - 2 * pad) // max(1, width)),
        max(1, (px[1] - header - footer - 2 * pad) // max(1, height)),
    )
    base = min(_PITCH * scale, fit)
    pitch = max(12 * scale, int(base * zoom))
    min_x = min(x for x, _ in at.values())
    max_y = max(y for _, y in at.values())
    x0 = int((px[0] - width * pitch) // 2) + pan[0]
    y0 = int(
        header + (px[1] - header - footer - height * pitch) // 2
    ) + pan[1]

    def col(x: int) -> float:
        """Column centre, in pitches from the grid's left edge."""
        wider = sum(g for gx, g in grow_x.items() if gx < x)
        return x - min_x + 0.5 + 2 * wider + grow_x.get(x, 0.0)

    def row(y: int) -> float:
        """Row centre, in pitches from the grid's top edge."""
        wider = sum(g for gy, g in grow_y.items() if gy > y)
        return max_y - y + 0.5 + 2 * wider + grow_y.get(y, 0.0)

    pixels = {
        name: (x0 + int(col(x) * pitch), y0 + int(row(y) * pitch))
        for name, (x, y) in at.items()
    }
    for names in stacks:
        cx, cy = pixels[names[0]]
        ring_px = stack_r * pitch / math.sin(math.pi / len(names))
        for name, (dx, dy) in zip(names, _ring(len(names))):
            pixels[name] = (cx + int(ring_px * dx), cy + int(ring_px * dy))
    return _Layout(
        px=px,
        scale=scale,
        pitch=pitch,
        pad=pad,
        header=header,
        footer=footer,
        pixels=pixels,
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
    label_w: float,
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
    # Name below the circle, at one uniform font for the whole map:
    # long names truncate to label_w. Never inside the circle, where
    # the drones live.
    color = _MUTED if zone.zone_type is ZoneType.BLOCKED else _FG
    shown = zone.name
    label = fonts[2].render(shown, True, color)
    while label.get_width() > label_w and len(shown) > 4:
        shown = shown[:-3] + ".."
        label = fonts[2].render(shown, True, color)
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
    stacked = {name for names in graph.stacked_zones() for name in names}
    for name, zone in graph.zones.items():
        cx, cy = layout.pixels[name]
        zone_r = _zone_r(zone, layout)
        # Fanned-out zones sit side by side: labels get one circle's width.
        label_w = 2 * zone_r if name in stacked else layout.pitch * 0.95
        _draw_zone(
            pygame, layer, fonts, zone, cx, cy, zone_r,
            layout.pitch, layout.scale, label_w,
        )
    return layer


def _zone_radius(pitch: int) -> int:
    """Circle radius: well below the pitch so links stay long."""
    return max(6, int(pitch * 0.30))


def _zone_r(zone: Zone, layout: _Layout) -> int:
    """Drawn radius of one zone; start and end hubs are a bit larger."""
    base_r = _zone_radius(layout.pitch)
    if zone.is_start or zone.is_end:
        return min(layout.pitch // 2 - 4 * layout.scale, int(base_r * 1.15))
    return base_r


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


def _flights(
    positions: Positions,
    transit: dict[int, dict[int, tuple[str, str]]],
    start: str,
    n_turns: int,
) -> list[dict[int, tuple[str, str]]]:
    """Per turn: drone id -> (src, dst) of the link it flies that turn.

    Index 0 (the parked fleet) is empty, waiting drones are absent,
    and a restricted crossing appears on both of its turns.
    """
    flights: list[dict[int, tuple[str, str]]] = [{}]
    for turn in range(1, n_turns + 1):
        legs: dict[int, tuple[str, str]] = {}
        for did, loc in positions[turn].items():
            before = positions[turn - 1].get(did) if turn > 1 else start
            edge: tuple[str, str] | None
            if loc is None:
                edge = transit.get(turn, {}).get(did)
            elif before is None:
                edge = transit.get(turn - 1, {}).get(did)
            elif before != loc:
                edge = (before, loc)
            else:
                continue
            if edge is not None:
                legs[did] = edge
        flights.append(legs)
    return flights


def _keyframes(
    graph: Graph,
    drones: list[Drone],
    n_turns: int,
    layout: _Layout,
    positions: Positions,
    transit: dict[int, dict[int, tuple[str, str]]],
) -> list[dict[int, tuple[int, int]]]:
    """Drone pixel positions per turn, starting at turn 0.

    Frame 0 is the whole fleet on the start zone. A drone in restricted
    transit sits midway along its connection that turn, so it crosses
    the link over two animation frames.
    """
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


def _ease(f: float) -> float:
    """Smoothstep: zero speed at both ends, fastest midway."""
    return f * f * (3.0 - 2.0 * f)


def _glide(f: float, from_mid: bool, to_mid: bool) -> float:
    """Eased progress along one turn's leg.

    A restricted crossing spans two legs (zone, link midpoint, zone)
    and eases over the whole crossing, so the drone accelerates out of
    its zone and decelerates into the next without stopping midway.
    """
    if to_mid and not from_mid:
        return 2.0 * _ease(f / 2.0)
    if from_mid and not to_mid:
        return 2.0 * _ease((1.0 + f) / 2.0) - 1.0
    return _ease(f)


def _drone_points(
    frames: list[dict[int, tuple[int, int]]],
    transit: dict[int, dict[int, tuple[str, str]]],
    tau: float,
) -> dict[int, tuple[int, int]]:
    """Drone pixels at continuous time tau; turn k spans k-1 .. k."""
    if tau <= 0.0 or len(frames) < 2:
        return dict(frames[0])
    leg = min(len(frames) - 1, math.ceil(tau))
    f = min(1.0, tau - (leg - 1))
    src, dst = frames[leg - 1], frames[leg]
    was_mid, is_mid = transit.get(leg - 1, {}), transit.get(leg, {})
    return {
        did: _lerp(
            point, dst.get(did, point),
            _glide(f, did in was_mid, did in is_mid),
        )
        for did, point in src.items()
    }


def _draw_highlights(
    pygame: Any,
    screen: Any,
    graph: Graph,
    layout: _Layout,
    flying: dict[int, tuple[str, str]],
) -> None:
    """Glow over every link flown during the shown turn.

    Trimmed to the zone rims so the zone art stays untouched.
    """
    width = max(2 * layout.scale, layout.pitch // 40)
    for key in {Connection(*edge).key() for edge in flying.values()}:
        za, zb = graph.zones[key[0]], graph.zones[key[1]]
        (ax, ay), (bx, by) = layout.pixels[za.name], layout.pixels[zb.name]
        length = math.hypot(bx - ax, by - ay)
        ra, rb = _zone_r(za, layout), _zone_r(zb, layout)
        if length <= ra + rb:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
        p = (int(ax + ux * ra), int(ay + uy * ra))
        q = (int(bx - ux * rb), int(by - uy * rb))
        pygame.draw.line(screen, _mix(_BG, _ACCENT, 0.3), p, q, 3 * width)
        pygame.draw.line(screen, _ACCENT, p, q, width)


def _draw_drones(
    pygame: Any,
    screen: Any,
    fonts: tuple[Any, Any, Any, Any],
    layout: _Layout,
    points: dict[int, tuple[int, int]],
    deliveries: dict[int, int],
    tau: float,
) -> None:
    """The drone markers; delivered drones are dimmed."""
    scale = layout.scale
    pitch = layout.pitch
    drone_r = _drone_radius(pitch, scale)
    for did, (x, y) in points.items():
        dim = deliveries.get(did, 0) <= tau
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


_HELP = (
    ("space", "play / pause"),
    ("left / right", "step one turn"),
    ("up / down", "faster / slower"),
    ("home / end", "first / last turn"),
    ("n / p", "next / previous map"),
    ("r", "restart"),
    ("wheel / drag", "zoom / pan"),
    ("h", "reset the view"),
    ("?", "show / hide this help"),
    ("q / esc", "quit"),
)


def _draw_help(
    pygame: Any, screen: Any, font: Any, layout: _Layout
) -> None:
    """The controls, in a panel centred on the window."""
    scale = layout.scale
    pad, gap = 14 * scale, 24 * scale
    keys = [font.render(key, True, _ACCENT) for key, _ in _HELP]
    acts = [font.render(act, True, _FG) for _, act in _HELP]
    key_w = max(g.get_width() for g in keys)
    row_h = max(g.get_height() for g in keys + acts) + 4 * scale
    box = pygame.Rect(
        0, 0,
        key_w + gap + max(g.get_width() for g in acts) + 2 * pad,
        row_h * len(_HELP) + 2 * pad,
    )
    box.center = (layout.px[0] // 2, layout.px[1] // 2)
    pygame.draw.rect(screen, _PANEL, box, border_radius=8 * scale)
    pygame.draw.rect(screen, _ACCENT, box, scale, border_radius=8 * scale)
    for i, (key, act) in enumerate(zip(keys, acts)):
        y = box.y + pad + i * row_h
        screen.blit(key, (box.x + pad, y))
        screen.blit(act, (box.x + pad + key_w + gap, y))


def _draw_chrome(
    pygame: Any,
    screen: Any,
    fonts: tuple[Any, Any, Any, Any],
    layout: _Layout,
    status: str,
    progress: float,
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
    title = fonts[0].render(status, True, _FG)
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
             win_h - layout.footer
             + (layout.footer - glyph.get_height()) // 2),
            pygame.Rect(0, 0, line_w, glyph.get_height()),
        )
    bar_x0, bar_x1 = layout.pad, win_w - layout.pad
    bar_y = win_h - 9 * scale
    pygame.draw.line(
        screen, _DOT, (bar_x0, bar_y), (bar_x1, bar_y), 3 * scale
    )
    head = bar_x0 + int((bar_x1 - bar_x0) * progress)
    pygame.draw.line(
        screen, _ACCENT, (bar_x0, bar_y), (head, bar_y), 3 * scale
    )


def _font(pygame: Any, size: int) -> Any:
    """The first installed font of _FONT_NAMES, else pygame's own.

    A system font at 0.75x the nominal size sets text as wide as the
    built-in font does (measured on Helvetica Neue), so the size
    table below stays valid for either.
    """
    path = pygame.font.match_font(_FONT_NAMES)
    if path is None:
        return pygame.font.Font(None, size)
    return pygame.font.Font(path, max(1, int(size * 0.75)))


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
    # The divisor is the font's measured line height per unit size.
    ratio = _font(pygame, 100).get_height() / 100
    name_size = int((pitch * 0.40 - 4 * scale) / ratio)
    name_size = max(9 * scale, min(26 * scale, name_size))
    drone_r = _drone_radius(pitch, scale)
    drone_size = max(9 * scale, min(17 * scale, int(drone_r * 0.9)))
    return (
        _font(pygame, 30 * scale),   # header title
        _font(pygame, 17 * scale),   # chrome: legend, lines
        _font(pygame, name_size),    # zone names
        _font(pygame, drone_size),   # drone ids, badges
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
    n_turns = len(result.turns)
    last_turn = max(1, n_turns)
    deliveries = _delivery_turns(drones)
    positions = positions_per_turn(graph, drones, n_turns)
    assert graph.start is not None
    transit = _transit_edges(drones)
    flights = _flights(positions, transit, graph.start.name, n_turns)
    title = f"Fly-in - {len(drones)} drones"

    owns_window = existing is None
    if existing is None:
        target = _make_target(pygame, title, _default_pts(pygame))
    else:
        target = existing
        if target.window is not None:
            target.window.title = title
    try:
        zoom = 1.0
        pan = (0, 0)
        dragging = False
        show_help = False
        layout = _layout(graph, target.px, target.scale, zoom, pan)
        fonts = _make_fonts(pygame, layout)
        frames = _keyframes(
            graph, drones, n_turns, layout, positions, transit
        )
        layer = _make_map_layer(pygame, graph, layout, fonts)
        # Turn `turn` glides frames[turn - 1] -> frames[turn]; frac is
        # its progress, so a settled frame shows the turn's end state
        # (turn 0's "glide" is a one-turn hold on the parked fleet).
        turn = 0
        frac = 0.0
        paused = False
        speed_i = _SPEEDS.index(1.0)
        printed = 0          # turns already written to stdout
        tick = pygame.time.get_ticks()

        def rebuild() -> None:
            """Relayout after resize, zoom or pan (user view state)."""
            nonlocal layout, fonts, frames, target, layer
            if target.window is not None:
                target = _Target(
                    window=target.window,
                    screen=target.window.get_surface(),
                )
            layout = _layout(graph, target.px, target.scale, zoom, pan)
            fonts = _make_fonts(pygame, layout)
            frames = _keyframes(
                graph, drones, n_turns, layout, positions, transit
            )
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
                    if event.key == pygame.K_ESCAPE and show_help:
                        show_help = False   # escape closes help first
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return "quit"
                    if event.unicode == "?" or event.key == pygame.K_SLASH:
                        show_help = not show_help
                    if event.key == pygame.K_HOME:
                        turn, frac, paused = 0, 1.0, True
                    if event.key == pygame.K_END:
                        turn, frac, paused = last_turn, 1.0, True
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_RIGHT:
                        paused = True
                        if turn < last_turn:
                            turn += 1
                            frac = 0.0   # the stepped turn still glides
                    if event.key == pygame.K_LEFT:
                        paused = True
                        turn = max(0, turn - 1)
                        frac = 1.0
                    if event.key == pygame.K_UP:
                        speed_i = min(len(_SPEEDS) - 1, speed_i + 1)
                    if event.key == pygame.K_DOWN:
                        speed_i = max(0, speed_i - 1)
                    if event.key == pygame.K_r:
                        turn = 0
                        frac = 0.0
                        paused = False
                    if event.key == pygame.K_n and has_next:
                        return "next"
                    if event.key == pygame.K_p and has_prev:
                        return "prev"
                    if event.key == pygame.K_h:
                        zoom = 1.0
                        pan = (0, 0)
                        rebuild()
                if event.type == pygame.MOUSEWHEEL:
                    old_zoom = zoom
                    zoom = max(0.5, min(4.0, zoom * 1.15 ** event.y))
                    if zoom != old_zoom:
                        # Anchor at the cursor: the map point under the
                        # mouse stays under the mouse after zooming.
                        mx, my = pygame.mouse.get_pos()
                        ref_x = target.px[0] / 2
                        ref_y = layout.header + (
                            target.px[1]
                            - layout.header - layout.footer
                        ) / 2
                        ratio = zoom / old_zoom
                        pan = (
                            int((mx - ref_x)
                                - (mx - ref_x - pan[0]) * ratio),
                            int((my - ref_y)
                                - (my - ref_y - pan[1]) * ratio),
                        )
                        rebuild()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging = True
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging = False
                if event.type == pygame.MOUSEMOTION and dragging:
                    pan = (pan[0] + event.rel[0], pan[1] + event.rel[1])
                    rebuild()
            # Advance by elapsed time, so a speed change never jumps.
            now = pygame.time.get_ticks()
            frac = min(
                1.0,
                frac + (now - tick) * _SPEEDS[speed_i] / frame_delay_ms,
            )
            tick = now
            if not paused and frac >= 1.0 and turn < last_turn:
                turn += 1
                frac = 0.0
            done = turn >= last_turn and frac >= 1.0
            # Mirror the output on stdout as the moves start: each turn
            # once, in order (a jump ahead flushes the skipped turns).
            while printed < min(turn, n_turns):
                printed += 1
                sys.stdout.write(result.turns[printed - 1] + "\n")
                sys.stdout.flush()
            tau = turn - 1 + frac if turn else 0.0
            screen = target.screen
            screen.blit(layer, (0, 0))
            _draw_highlights(pygame, screen, graph, layout, flights[turn])
            points = _drone_points(frames, transit, tau)
            _draw_drones(
                pygame, screen, fonts, layout, points, deliveries, tau
            )
            line = result.turns[turn - 1] if turn >= 1 else ""
            if done:
                state = f"DONE  total turns: {result.makespan}"
            else:
                state = "PAUSED" if paused else "PLAYING"
            delivered = sum(1 for t in deliveries.values() if t <= tau)
            status = (
                f"Fly-in   Turn {turn}/{last_turn}   {state}   "
                f"{_SPEEDS[speed_i]:g}x   "
                f"delivered {delivered}/{len(drones)}   -   ? help"
            )
            _draw_chrome(
                pygame, screen, fonts, layout, status, tau / last_turn,
                line, map_label,
            )
            if show_help:
                _draw_help(pygame, screen, fonts[1], layout)
            target.flip()
            if auto_exit and done:
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
    """Play one simulation in a window.

    Returns False when there is no GUI to show - pygame missing, or no
    display to open a window on - so the CLI can fall back to the
    terminal renderer instead of crashing.
    """
    # pygame prints a banner on import; keep stdout to the moves alone.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
    except ImportError:
        return False
    pygame.init()
    try:
        try:
            target = _make_target(
                pygame, f"Fly-in - {len(drones)} drones",
                _default_pts(pygame),
            )
        except pygame.error:
            return False
        try:
            _run(
                pygame, graph, result, drones, None, False, False,
                frame_delay_ms, auto_exit, target,
            )
        finally:
            if target.window is not None:
                target.window.destroy()
        return True
    finally:
        pygame.quit()


def launch_tour(
    paths: list[str],
    index: int = 0,
    frame_delay_ms: int = 500,
    auto_exit: bool = False,
) -> bool:
    """Play maps in order, switching with n/p.

    Every map animates in the same default-size window (which the
    user can resize; the size then persists across map switches).
    Returns False when there is no GUI to show (pygame missing or no
    display), so the CLI can fall back to the terminal.
    """
    # pygame prints a banner on import; keep stdout to the moves alone.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
    except ImportError:
        return False
    pygame.init()
    target: _Target | None = None
    try:
        try:
            target = _make_target(pygame, "Fly-in", _default_pts(pygame))
        except pygame.error:
            return False
        while 0 <= index < len(paths):
            loaded = _plan_map(paths[index])
            if loaded is None:
                index += 1
                continue
            graph, result, drones = loaded
            label = (
                f"{Path(paths[index]).name}   {index + 1}/{len(paths)}"
            )
            sys.stdout.write(f"\n== {Path(paths[index]).name} ==\n")
            sys.stdout.flush()
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
