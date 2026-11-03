"""Parser for the Fly-in map file format with strict validation."""

from __future__ import annotations

import re
from typing import Callable

from fly_in.errors import ParseError
from fly_in.model import Connection, Graph, Zone, ZoneType

_HEX_RE = re.compile(r"^-?\d+$")
_ZONE_KEYS: dict[str, Callable[[Zone, str], None]] = {}
_CONN_KEYS: set[str] = {"max_link_capacity", "color"}


def _parse_int(value: str, what: str, line_no: int) -> int:
    """Parse a strict integer string or raise ParseError with the line."""
    if not _HEX_RE.match(value):
        raise ParseError(
            f"line {line_no}: invalid integer for {what}: '{value}'"
        )
    return int(value)


def _strip_comment(line: str) -> str:
    """Remove an inline comment starting at a '#' preceded by whitespace.

    A '#' that is the first non-blank character also starts a comment. A '#'
    inside a zone name cannot occur because names forbid spaces, so a '#'
    attached to a name token is preserved.
    """
    for i, ch in enumerate(line):
        if ch == "#":
            if i == 0 or line[i - 1].isspace():
                return line[:i]
    return line


def _parse_metadata(
    raw: str, line_no: int
) -> dict[str, str]:
    """Parse a ``[key=value key=value]`` metadata block."""
    text = raw.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ParseError(
            f"line {line_no}: metadata must be enclosed in brackets"
        )
    inner = text[1:-1].strip()
    result: dict[str, str] = {}
    if not inner:
        return result
    for token in inner.split():
        if "=" not in token:
            raise ParseError(
                f"line {line_no}: malformed metadata token '{token}'"
            )
        key, _, value = token.partition("=")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ParseError(
                f"line {line_no}: empty key or value in metadata '{token}'"
            )
        if key in result:
            raise ParseError(
                f"line {line_no}: duplicate metadata key '{key}'"
            )
        result[key] = value
    return result


def _validate_name(name: str, line_no: int) -> str:
    """Validate a zone name: non-empty, no dash, no space."""
    if not name:
        raise ParseError(f"line {line_no}: empty zone name")
    if "-" in name:
        raise ParseError(
            f"line {line_no}: zone name '{name}' contains a dash"
        )
    if any(c.isspace() for c in name):
        raise ParseError(
            f"line {line_no}: zone name '{name}' contains whitespace"
        )
    return name


def _build_zone(
    prefix: str, name: str, x_str: str, y_str: str,
    meta: dict[str, str], line_no: int,
) -> Zone:
    """Build a Zone from parsed tokens, validating types and capacities."""
    name = _validate_name(name, line_no)
    x = _parse_int(x_str, "x coordinate", line_no)
    y = _parse_int(y_str, "y coordinate", line_no)
    zone_type = ZoneType.NORMAL
    color: str | None = None
    max_drones = 1
    is_start = prefix == "start_hub"
    is_end = prefix == "end_hub"
    allowed = {"zone", "color", "max_drones"}
    for key, value in meta.items():
        if key not in allowed:
            raise ParseError(
                f"line {line_no}: unknown zone metadata key '{key}'"
            )
        if key == "zone":
            try:
                zone_type = ZoneType.from_str(value)
            except ValueError as exc:
                raise ParseError(
                    f"line {line_no}: {exc}"
                ) from exc
        elif key == "color":
            if any(c.isspace() for c in value):
                raise ParseError(
                    f"line {line_no}: color must be a single word"
                )
            color = value
        elif key == "max_drones":
            max_drones = _parse_int(value, "max_drones", line_no)
            if max_drones <= 0:
                raise ParseError(
                    f"line {line_no}: max_drones must be a positive integer"
                )
    # max_drones is ignored on start/end (capacity is unlimited there).
    return Zone(
        name=name,
        x=x,
        y=y,
        zone_type=zone_type,
        color=color,
        max_drones=max_drones,
        is_start=is_start,
        is_end=is_end,
    )


def _build_connection(
    src: str, dst: str, meta: dict[str, str], line_no: int
) -> Connection:
    """Build a Connection from parsed tokens, validating capacities."""
    src = _validate_name(src, line_no)
    dst = _validate_name(dst, line_no)
    max_link_capacity = 1
    for key, value in meta.items():
        if key not in _CONN_KEYS:
            raise ParseError(
                f"line {line_no}: unknown connection metadata key '{key}'"
            )
        if key == "max_link_capacity":
            max_link_capacity = _parse_int(value, "max_link_capacity", line_no)
            if max_link_capacity <= 0:
                raise ParseError(
                    f"line {line_no}: max_link_capacity must be positive"
                )
        elif key == "color":
            if any(c.isspace() for c in value):
                raise ParseError(
                    f"line {line_no}: color must be a single word"
                )
    return Connection(src=src, dst=dst, max_link_capacity=max_link_capacity)


def _split_definition(line: str, line_no: int) -> tuple[str, list[str], str]:
    """Split a ``prefix: rest [meta]`` line into (prefix, body_parts, meta_raw).

    ``body_parts`` are whitespace-separated tokens before any metadata block.
    """
    if ":" not in line:
        raise ParseError(f"line {line_no}: missing ':' in definition")
    prefix, _, rest = line.partition(":")
    prefix = prefix.strip()
    rest = rest.strip()
    if not prefix:
        raise ParseError(f"line {line_no}: empty definition prefix")
    if not rest:
        raise ParseError(f"line {line_no}: empty definition body")
    meta_raw = ""
    bracket = rest.find("[")
    if bracket != -1:
        meta_raw = rest[bracket:]
        rest = rest[:bracket].strip()
    elif rest.endswith("]") and "[" not in rest:
        raise ParseError(f"line {line_no}: unmatched ']' in definition")
    body = rest.split()
    return prefix, body, meta_raw


def _add_zone(graph: Graph, zone: Zone, line_no: int) -> None:
    """Register a zone, translating model errors into ParseError."""
    try:
        graph.add_zone(zone)
    except ValueError as exc:
        raise ParseError(f"line {line_no}: {exc}") from exc


def _add_connection(graph: Graph, conn: Connection, line_no: int) -> None:
    """Register a connection, translating model errors into ParseError."""
    try:
        graph.add_connection(conn)
    except ValueError as exc:
        raise ParseError(f"line {line_no}: {exc}") from exc


def parse_map(text: str) -> Graph:
    """Parse a full map document into a validated Graph."""
    graph = Graph()
    drone_count: int | None = None
    lines = text.splitlines()
    nb_seen = False
    for index, raw_line in enumerate(lines, start=1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        prefix, body, meta_raw = _split_definition(line, index)
        meta = _parse_metadata(meta_raw, index) if meta_raw else {}
        if prefix != "nb_drones" and not nb_seen:
            raise ParseError(
                f"line {index}: nb_drones must be the first definition"
            )
        if prefix == "nb_drones":
            if nb_seen:
                raise ParseError(f"line {index}: duplicate nb_drones line")
            if len(body) != 1:
                raise ParseError(
                    f"line {index}: nb_drones expects a single value"
                )
            drone_count = _parse_int(body[0], "nb_drones", index)
            if drone_count <= 0:
                raise ParseError(
                    f"line {index}: nb_drones must be a positive integer"
                )
            nb_seen = True
            continue
        if prefix == "start_hub":
            if len(body) != 3:
                raise ParseError(
                    f"line {index}: start_hub expects exactly name x y"
                )
            zone = _build_zone(
                "start_hub", body[0], body[1], body[2], meta, index
            )
            _add_zone(graph, zone, index)
            continue
        if prefix == "end_hub":
            if len(body) != 3:
                raise ParseError(
                    f"line {index}: end_hub expects exactly name x y"
                )
            zone = _build_zone(
                "end_hub", body[0], body[1], body[2], meta, index
            )
            _add_zone(graph, zone, index)
            continue
        if prefix == "hub":
            if len(body) != 3:
                raise ParseError(
                    f"line {index}: hub expects exactly name x y"
                )
            zone = _build_zone(
                "hub", body[0], body[1], body[2], meta, index
            )
            _add_zone(graph, zone, index)
            continue
        if prefix == "connection":
            if len(body) != 1:
                raise ParseError(
                    f"line {index}: connection expects a single 'a-b' token"
                )
            token = body[0]
            if token.count("-") != 1:
                raise ParseError(
                    f"line {index}: connection '{token}' must contain "
                    "exactly one dash separating two zone names"
                )
            src, dst = token.split("-")
            conn = _build_connection(src, dst, meta, index)
            _add_connection(graph, conn, index)
            continue
        raise ParseError(f"line {index}: unknown definition '{prefix}'")

    if not nb_seen or drone_count is None:
        raise ParseError("missing nb_drones line")
    graph.num_drones = drone_count
    try:
        graph.validate()
    except ValueError as exc:
        raise ParseError(str(exc)) from exc
    return graph


def parse_map_file(path: str) -> Graph:
    """Read and parse a map file, raising ParseError on any issue."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise ParseError(f"cannot read file '{path}': {exc}") from exc
    return parse_map(text)
