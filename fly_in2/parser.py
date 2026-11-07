"""Strict parser for the Fly-in map format: text to a validated Graph.

Every rejection raised here carries ``line N: <cause>`` so the user can
point at the exact offending line of the map file.
"""

from __future__ import annotations

import re

from fly_in2.errors import ParseError
from fly_in2.model import Connection, Graph, Zone, ZoneType

_INT_RE = re.compile(r"^-?\d+$")

# Metadata keys the spec defines for zones.
_ZONE_META_KEYS = {"zone", "color", "max_drones"}
# The spec documents only max_link_capacity for connections, but a
# color tag is a harmless visual hint, so it is tolerated there too.
_CONN_META_KEYS = {"max_link_capacity", "color"}


def _parse_int(value: str, what: str, line_no: int) -> int:
    """Parse a strict integer or raise a ParseError with the line."""
    if not _INT_RE.match(value):
        raise ParseError(
            f"line {line_no}: invalid integer for {what}: '{value}'"
        )
    return int(value)


def _strip_comment(line: str) -> str:
    """Remove a comment starting at a '#' that begins the line or is
    preceded by whitespace.

    A '#' glued directly to a token (``2#note``) is *not* a comment:
    the spec reserves no escape syntax, so such a line is left intact
    and fails integer validation with the exact line number.
    """
    for i, ch in enumerate(line):
        if ch == "#":
            if i == 0 or line[i - 1].isspace():
                return line[:i]
    return line


def _parse_metadata(raw: str, line_no: int) -> dict[str, str]:
    """Parse a ``[key=value key=value]`` block; tags may be in any order.

    Raises:
        ParseError: on missing brackets, tokens without '=', empty
            keys or values, or duplicate keys.
    """
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
                f"line {line_no}: empty key or value in metadata "
                f"'{token}'"
            )
        if key in result:
            raise ParseError(
                f"line {line_no}: duplicate metadata key '{key}'"
            )
        result[key] = value
    return result


def _validate_name(name: str, line_no: int) -> str:
    """Validate a zone name: non-empty, no dash, no whitespace.

    The dash is reserved as the connection separator, which is why the
    spec forbids it inside names.
    """
    if not name:
        raise ParseError(f"line {line_no}: empty zone name")
    if "-" in name:
        raise ParseError(
            f"line {line_no}: zone name '{name}' contains a dash"
        )
    if any(ch.isspace() for ch in name):
        raise ParseError(
            f"line {line_no}: zone name '{name}' contains whitespace"
        )
    return name


def _build_zone(
    prefix: str, name: str, x_str: str, y_str: str,
    meta: dict[str, str], line_no: int,
) -> Zone:
    """Build a Zone from parsed tokens, validating types and metadata.

    Raises:
        ParseError: on a bad zone type, non-single-word color, or a
            non-positive max_drones value.
    """
    name = _validate_name(name, line_no)
    x = _parse_int(x_str, "x coordinate", line_no)
    y = _parse_int(y_str, "y coordinate", line_no)
    zone_type = ZoneType.NORMAL
    color: str | None = None
    max_drones = 1
    for key, value in meta.items():
        if key not in _ZONE_META_KEYS:
            raise ParseError(
                f"line {line_no}: unknown zone metadata key '{key}'"
            )
        if key == "zone":
            try:
                zone_type = ZoneType.from_str(value)
            except ValueError as exc:
                raise ParseError(f"line {line_no}: {exc}") from exc
        elif key == "color":
            if any(ch.isspace() for ch in value):
                raise ParseError(
                    f"line {line_no}: color must be a single word"
                )
            color = value
        elif key == "max_drones":
            max_drones = _parse_int(value, "max_drones", line_no)
            if max_drones <= 0:
                raise ParseError(
                    f"line {line_no}: max_drones must be a positive "
                    "integer"
                )
    return Zone(
        name=name,
        x=x,
        y=y,
        zone_type=zone_type,
        color=color,
        max_drones=max_drones,
        is_start=prefix == "start_hub",
        is_end=prefix == "end_hub",
    )


def _build_connection(
    src: str, dst: str, meta: dict[str, str], line_no: int
) -> Connection:
    """Build a Connection from parsed tokens, validating metadata.

    Raises:
        ParseError: on a non-single-word color or a non-positive
            max_link_capacity value.
    """
    src = _validate_name(src, line_no)
    dst = _validate_name(dst, line_no)
    max_link_capacity = 1
    for key, value in meta.items():
        if key not in _CONN_META_KEYS:
            raise ParseError(
                f"line {line_no}: unknown connection metadata key "
                f"'{key}'"
            )
        if key == "max_link_capacity":
            max_link_capacity = _parse_int(
                value, "max_link_capacity", line_no
            )
            if max_link_capacity <= 0:
                raise ParseError(
                    f"line {line_no}: max_link_capacity must be positive"
                )
        elif key == "color":
            if any(ch.isspace() for ch in value):
                raise ParseError(
                    f"line {line_no}: color must be a single word"
                )
    return Connection(src=src, dst=dst, max_link_capacity=max_link_capacity)


def _split_definition(
    line: str, line_no: int
) -> tuple[str, list[str], str]:
    """Split a ``prefix: body [meta]`` line into its three parts.

    Raises:
        ParseError: on a missing colon, empty prefix, empty body, an
            unclosed metadata block, text after the metadata block,
            or an unmatched closing bracket.
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
        closing = meta_raw.find("]")
        if closing == -1:
            raise ParseError(
                f"line {line_no}: metadata block is missing its "
                "closing ']'"
            )
        trailing = meta_raw[closing + 1:].strip()
        if trailing:
            raise ParseError(
                f"line {line_no}: unexpected text after the metadata "
                f"block: '{trailing}'"
            )
    elif rest.endswith("]") and "[" not in rest:
        raise ParseError(f"line {line_no}: unmatched ']' in definition")
    return prefix, rest.split(), meta_raw


def _add_zone(graph: Graph, zone: Zone, line_no: int) -> None:
    """Register a zone, stamping model errors with the line number."""
    try:
        graph.add_zone(zone)
    except ValueError as exc:
        raise ParseError(f"line {line_no}: {exc}") from exc


def _add_connection(graph: Graph, conn: Connection, line_no: int) -> None:
    """Register a connection, stamping model errors with the line."""
    try:
        graph.add_connection(conn)
    except ValueError as exc:
        raise ParseError(f"line {line_no}: {exc}") from exc


def parse_map(text: str) -> Graph:
    """Parse a full map document into a validated Graph.

    Raises:
        ParseError: on the first invalid line, with its number and
            cause; or without a number when the file lacks a
            nb_drones line, a start hub or an end hub (no single
            offending line exists for those).
    """
    graph = Graph()
    drone_count: int | None = None
    nb_seen = False
    for index, raw_line in enumerate(text.splitlines(), start=1):
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
                raise ParseError(
                    f"line {index}: duplicate nb_drones line"
                )
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
        if prefix in ("start_hub", "end_hub", "hub"):
            if len(body) != 3:
                raise ParseError(
                    f"line {index}: {prefix} expects exactly name x y"
                )
            zone = _build_zone(
                prefix, body[0], body[1], body[2], meta, index
            )
            _add_zone(graph, zone, index)
            continue
        if prefix == "connection":
            if len(body) != 1:
                raise ParseError(
                    f"line {index}: connection expects a single "
                    "'a-b' token"
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
            return parse_map(handle.read())
    except UnicodeDecodeError as exc:
        raise ParseError(
            f"cannot read file '{path}': not valid UTF-8 text"
        ) from exc
    except OSError as exc:
        raise ParseError(f"cannot read file '{path}': {exc}") from exc
