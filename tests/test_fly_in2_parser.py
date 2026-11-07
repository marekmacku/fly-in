"""Adversarial map corpus for the fly_in2 parser.

Valid maps under maps_custom/valid must parse; each map under
maps_custom/invalid must be rejected with the exact offending line
number and cause listed in EXPECTED_ERRORS.
"""

from pathlib import Path

import pytest

from fly_in2.errors import ParseError
from fly_in2.parser import parse_map, parse_map_file

MAPS_DIR = Path(__file__).resolve().parent.parent / "maps_custom"
VALID_MAPS = sorted((MAPS_DIR / "valid").glob("*.txt"))

# name -> (expected error line, expected cause fragment); a line of
# None means the error has no single offending line (missing hub).
EXPECTED_ERRORS: dict[str, tuple[int | None, str]] = {
    "i01_nb_not_first.txt": (4, "nb_drones must be the first"),
    "i02_nb_bad_value.txt": (5, "invalid integer for nb_drones: '2.5'"),
    "i03_dash_name.txt": (6, "contains a dash"),
    "i04_bad_coord.txt": (5, "x coordinate: 'one'"),
    "i05_bad_zone_type.txt": (7, "invalid zone type 'weird'"),
    "i06_bad_max_drones.txt": (6, "max_drones must be a positive"),
    "i07_meta_wrong_target.txt": (
        7, "unknown zone metadata key 'max_link_capacity'",
    ),
    "i08_duplicate_zone.txt": (8, "duplicate zone name 'alpha'"),
    "i09_conn_unknown_zone.txt": (9, "unknown zone 'ghost'"),
    "i10_conn_forward_ref.txt": (6, "unknown zone 'late'"),
    "i11_duplicate_connection.txt": (9, "duplicate connection 'e-m'"),
    "i12_self_loop.txt": (7, "cannot connect to itself"),
    "i13_double_dash.txt": (8, "exactly one dash"),
    "i14_junk_after_meta.txt": (
        6, "unexpected text after the metadata block: 'junk'",
    ),
    "i15_unknown_prefix.txt": (5, "unknown definition 'end_hubX'"),
    "i16_no_colon.txt": (6, "missing ':'"),
    "i17_hash_glued.txt": (6, "y coordinate: '2#note'"),
    "i18_second_start.txt": (7, "multiple start_hub"),
    "i19_unmatched_bracket.txt": (5, "unmatched ']'"),
    "i20_missing_end.txt": (None, "missing end_hub zone"),
    "i21_unclosed_bracket.txt": (6, "missing its closing ']'"),
    "i22_second_metadata_block.txt": (
        6, "unexpected text after the metadata block: '[zone=normal]'",
    ),
}


def test_valid_maps_parse() -> None:
    """Every nasty-but-legal map produces a graph with start and end."""
    for map_path in VALID_MAPS:
        graph = parse_map_file(str(map_path))
        assert graph.start is not None, map_path.name
        assert graph.end is not None, map_path.name
        assert graph.num_drones >= 1, map_path.name


def test_invalid_maps_reject_with_line_and_cause() -> None:
    """Every trap is rejected at its exact line with the right cause."""
    for name, (line, keyword) in EXPECTED_ERRORS.items():
        with pytest.raises(ParseError) as excinfo:
            parse_map_file(str(MAPS_DIR / "invalid" / name))
        message = str(excinfo.value)
        if line is not None:
            assert f"line {line}:" in message, f"{name}: {message}"
        assert keyword in message, f"{name}: {message}"


def test_missing_nb_drones_has_no_line() -> None:
    """An empty file fails with a message, not a crash or traceback."""
    with pytest.raises(ParseError, match="missing nb_drones line"):
        parse_map("")
