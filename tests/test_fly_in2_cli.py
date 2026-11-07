"""Tests for fly_in2.cli: exit codes and the runtime validation gate."""

import io
from pathlib import Path

import pytest

import fly_in2.cli as cli
from fly_in2 import gui as fly_gui
from fly_in2.model import Drone, TurnRecord

ROOT = Path(__file__).resolve().parent.parent
LINEAR = ROOT / "maps_download" / "easy" / "01_linear_path.txt"
BAD_MAP = ROOT / "maps_custom" / "invalid" / "i04_bad_coord.txt"

UNREACHABLE = """\
nb_drones: 1
start_hub: s 0 0
end_hub: e 9 9
hub: wall 1 0 [zone=blocked]
connection: s-wall
connection: wall-e
"""


def test_successful_run_prints_simulation() -> None:
    out = io.StringIO()
    code = cli.main([str(LINEAR), "--plain"], out)
    assert code == 0
    text = out.getvalue()
    assert "Turn 1: D1-waypoint1" in text
    assert "Total turns: 4" in text


def test_stats_flag_prints_metrics() -> None:
    out = io.StringIO()
    code = cli.main([str(LINEAR), "--plain", "--stats"], out)
    assert code == 0
    assert "Drones: 2" in out.getvalue()


def test_invalid_map_file_exits_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = io.StringIO()
    code = cli.main([str(BAD_MAP)], out)
    assert code == 2
    assert "error: line 5" in capsys.readouterr().err
    assert out.getvalue() == ""


def test_unsolvable_map_exits_3(tmp_path: Path) -> None:
    path = tmp_path / "walled.txt"
    path.write_text(UNREACHABLE)
    code = cli.main([str(path), "--plain"], io.StringIO())
    assert code == 3


def test_binary_map_file_exits_2(tmp_path: Path) -> None:
    """Non-UTF-8 garbage is rejected like any bad file, no crash."""
    path = tmp_path / "garbage.bin"
    path.write_bytes(b"\x00\xff\xfe\x81binary\x00junk")
    code = cli.main([str(path), "--plain"], io.StringIO())
    assert code == 2


def test_gate_blocks_invalid_plan_with_no_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A sabotaged plan trips the gate: exit 4, nothing printed."""
    monkeypatch.setattr(
        cli, "plan",
        lambda graph: [Drone(drone_id=1, trajectory=[
            TurnRecord(1, "nowhere", "nowhere"),
        ])],
    )
    out = io.StringIO()
    code = cli.main([str(LINEAR), "--plain"], out)
    assert code == 4
    assert out.getvalue() == ""  # nothing unverified reaches the screen
    assert "internal error" in capsys.readouterr().err


def test_gui_flag_falls_back_without_pygame(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--gui without pygame available still renders to the terminal."""
    monkeypatch.setattr(fly_gui, "launch", lambda *a, **k: False)
    out = io.StringIO()
    code = cli.main([str(LINEAR), "--gui", "--plain"], out)
    assert code == 0
    assert "Total turns: 4" in out.getvalue()  # terminal fallback ran
    assert "pygame" in capsys.readouterr().err
