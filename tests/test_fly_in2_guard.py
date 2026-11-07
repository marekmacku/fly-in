"""Guard tests: absurd fleets fail fast instead of hanging."""

import re

import pytest

from fly_in2.errors import SimulationError
from fly_in2.parser import parse_map
from fly_in2.planner import plan

ABSURD = """\
nb_drones: 890582938590238590238590238590238590238590238590238590238590
start_hub: s 0 0
hub: w1 1 0
hub: w2 2 0
end_hub: e 3 0
connection: s-w1
connection: w1-w2
connection: w2-e
"""


def test_absurd_drone_count_fails_fast() -> None:
    """A 60-digit fleet is rejected instantly, not looped over."""
    with pytest.raises(SimulationError, match="too many drones"):
        plan(parse_map(ABSURD))


def test_reasonable_fleets_still_pass_the_guard() -> None:
    """The guard must not reject fleets that fit the horizon."""
    text = re.sub(r"nb_drones: \d+", "nb_drones: 2000", ABSURD, count=1)
    drones = plan(parse_map(text))
    assert len(drones) == 2000
