"""Command-line interface for the Fly-in drone simulation."""

from __future__ import annotations

import argparse
import sys
from typing import TextIO

from fly_in2.errors import FlyInError
from fly_in2.model import Drone
from fly_in2.parser import parse_map_file
from fly_in2.planner import plan
from fly_in2.simulator import Simulator
from fly_in2.validator import validate
from fly_in2.visualizer import Visualizer
from fly_in2 import gui


def _print_stats(
    out: TextIO, drones: list[Drone], makespan: int
) -> None:
    """Print secondary performance metrics after the simulation."""
    moved_total = sum(
        1 for d in drones for rec in d.trajectory if rec.token is not None
    )
    avg = moved_total / len(drones) if drones else 0.0
    out.write(f"Drones: {len(drones)}\n")
    out.write(f"Total turns (makespan): {makespan}\n")
    out.write(f"Total movements: {moved_total}\n")
    out.write(f"Average moves/drone: {avg:.2f}\n")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the simulator."""
    parser = argparse.ArgumentParser(
        prog="fly-in",
        description="Route a fleet of drones through connected zones.",
    )
    parser.add_argument(
        "map", help="path to the map file to simulate"
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="output only clean movement lines (no color, no grid)",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="disable the per-turn grid visualization",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable ANSI colors in the output",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="animated pygame window (falls back to terminal output "
        "if pygame is unavailable)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print secondary performance metrics after the simulation",
    )
    return parser


def main(
    argv: list[str] | None = None, stream: TextIO | None = None
) -> int:
    """Entry point: parse, plan, playback, validate, render.

    Nothing reaches the output stream until the plan has passed the
    validator, so an internal bug can never print a broken simulation.

    Returns:
        0 on success; 2 for an unreadable or invalid map file; 3 for
        an unsolvable map or planning failure; 4 when the finished
        plan fails validation (an internal error, never the map's).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    out = stream if stream is not None else sys.stdout
    try:
        graph = parse_map_file(args.map)
    except FlyInError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    try:
        drones = plan(graph)
    except FlyInError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 3
    result = Simulator(drones).run()

    verdict = validate(graph, drones)
    if not verdict.valid:
        messages = "; ".join(
            f"turn {v.turn}: {v.message}" for v in verdict.violations
        )
        sys.stderr.write(
            f"internal error: plan failed validation: {messages}\n"
        )
        return 4

    colored = not args.plain and not args.no_color
    show_grid = not args.plain and not args.no_grid
    if args.gui:
        if gui.launch(graph, result, drones):
            if args.stats:
                _print_stats(out, drones, result.makespan)
            return 0
        sys.stderr.write(
            "note: pygame is not available; using terminal output\n"
        )
    Visualizer(graph, out).render(result, drones, show_grid, colored)
    if args.stats:
        _print_stats(out, drones, result.makespan)
    return 0
