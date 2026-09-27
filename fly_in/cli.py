"""Command-line interface for the Fly-in drone simulation."""

from __future__ import annotations

import argparse
import sys
from typing import TextIO

from fly_in.errors import FlyInError
from fly_in.model import Drone, Graph
from fly_in.parser import parse_map_file
from fly_in.planner import plan
from fly_in.simulator import Simulator
from fly_in.visualizer import Visualizer


def _drone_positions(
    graph: Graph, drones: list[Drone], makespan: int
) -> list[dict[int, str | None]]:
    """Compute each drone's location for every turn 1..makespan+1."""
    assert graph.start is not None and graph.end is not None
    start_name = graph.start.name
    end_name = graph.end.name
    positions: list[dict[int, str | None]] = [
        {} for _ in range(makespan + 2)
    ]
    for drone in drones:
        current: str | None = start_name
        records_by_turn = {rec.turn: rec for rec in drone.trajectory}
        delivery = 0
        for record in drone.trajectory:
            if record.token is not None and record.turn > delivery:
                delivery = record.turn
        for turn in range(1, makespan + 2):
            if turn > delivery and delivery > 0:
                positions[turn][drone.drone_id] = end_name
                continue
            rec = records_by_turn.get(turn)
            if rec is None:
                positions[turn][drone.drone_id] = current
            elif rec.location is not None:
                current = rec.location
                positions[turn][drone.drone_id] = current
            else:
                positions[turn][drone.drone_id] = None
    return positions


def _print_stats(graph: Graph, drones: list[Drone], makespan: int) -> None:
    moved_total = sum(
        1 for d in drones for rec in d.trajectory if rec.token is not None
    )
    avg = moved_total / len(drones) if drones else 0.0
    sys.stdout.write(f"Drones: {len(drones)}\n")
    sys.stdout.write(f"Total turns (makespan): {makespan}\n")
    sys.stdout.write(f"Total movements: {moved_total}\n")
    sys.stdout.write(f"Average moves/drone: {avg:.2f}\n")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the simulator."""
    parser = argparse.ArgumentParser(
        prog="fly-in",
        description="Route a fleet of drones through connected zones.",
    )
    parser.add_argument(
        "map",
        help="path to the map file to simulate",
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
        "--stats",
        action="store_true",
        help="print secondary performance metrics after the simulation",
    )
    return parser


def main(argv: list[str] | None = None,
         stream: TextIO | None = None) -> int:
    """Entry point: parse, plan, simulate, render."""
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
    simulator = Simulator(graph, drones)
    result = simulator.run()
    positions = _drone_positions(graph, drones, result.makespan)
    colored = not args.plain and not args.no_color
    show_grid = not args.plain and not args.no_grid
    visualizer = Visualizer(graph, out)
    visualizer.render(result, positions, show_grid, colored)
    if args.stats:
        _print_stats(graph, drones, result.makespan)
    return 0
