"""Simulation playback: turns planned trajectories into output lines."""

from __future__ import annotations

from fly_in2.model import Drone, SimulationResult, TurnRecord


def _delivery_turn(drone: Drone) -> int:
    """Turn on which the drone reached the end zone (0 if never)."""
    for rec in reversed(drone.trajectory):
        if rec.token is not None:
            return rec.turn
    return 0


class Simulator:
    """Merges planned trajectories into per-turn output lines.

    Pure formatting: no rules are checked and no decisions are made.
    Every turn line holds the tokens of the drones that acted, sorted
    by drone id; a turn where nobody moves produces an empty line.
    Only the trajectories are needed - the plan is already solved.
    """

    def __init__(self, drones: list[Drone]) -> None:
        self._drones = drones

    def run(self) -> SimulationResult:
        """Group every drone's records by turn and format the lines."""
        by_turn: dict[int, list[tuple[int, TurnRecord]]] = {}
        last_turn = 0
        for drone in self._drones:
            for rec in drone.trajectory:
                by_turn.setdefault(rec.turn, []).append(
                    (drone.drone_id, rec)
                )
                if rec.turn > last_turn:
                    last_turn = rec.turn

        lines: list[str] = []
        for turn in range(1, last_turn + 1):
            entries = by_turn.get(turn, [])
            entries.sort(key=lambda item: item[0])
            tokens = [
                f"D{drone_id}-{rec.token}"
                for drone_id, rec in entries
                if rec.token is not None
            ]
            lines.append(" ".join(tokens))

        makespan = max(
            (_delivery_turn(drone) for drone in self._drones), default=0
        )
        return SimulationResult(turns=lines, makespan=makespan)
