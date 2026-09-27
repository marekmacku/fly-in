"""Simulation engine: executes planned trajectories and emits output lines."""

from __future__ import annotations

from dataclasses import dataclass, field

from fly_in.model import Drone, Graph, TurnRecord


@dataclass
class SimulationResult:
    """Outcome of running all drone trajectories."""

    turns: list[str] = field(default_factory=list)
    makespan: int = 0
    per_turn_records: list[list[tuple[int, TurnRecord]]] = field(
        default_factory=list
    )


def _delivery_turn(drone: Drone) -> int:
    """Turn on which the drone reached the end zone (0 if never)."""
    for rec in reversed(drone.trajectory):
        if rec.token is not None:
            return rec.turn
    return 0


def label_for(drone_id: int) -> str:
    """Return the ``D<id>`` label for a drone identifier."""
    return f"D{drone_id}"


class Simulator:
    """Runs planned trajectories and groups movements per turn."""

    def __init__(self, graph: Graph, drones: list[Drone]) -> None:
        self._graph = graph
        self._drones = drones

    def run(self) -> SimulationResult:
        """Execute the simulation and return per-turn output lines."""
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
        per_turn: list[list[tuple[int, TurnRecord]]] = []
        for turn in range(1, last_turn + 1):
            entries = by_turn.get(turn, [])
            entries.sort(key=lambda item: item[0])
            tokens: list[str] = []
            for drone_id, rec in entries:
                if rec.token is not None:
                    tokens.append(f"{label_for(drone_id)}-{rec.token}")
            per_turn.append(entries)
            lines.append(" ".join(tokens))
        makespan = max(
            (_delivery_turn(d) for d in self._drones), default=0
        )
        return SimulationResult(
            turns=lines, makespan=makespan, per_turn_records=per_turn
        )
