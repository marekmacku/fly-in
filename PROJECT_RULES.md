# Project Rules

## Language
Python 3.10+ (stdlib only at runtime). flake8 + mypy --strict mandatory.

## Mandatory requirements
- Parser for the Fly-in map format with strict validation (line/cause errors).
- Simulation engine: discrete turns, zone & connection capacity, movement costs
  (normal/priority=1, restricted=2, blocked=inaccessible).
- Start zone: unlimited capacity (all drones begin there).
- End zone: unlimited capacity (delivered drones).
- Multi-turn restricted movement: drone occupies connection in transit, MUST
  arrive next turn (cannot wait on connection).
- Pathfinding that minimizes total simulation turns across all drones.
- Visual representation: colored terminal output (drone movements & zone states).
- Output format: per-turn line of `D<ID>-<zone>` / `D<ID>-<connection>` tokens.
- Fully object-oriented, fully type-safe (mypy --strict, flake8).
- Makefile with install/run/debug/clean/lint/(lint-strict).
- README.md with required sections + algorithm description.
- .gitignore.

## Forbidden approaches
- Graph libraries (networkx, graphlib, etc.).
- Any runtime dependency not in the standard library.
- Hardcoding solutions to sample maps.

## Allowed dependencies
- Runtime: Python standard library only.
- Dev/lint: flake8, mypy, pytest (Makefile install).

## Required files
- Makefile, README.md, .gitignore, setup.cfg, main.py, fly_in/ package, tests/.

## Input/output contract
- Input: path to a map file (CLI arg or stdin fallback).
- Output (stdout): one line per simulation turn listing drone movements.
- Visual colored output toggle via flags; movement lines stay parseable.

## Performance thresholds
- Easy: <=6-8 turns; Medium: <=12-15 turns; Hard: <=30-45 turns.
- Challenger (optional): beat 45 turns on The Impossible Dream.

## Complexity constraints
- Per-drone planning: Dijkstra on time-expanded graph, O(V*T log(V*T)) per drone.
- Static shortest path: O((V+E) log V) Dijkstra. No graph libraries.

## Platform assumptions
- Linux evaluation. Portable POSIX/Python behavior only.

## Deliverables
- A working simulation with parser, engine, pathfinder, visualizer, Makefile, README.

## Bonus
- Meet/beat all performance targets; solve challenger map under 45 turns.

## Open questions
- Exact semantics of restricted 2-turn transit are ambiguous in the spec;
  chosen model documented in README and code (transit 1 turn on connection,
  arrival next turn, drone busy for 2 turns, ready to act on the 3rd turn).
