*This project has been created as part of the 42 curriculum by mmacku.*

# Fly-in

## Description

Fly-in is a drone routing simulation. Given a map describing a network of
connected zones, a fleet of drones must be routed from a central start zone to
a target end zone in the fewest possible simulation turns, while respecting
zone occupancy limits, connection capacities and per-zone-type movement costs
(normal and priority zones cost one turn, restricted zones cost two turns,
blocked zones cannot be entered).

The project is written entirely in Python (standard library only) and is
fully object-oriented and fully type-checked (`mypy --strict` and `flake8`
pass with zero warnings). It contains:

- a strict map-file **parser** with precise error reporting (line + cause);
- a **graph** model with bidirectional connections and capacity metadata;
- a **pathfinding** layer (Dijkstra) implemented from scratch, with no graph
  library;
- a **planner** that performs time-expanded prioritized planning with a
  reservation table to schedule conflict-free, capacity-respecting
  trajectories for every drone;
- a **simulator** that runs the trajectories and emits the required
  `D<ID>-<zone>` / `D<ID>-<connection>` output format;
- a **visualizer** that renders a colored per-turn grid and colorizes the
  movement lines;
- a **validator** that replays every plan and checks all movement and capacity
  rules (used by the test suite and during defense).

## Instructions

### Requirements

- Python 3.10 or later.
- Development tools: `flake8`, `mypy`, `pytest` (only needed for lint/tests).

### Installation

```bash
make install        # installs flake8, mypy and pytest in a venv you manage
# or, in a dedicated virtual environment:
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

### Execution

```bash
make run MAP=maps_download/easy/01_linear_path.txt
# or directly:
python3 main.py maps_download/easy/01_linear_path.txt
```

Output flags:

| Flag         | Effect                                                    |
|--------------|-----------------------------------------------------------|
| `--plain`    | Clean movement lines only (no color, no grid)            |
| `--no-grid`  | Disable the per-turn grid, keep colored movement lines    |
| `--no-color` | Disable ANSI colors                                        |
| `--stats`    | Print secondary metrics (moves, average moves/drone)       |

### Linting and type checking

```bash
make lint            # flake8 + mypy (project-level flags)
make lint-strict     # flake8 + mypy --strict
```

### Tests

```bash
python3 -m pytest tests/ -q
```

The test suite validates the parser, the pathfinding, the restricted-transit
semantics, and **replays every provided map** through the validator to assert
that all movement and capacity rules are respected and that the performance
targets are met.

### Debug

```bash
make debug           # runs the simulator under pdb
```

### Clean

```bash
make clean           # removes caches
```

## Algorithm choices and implementation strategy

### Static pathfinding

`fly_in/pathfinding.py` implements Dijkstra's algorithm from scratch (no graph
library is used, as required). Edge weights are the movement cost of the
*destination* zone (normal/priority = 1, restricted = 2, blocked = infinity).
Distances remain exact integer costs; the spec's "priority zones should be
prioritized" rule is honoured through a **secondary tie-break key**: among
equal-cost paths, the search prefers the one that steps through fewer
non-priority zones, so priority zones are preferred without corrupting the
reported distances. Complexity: O((V + E) log V).

### Multi-drone planning

The core of the project is `fly_in/planner.py`. It uses **time-expanded
prioritized planning** with a reservation table, a well-known multi-agent
pathfinding technique:

1. Each drone is planned one at a time, in priority order.
2. For a single drone we run Dijkstra over a **time-expanded graph** whose
   states are `(zone, ready_turn)` pairs. The start state is
   `(start_zone, 1)`; the goal is any state reaching the end zone.
3. Three actions are available from a state `(v, t)`:
   - **WAIT**: `(v, t) -> (v, t+1)`, occupies `v` at turn `t`;
   - **MOVE** to a normal/priority neighbour `u`: `(v, t) -> (u, t+1)`,
     occupies the connection at turn `t` and `u` at turn `t` (move-in). `v` is
     freed at turn `t` (a drone moving out frees capacity for that same turn,
     per the spec);
   - **TRANSIT** to a restricted neighbour `u`: `(v, t) -> (u, t+2)`, occupies
     the connection at turn `t` (in transit) and `u` at turn `t+1` (arrival).
     The drone is busy for two turns and may only act again at turn `t+2`,
     matching "restricted costs 2 turns" and "the drone MUST reach its
     destination during the next turn".
4. A **reservation table** counts occupants per `(zone, turn)` and per
   `(connection, turn)`. While planning a drone, every action is checked
   against the reservations of all previously planned drones; an action is
   rejected if it would exceed a zone's `max_drones` or a connection's
   `max_link_capacity`. The start and end zones have unlimited capacity.
5. Once a conflict-free trajectory is found, its reservations are committed
   to the table and the next drone is planned.

Because the start zone has unlimited capacity, a drone can always wait there
until the network frees up, so a plan always exists when the end zone is
reachable. To improve makespan, the planner tries several drone orderings
(ascending, descending and several deterministic shuffles) and keeps the best
result.

Complexity per drone: O(V · T · log(V · T)) where T is the search horizon
(bounded by a function of the static shortest path and the drone count).

### Restricted-zone semantics (documented interpretation)

The spec is terse about the two-turn restricted move. The implementation uses
this model, which is internally consistent and produces both output forms:

- Turn `t`: the drone leaves `v`, enters the connection and is **in transit**.
  Output: `D<id>-<src>-<dst>`.
- Turn `t+1`: the drone **arrives** at the restricted zone. Output:
  `D<id>-<zone>`.
- The drone may act again starting at turn `t+2` (the move consumed turns `t`
  and `t+1`). Once transit begins, arrival is forced — the planner only starts
  a restricted transit when the destination is guaranteed to have capacity at
  the arrival turn.

### Output generation

`fly_in/simulator.py` merges all drone trajectories by turn and emits one line
per turn. Each token is `D<id>-<zone>` for moves/arrivals or
`D<id>-<src>-<dst>` for in-transit drones. Drones that do not move on a turn
are omitted; drones that reach the end zone are no longer tracked.

### Validation

`fly_in/validator.py` replays every plan turn by turn and independently
re-checks: start/end reachability, no entry into blocked zones, valid
connections, the two-turn restricted pattern, zone capacity and connection
capacity. This is the correctness oracle the project is graded against during
defense.

## Visual representation

`fly_in/visualizer.py` provides colored terminal feedback:

- A **per-turn grid** drawn from zone `(x, y)` coordinates. Each cell shows the
  zone symbol (`S` start, `E` end, `P` priority, `R` restricted, `#` blocked,
  `O` normal) and the drones currently occupying it (`[O D1,D2]`).
- **ANSI colors** mapped from each zone's `color` metadata, with a sensible
  fallback table for common color names and a default for unknown ones.
- **Colorized movement lines** where each `D<id>-<token>` is highlighted, so
  drone movements and zone states are easy to follow during peer evaluation.

The `--plain` flag strips colors and the grid for a clean, machine-parseable
output (exactly the format required by the specification). The grid helps
evaluators see zone states and drone positions at a glance, while the movement
lines remain the authoritative simulation output.

## Example input and expected output

Input (`maps_download/easy/01_linear_path.txt`):

```
# Easy Level 1: Simple linear path
nb_drones: 2

start_hub: start 0 0 [color=green]
hub: waypoint1 1 0 [color=blue]
hub: waypoint2 2 0 [color=blue]
end_hub: goal 3 0 [color=red]

connection: start-waypoint1
connection: waypoint1-waypoint2
connection: waypoint2-goal
```

Command:

```bash
python3 main.py maps_download/easy/01_linear_path.txt --plain
```

Expected output:

```
Turn 1: D1-waypoint1
Turn 2: D1-waypoint2 D2-waypoint1
Turn 3: D1-goal D2-waypoint2
Turn 4: D2-goal

Total turns: 4
```

### Performance results (all targets met)

| Map                       | Drones | Target | Result |
|---------------------------|--------|--------|--------|
| Linear path               | 2      | <=6    | 4      |
| Simple fork               | 4      | <=8    | 4      |
| Basic capacity            | 4      | <=6    | 4      |
| Dead end trap             | 5      | <=12   | 8      |
| Circular loop             | 6      | <=15   | 10     |
| Priority puzzle           | 5      | <=12   | 6      |
| Maze nightmare           | 8      | <=30   | 13     |
| Capacity hell             | 12     | <=35   | 16     |
| Ultimate challenge        | 15     | <=45   | 26     |
| The Impossible Dream      | 25     | <45    | 43     |

## Resources

- Dijkstra, E. W. (1959). *A note on two problems in connexion with graphs.*
  Numerische Mathematik.
- Sharon et al. (2015). *Conflict-based search for optimal multi-agent
  pathfinding.* Artificial Intelligence — reference for the
  reservation-table / prioritized-planning family of MAPF techniques.
- Stern et al. (2019). *Multi-Agent Pathfinding: Definitions, Variants, and
  Benchmarks.* SoCS — the time-expanded graph and vertex-capacity model used
  here follow the standard MAPF formulation.
- Python documentation: `heapq`, `argparse`, `enum`, `dataclasses`.

### How AI was used

AI assistance was used for:

- exploring the standard time-expanded MAPF formulation and selecting a
  defensible planning strategy (prioritized planning with a reservation
  table);
- drafting the parser's validation logic and the test matrix for parser edge
  cases;
- cross-checking the restricted-zone movement semantics against the
  specification wording;
- generating boilerplate (Makefile, `setup.cfg`, `.gitignore`) and the
  README structure.

All generated code was reviewed, understood, type-checked (`mypy --strict`),
linted (`flake8`) and verified by the test suite and the validator before being
kept. The author takes full responsibility for the final implementation and
can explain every module during the peer evaluation.
