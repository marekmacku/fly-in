*This project has been created as part of the 42 curriculum by mmacku.*

# Fly-in

## Description

Fly-in is a drone routing simulation. Given a map describing a network of
connected zones, a fleet of drones must be routed from a central start zone
to a target end zone in the fewest possible simulation turns, while
respecting zone occupancy limits, connection capacities and per-zone-type
movement costs (normal and priority zones cost one turn, restricted zones
cost two turns, blocked zones cannot be entered).

The project is written entirely in Python (standard library only), is
fully object-oriented and fully type-checked (`mypy --strict` and `flake8`
pass with zero warnings). The active implementation is the `fly_in2`
package; it is a from-scratch rewrite of the original `fly_in` package,
which is kept in the repository as a reference. It contains:

- a strict map-file **parser** with precise error reporting (`line N: cause`
  on every rejection, integer validation included);
- a **graph** model with bidirectional connections, capacity metadata and
  `None` sentinels for unlimited capacity and unenterable zones;
- a **pathfinding** layer (Dijkstra) implemented from scratch, with no
  graph library, integer-only distances and a priority-zone tie-break;
- a **planner** that performs time-expanded prioritized planning with a
  reservation table, in a single ascending pass, preferring priority
  zones among equal-time routes;
- a **simulator** that replans nothing and decides nothing - it merges
  the planned trajectories into the required `D<ID>-<zone>` /
  `D<ID>-<connection>` output lines;
- a **validator** that independently replays every plan and re-checks
  all movement and capacity rules; it also runs as a gate before any
  output is printed (see below);
- a **visualizer** that renders a colored per-turn grid and colorizes
  the movement lines.

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
| `--gui`      | Animated pygame window (falls back to terminal output)    |
| `--plain`    | Clean movement lines only (no color, no grid)            |
| `--no-grid`  | Disable the per-turn grid, keep colored movement lines    |
| `--no-color` | Disable ANSI colors                                        |
| `--stats`    | Print secondary metrics (moves, average moves/drone)       |

### Exit codes and failure classes

The pipeline is parse -> plan -> playback -> **validate** -> render. The
validator runs before anything is printed, so an internal bug can never
show a broken simulation to the user. Failures are separated by cause:

| Code | Meaning                                                    |
|------|------------------------------------------------------------|
| 0    | Success                                                     |
| 2    | The map file is unreadable or invalid (`line N: cause`)     |
| 3    | The map is solvable-checked but has no route (unreachable)  |
| 4    | Internal error: the finished plan failed validation         |

### Linting and type checking

```bash
make lint            # flake8 + mypy (project-level flags)
make lint-strict     # flake8 + mypy --strict
```

### Tests

```bash
python3 -m pytest tests/ -q
```

The suite covers the parser with an adversarial map corpus
(`maps_custom/`: 9 nasty-but-legal maps that must parse, 22 traps that
must be rejected at the exact line with the right cause, and 10
challenge maps with hand-computed expected turn counts), unit tests
for pathfinding and the planner semantics, CLI exit codes including
the validation gate, and replays every provided and custom map through
the validator oracle.

### Known limitation (documented honestly)

Greedy prioritized planning can pile every drone onto one corridor
when a map offers two parallel routes of slightly different length:
each drone's individually fastest route is the same one, so the fleet
ignores the second corridor. The challenge map
`maps_custom/challenge/m08_big_world.txt` demonstrates this (31 turns
versus a theoretical ~22 with perfect route splitting). All provided
benchmark targets are met regardless; fixing this properly requires
route-diversity heuristics or an optimal MAPF solver such as
Conflict-Based Search.

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

`fly_in2/pathfinding.py` implements Dijkstra's algorithm from scratch (no
graph library, per the constraints). Edge weights are the movement cost
of the *destination* zone (normal/priority = 1, restricted = 2, blocked =
never entered - the model's `movement_cost` returns `None` for blocked).
Distances are exact integers; a zone missing from the result is simply
unreachable (no `inf` sentinels, no casts).

The spec's "priority zones should be prioritized" rule is honoured with a
**secondary tie-break key**: among equal-cost paths, the search prefers
the one that steps through fewer non-priority zones. The key is only a
heap tie-breaker - it never enters the distance, so reported costs stay
exact.

### Multi-drone planning

`fly_in2/planner.py` uses **time-expanded prioritized planning** with a
reservation table:

1. A static Dijkstra run decides solvability (unreachable end zone ->
   error) and provides the baseline cost for the search horizon.
2. Each drone is planned one at a time, in ascending id order, with
   Dijkstra over a **time-expanded graph** whose states are
   `(zone, ready_turn)` pairs. Three actions exist: WAIT
   (`(v,t) -> (v,t+1)`, occupies `v` at turn `t`), MOVE into a
   normal/priority neighbour (occupies the link at turn `t` and the
   destination from turn `t`; the source zone is freed that same turn,
   per the spec), and TRANSIT into a restricted neighbour
   (`(v,t) -> (u,t+2)`: on the link at turn `t`, forced arrival at
   turn `t+1`).
3. A **reservation table** counts occupants per `(zone, turn)` and per
   `(connection, turn)`; every action of the drone currently being
   planned is checked against the reservations of all previously
   planned drones. Start and end hubs have unlimited capacity, so a
   drone can always wait at the start until the network drains - a plan
   always exists on a reachable map.
4. Among states with the same ready turn, the search prefers routes
   entering fewer non-priority zones - the same tie-break as static
   pathfinding, applied where drones actually fly.

Only **one pass** is needed: all drones are interchangeable (same start
zone, same goal), so the planning order cannot change the outcome. This
was verified empirically against a multi-ordering variant, which produced
identical plans on every map; the single pass removes that dead code and
roughly 90% of planning runtime.

The search horizon is bounded (`MAX_STATES` zones-x-turns budget, plus a
formula above the provable worst-case plan length) so memory stays
bounded on maps of any size.

Complexity: O(n - V - T log(V - T)) per drone, where T is the horizon;
empirically the planner behaves near-linearly in the number of drones
(about 3.5 ms per drone on the challenger map, verified up to 400
drones), because early drones terminate early and reservations prune the
search space of later ones.

### Restricted-zone semantics (documented interpretation)

The spec is terse about the two-turn restricted move. The implementation
uses this model, which is internally consistent and produces both output
forms:

- Turn `t`: the drone leaves `v`, enters the connection and is **in
  transit**. Output: `D<id>-<src>-<dst>`.
- Turn `t+1`: the drone **arrives** at the restricted zone. Output:
  `D<id>-<zone>`.
- The drone may act again starting at turn `t+2`. Once transit begins,
  arrival is forced - the planner only starts a restricted transit when
  the destination is guaranteed to have capacity at the arrival turn.

### Validation

`fly_in2/validator.py` replays every plan turn by turn and independently
re-checks: start and delivery, no entry into blocked zones, valid
connections, the two-turn restricted pattern, zone and connection
capacity. Its occupancy bookkeeping deliberately **duplicates** the
planner's reservation logic instead of sharing it: an oracle that reuses
the code it checks cannot catch bugs in that code. Generation and
verification are separate by design.

### Runtime gate

The CLI runs the validator between playback and rendering. If the replay
finds any violation, the program exits with code 4, prints
`internal error: plan failed validation` (with turn-stamped violations)
to stderr, and produces **no simulation output** - nothing unverified
ever reaches the screen.

## Visual representation

Two visual modes are provided (the spec's "both options"):

- **`--gui`: an animated pygame window.** Zones are drawn as colored
  tiles at their map coordinates (colors from each zone's `color`
  metadata, with per-type fallbacks), connections as links, and drones
  as markers that move zone to zone each turn - drones in restricted
  transit are drawn midway over their connection. Controls: space
  pauses, right arrow steps one turn, q/escape quits. pygame is
  imported lazily; if it is not installed the program notes it on
  stderr and falls back to the terminal renderer, so the simulation
  never fails because of the visualization. Only existing zones are
  drawn, so absurd coordinate spans degrade gracefully instead of
  hanging.
- **Default: colored terminal output.** A per-turn grid drawn from
  zone `(x, y)` coordinates (zone symbols `S`/`E`/`P`/`R`/`#`/`O` plus
  occupying drones), ANSI colors from zone metadata with a fallback
  table, and colorized movement lines.

The `--plain` flag strips colors and the grid for a clean,
machine-parseable output (exactly the format required by the
specification). The movement lines remain the authoritative simulation
output in every mode.

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

- Dijkstra, E. W. (1959). *A note on two problems in connexion with
  graphs.* Numerische Mathematik.
- Sharon et al. (2015). *Conflict-based search for optimal multi-agent
  pathfinding.* Artificial Intelligence - reference for the
  reservation-table / prioritized-planning family of MAPF techniques.
- Stern et al. (2019). *Multi-Agent Pathfinding: Definitions, Variants,
  and Benchmarks.* SoCS - the time-expanded graph and vertex-capacity
  model used here follow the standard MAPF formulation.
- Python documentation: `heapq`, `argparse`, `enum`, `dataclasses`.

### How AI was used

AI assistance was used for:

- exploring the standard time-expanded MAPF formulation and selecting a
  defensible planning strategy (prioritized planning with a reservation
  table);
- drafting the parser's validation logic and generating an adversarial
  map corpus to attack it;
- cross-checking the restricted-zone movement semantics against the
  specification wording;
- rewriting the implementation from scratch (`fly_in2`) with AI as the
  author and the project owner reviewing, questioning and approving
  every module - each design decision (sentinels, single-pass planning,
  the runtime validation gate) was discussed and understood before
  being kept;
- boilerplate (Makefile, `setup.cfg`, `.gitignore`) and README structure.

All generated code was reviewed, understood, type-checked
(`mypy --strict`), linted (`flake8`) and verified by the test suite and
the validator oracle before being kept. The author takes full
responsibility for the final implementation and can explain every module
during the peer evaluation.
