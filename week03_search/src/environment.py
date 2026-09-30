# CS F407 Lab, Week 3 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Warehouse environment, ASCII grid parser, action modeling, and path validation."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import NamedTuple

GridState = tuple[int, int]  # (row, column)


class Action(Enum):
    """Discrete 4-connected movements with unit cost = 1."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def delta_row(self) -> int:
        return self.value[0]

    @property
    def delta_col(self) -> int:
        return self.value[1]


class MapComponents(NamedTuple):
    grid: tuple[str, ...]
    start: GridState
    goal: GridState
    rows: int
    cols: int


# The official 9x17 warehouse map from the laboratory sheet
SHEET_MAP_RAW: str = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

# Test 2: Trivial adjacent goal (3x5)
TEST2_TRIVIAL_RAW: str = """\
#####
#SG##
#####"""

# Test 3: Unreachable goal (5x7)
TEST3_NO_SOLUTION_RAW: str = """\
#######
#S....#
###.###
#...#G#
#######"""

# Test 4: Alternative paths with two equal-length shortest routes (length 12)
# and a longer decoy route (length 16)
TEST4_ALTERNATIVE_PATHS_RAW: str = """\
###########
#.........#
#.#######.#
#S#.....#G#
#.#######.#
#.........#
#.#######.#
#.........#
###########"""


def parse_map(raw_map: str | Sequence[str]) -> MapComponents:
    """Parse and strictly validate an ASCII grid map.

    Validates:
      - Non-empty rectangular grid (all rows equal length)
      - Only valid characters: '#', '.', 'S', 'G'
      - Exactly one start 'S' and exactly one goal 'G'
    """
    if isinstance(raw_map, str):
        lines = [line.strip() for line in raw_map.strip().splitlines() if line.strip()]
    else:
        lines = [line.strip() for line in raw_map if line.strip()]

    if not lines:
        raise ValueError("Grid map cannot be empty.")

    num_rows = len(lines)
    num_cols = len(lines[0])
    if num_cols == 0:
        raise ValueError("Grid rows cannot be empty.")

    valid_chars = {"#", ".", "S", "G"}
    starts: list[GridState] = []
    goals: list[GridState] = []

    for r, row in enumerate(lines):
        if len(row) != num_cols:
            raise ValueError(
                f"Grid is not rectangular: row {r} has length {len(row)}, expected {num_cols}."
            )
        for c, char in enumerate(row):
            if char not in valid_chars:
                raise ValueError(f"Invalid character '{char}' at ({r}, {c}).")
            if char == "S":
                starts.append((r, c))
            elif char == "G":
                goals.append((r, c))

    if len(starts) != 1:
        raise ValueError(f"Grid must have exactly 1 start 'S', found {len(starts)}.")
    if len(goals) != 1:
        raise ValueError(f"Grid must have exactly 1 goal 'G', found {len(goals)}.")

    return MapComponents(
        grid=tuple(lines),
        start=starts[0],
        goal=goals[0],
        rows=num_rows,
        cols=num_cols,
    )


class Environment:
    """Discrete 2D warehouse grid environment."""

    def __init__(self, raw_map: str | Sequence[str] = SHEET_MAP_RAW) -> None:
        parsed = parse_map(raw_map)
        self.grid: tuple[str, ...] = parsed.grid
        self.start: GridState = parsed.start
        self.goal: GridState = parsed.goal
        self.rows: int = parsed.rows
        self.cols: int = parsed.cols

    def is_in_bounds(self, coord: GridState) -> bool:
        """Check if coordinates fall within grid boundaries."""
        r, c = coord
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, coord: GridState) -> bool:
        """Check if cell is an obstacle '#' or out of bounds."""
        if not self.is_in_bounds(coord):
            return True
        r, c = coord
        return self.grid[r][c] == "#"

    def is_valid_action(self, coord: GridState, action: Action) -> bool:
        """Check if executing action from coord lands on a free/start/goal cell."""
        next_coord = (coord[0] + action.delta_row, coord[1] + action.delta_col)
        return not self.is_wall(next_coord)

    def step(self, coord: GridState, action: Action) -> GridState:
        """Execute deterministic transition T(s, a).

        Raises ValueError if action is invalid (hits a wall or out of bounds).
        """
        if not self.is_valid_action(coord, action):
            raise ValueError(f"Invalid action {action.name} from state {coord}.")
        return (coord[0] + action.delta_row, coord[1] + action.delta_col)

    def get_valid_actions(self, coord: GridState) -> list[Action]:
        """Return all valid actions from given coordinate."""
        return [action for action in Action if self.is_valid_action(coord, action)]

    def get_neighbors(self, coord: GridState) -> list[tuple[Action, GridState]]:
        """Return list of (action, successor_state) pairs."""
        results: list[tuple[Action, GridState]] = []
        for action in Action:
            if self.is_valid_action(coord, action):
                results.append((action, (coord[0] + action.delta_row, coord[1] + action.delta_col)))
        return results

    def is_goal(self, coord: GridState) -> bool:
        """Goal test: checks if state matches goal coordinates."""
        return coord == self.goal

    def validate_path(self, path: Sequence[GridState]) -> bool:
        """Verify that a path is unbroken, wall-free, starts at S and ends at G."""
        if not path:
            return False
        if path[0] != self.start:
            return False
        if path[-1] != self.goal:
            return False

        for i in range(len(path)):
            curr = path[i]
            if self.is_wall(curr):
                return False
            if i > 0:
                prev = path[i - 1]
                dr = abs(curr[0] - prev[0])
                dc = abs(curr[1] - prev[1])
                # Must be 1-step orthogonal move
                if (dr + dc) != 1:
                    return False
        return True

    def render_with_path(self, path: Sequence[GridState]) -> str:
        """Render the grid with path marked using '*' characters."""
        path_set = set(path)
        rendered_lines: list[str] = []
        for r in range(self.rows):
            row_chars: list[str] = []
            for c in range(self.cols):
                coord = (r, c)
                if coord == self.start:
                    row_chars.append("S")
                elif coord == self.goal:
                    row_chars.append("G")
                elif coord in path_set:
                    row_chars.append("*")
                else:
                    row_chars.append(self.grid[r][c])
            rendered_lines.append("".join(row_chars))
        return "\n".join(rendered_lines)
