# CS F407 Lab, Week 2 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Warehouse Environment representation, map parsing, and validation."""

from __future__ import annotations

SHEET_MAP_STRING: str = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################\
"""

VALID_SYMBOLS: set[str] = {"#", ".", "S", "G"}


class Environment:
    """Represents the 2D grid warehouse environment.

    Identifies the discrete spatial layout, obstacle coordinates (shelving units),
    starting position (loading bay 'S'), and goal position (dispatch area 'G').
    """

    def __init__(
        self,
        grid: tuple[str, ...],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> None:
        """Initialize the warehouse environment.

        Args:
            grid: Tuple of row strings representing the map.
            start: Coordinate (row, col) of the loading bay 'S'.
            goal: Coordinate (row, col) of the dispatch area 'G'.
        """
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        self.start = start
        self.goal = goal

    def is_within_bounds(self, pos: tuple[int, int]) -> bool:
        """Check whether a position lies within grid boundaries.

        Args:
            pos: (row, col) coordinate.

        Returns:
            True if position is within grid boundaries, False otherwise.
        """
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_wall(self, pos: tuple[int, int]) -> bool:
        """Check whether a position contains a wall/obstacle.

        Args:
            pos: (row, col) coordinate.

        Returns:
            True if position is an obstacle ('#') or out of bounds.
        """
        if not self.is_within_bounds(pos):
            return True
        r, c = pos
        return self.grid[r][c] == "#"

    def is_valid_move(self, pos: tuple[int, int]) -> bool:
        """Check whether the agent can legitimately enter the given position.

        Args:
            pos: (row, col) coordinate.

        Returns:
            True if the cell is within bounds and not a wall.
        """
        return self.is_within_bounds(pos) and not self.is_wall(pos)

    def render(
        self,
        path: list[tuple[int, int]] | None = None,
        path_char: str = "*",
    ) -> str:
        """Render the map as a string, optionally overlaying a path.

        Args:
            path: Sequence of (row, col) positions forming the route.
            path_char: Symbol used to indicate path steps (default '*').

        Returns:
            Multi-line string representation of the grid.
        """
        path_set = set(path) if path else set()
        rendered_rows: list[str] = []

        for r in range(self.height):
            row_chars: list[str] = []
            for c in range(self.width):
                pos = (r, c)
                if pos == self.start:
                    row_chars.append("S")
                elif pos == self.goal:
                    row_chars.append("G")
                elif pos in path_set:
                    row_chars.append(path_char)
                else:
                    row_chars.append(self.grid[r][c])
            rendered_rows.append("".join(row_chars))

        return "\n".join(rendered_rows)


def parse_map(raw_map: str | list[str]) -> Environment:
    """Parse and strictly validate a warehouse map string or list of lines.

    Validation rules:
    1. The map must not be empty.
    2. The map must be rectangular (all lines have identical non-zero length).
    3. The map must contain exactly one start position 'S'.
    4. The map must contain exactly one goal position 'G'.
    5. Only the symbols '#', '.', 'S', 'G' are permitted.

    Args:
        raw_map: Map string or list of line strings.

    Returns:
        Validated Environment instance.

    Raises:
        ValueError: If any validation rule is violated.
    """
    if isinstance(raw_map, str):
        lines = [line.strip() for line in raw_map.strip().splitlines() if line.strip()]
    elif isinstance(raw_map, list):
        lines = [line.strip() for line in raw_map if line.strip()]
    else:
        raise ValueError(f"Expected str or list of str, got {type(raw_map).__name__}")

    if not lines:
        raise ValueError("Map cannot be empty.")

    width = len(lines[0])
    if width == 0:
        raise ValueError("Map row width cannot be zero.")

    start_positions: list[tuple[int, int]] = []
    goal_positions: list[tuple[int, int]] = []

    for r, line in enumerate(lines):
        if len(line) != width:
            raise ValueError(
                f"Ragged map detected: row {r} has width {len(line)}, expected {width}."
            )
        for c, char in enumerate(line):
            if char not in VALID_SYMBOLS:
                raise ValueError(
                    f"Invalid character '{char}' at ({r}, {c}). Allowed: {VALID_SYMBOLS}."
                )
            if char == "S":
                start_positions.append((r, c))
            elif char == "G":
                goal_positions.append((r, c))

    if len(start_positions) != 1:
        raise ValueError(
            f"Map must contain exactly one start 'S', found {len(start_positions)}."
        )

    if len(goal_positions) != 1:
        raise ValueError(
            f"Map must contain exactly one goal 'G', found {len(goal_positions)}."
        )

    return Environment(
        grid=tuple(lines),
        start=start_positions[0],
        goal=goal_positions[0],
    )
