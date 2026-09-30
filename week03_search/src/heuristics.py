# CS F407 Lab, Week 3 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Heuristic functions for 4-connected grid navigation."""

from __future__ import annotations

import math
from collections.abc import Callable

from week03_search.src.environment import GridState

HeuristicFn = Callable[[GridState, GridState], float]


def manhattan_distance(a: GridState, b: GridState) -> float:
    """Compute Manhattan (taxicab) distance between two grid coordinates.

    h(n) = |r - r_G| + |c - c_G|
    Admissible and consistent on 4-connected unit-cost grids.
    """
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def euclidean_distance(a: GridState, b: GridState) -> float:
    """Compute Euclidean straight-line distance between two grid coordinates.

    h(n) = sqrt((r - r_G)^2 + (c - c_G)^2)
    Admissible on 4-connected grids since Euclidean <= Manhattan <= h*(n).
    """
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def zero_heuristic(a: GridState, b: GridState) -> float:
    """Return 0 for all states (blind search / Dijkstra / UCS).

    h(n) = 0 is trivially admissible and consistent.
    """
    return 0.0


def scaled_manhattan(k: float) -> HeuristicFn:
    """Return a scaled Manhattan heuristic h(n) = k * Manhattan(n, goal).

    For k > 1.0, this heuristic is inadmissible (can overestimate h*(n))
    and acts as weighted / aggressive A* search.
    """

    def heuristic(a: GridState, b: GridState) -> float:
        return float(k * (abs(a[0] - b[0]) + abs(a[1] - b[1])))

    return heuristic
