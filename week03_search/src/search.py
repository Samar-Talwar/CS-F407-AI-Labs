# CS F407 Lab, Week 3 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Search algorithms (A* and BFS) with path reconstruction and metric tracking."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from week03_search.src.environment import Environment, GridState
from week03_search.src.heuristics import HeuristicFn, manhattan_distance


@dataclass(frozen=True)
class SearchResult:
    """Reported metrics and reconstructed solution path from a search algorithm."""

    found: bool
    path: list[GridState]
    path_length: int
    cost: float
    states_expanded: int
    algorithm: str
    heuristic: str | None
    expanded_order: list[GridState]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return asdict(self)


def reconstruct_path(
    parents: dict[GridState, GridState | None],
    goal: GridState,
) -> list[GridState]:
    """Reconstruct solution path from start to goal via parent pointers."""
    path: list[GridState] = []
    curr: GridState | None = goal
    while curr is not None:
        path.append(curr)
        curr = parents.get(curr)
    path.reverse()
    return path


def astar_search(
    env: Environment,
    heuristic: HeuristicFn = manhattan_distance,
    heuristic_name: str = "manhattan",
) -> SearchResult:
    """Execute A* search using standard library heapq.

    Evaluation function: f(n) = g(n) + h(n)
    - Frontier: min-heap storing tuples (f_cost, counter, state)
    - Tie-breaking: incremental integer counter ensures deterministic FIFO ordering on ties
    - Closed set: tracks expanded states to avoid duplicate expansions
    - Goal test: evaluated on state expansion to guarantee optimality with admissible heuristics
    """
    counter = 0
    start_state = env.start
    goal_state = env.goal

    g_costs: dict[GridState, float] = {start_state: 0.0}
    parents: dict[GridState, GridState | None] = {start_state: None}
    closed_set: set[GridState] = set()
    expanded_order: list[GridState] = []

    # Calculate initial f(s0) = g(s0) + h(s0) = 0 + h(s0)
    initial_h = heuristic(start_state, goal_state)
    initial_f = 0.0 + initial_h

    # Frontier elements: (f_score, tie_breaker_counter, state)
    frontier: list[tuple[float, int, GridState]] = [(initial_f, counter, start_state)]

    while frontier:
        f_cost, _, current_state = heapq.heappop(frontier)

        # Skip states already expanded (lazy deletion for duplicate entries)
        if current_state in closed_set:
            continue

        closed_set.add(current_state)
        expanded_order.append(current_state)

        # Goal test on expansion (crucial for optimality)
        if env.is_goal(current_state):
            path = reconstruct_path(parents, current_state)
            path_len = len(path) - 1
            return SearchResult(
                found=True,
                path=path,
                path_length=path_len,
                cost=float(path_len),
                states_expanded=len(closed_set),
                algorithm="astar",
                heuristic=heuristic_name,
                expanded_order=expanded_order,
            )

        current_g = g_costs[current_state]

        # Expand valid orthogonal successor states
        for _action, neighbor in env.get_neighbors(current_state):
            tentative_g = current_g + 1.0  # Unit step cost c(n, n') = 1.0

            if neighbor in closed_set:
                continue

            if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g
                parents[neighbor] = current_state
                h_cost = heuristic(neighbor, goal_state)
                f_cost = tentative_g + h_cost
                counter += 1
                heapq.heappush(frontier, (f_cost, counter, neighbor))

    # Goal unreachable
    return SearchResult(
        found=False,
        path=[],
        path_length=0,
        cost=0.0,
        states_expanded=len(closed_set),
        algorithm="astar",
        heuristic=heuristic_name,
        expanded_order=expanded_order,
    )


def bfs_search(env: Environment) -> SearchResult:
    """Execute Breadth-First Search (BFS) using collections.deque."""
    start_state = env.start

    parents: dict[GridState, GridState | None] = {start_state: None}
    closed_set: set[GridState] = set()
    expanded_order: list[GridState] = []
    queue: deque[GridState] = deque([start_state])

    while queue:
        current_state = queue.popleft()

        if current_state in closed_set:
            continue

        closed_set.add(current_state)
        expanded_order.append(current_state)

        if env.is_goal(current_state):
            path = reconstruct_path(parents, current_state)
            path_len = len(path) - 1
            return SearchResult(
                found=True,
                path=path,
                path_length=path_len,
                cost=float(path_len),
                states_expanded=len(closed_set),
                algorithm="bfs",
                heuristic=None,
                expanded_order=expanded_order,
            )

        for _action, neighbor in env.get_neighbors(current_state):
            if neighbor not in closed_set and neighbor not in parents:
                parents[neighbor] = current_state
                queue.append(neighbor)

    return SearchResult(
        found=False,
        path=[],
        path_length=0,
        cost=0.0,
        states_expanded=len(closed_set),
        algorithm="bfs",
        heuristic=None,
        expanded_order=expanded_order,
    )
