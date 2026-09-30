# CS F407 Lab, Week 2 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Explicit goal-based agent architecture with search planner."""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from week02_agents.src.environment import Environment

# --- Actions (Up/Down/Left/Right with validity check) ---


class Action(Enum):
    """Discrete actions available to the warehouse navigation agent."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def delta(self) -> tuple[int, int]:
        """Return (row_delta, col_delta) for this action."""
        return self.value

    @property
    def name_str(self) -> str:
        """Return human-readable action name."""
        return self.name.capitalize()


ACTION_NAMES: dict[int, str] = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
}

ACTION_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1),   # Right
)


def apply_action(
    pos: tuple[int, int],
    action: Action | int | tuple[int, int],
    env: Environment,
) -> tuple[int, int] | None:
    """Apply a movement action, returning the new position if valid, else None.

    Args:
        pos: Current (row, col) coordinate.
        action: Action enum, action index, or delta tuple.
        env: Environment to check validity against.

    Returns:
        New (row, col) position if valid move, else None.
    """
    r, c = pos
    if isinstance(action, Action):
        dr, dc = action.delta
    elif isinstance(action, int):
        dr, dc = ACTION_DELTAS[action]
    elif isinstance(action, tuple) and len(action) == 2:
        dr, dc = action
    else:
        raise ValueError(f"Invalid action representation: {action}")

    new_pos = (r + dr, c + dc)
    if env.is_valid_move(new_pos):
        return new_pos
    return None


# --- Agent Architecture Components (visible in code and docstrings) ---


class AgentState:
    """State maintained by the agent to choose its next action.

    A goal-based agent maintains an explicit objective and selects actions
    that move it towards that objective — here via a search planner.
    """

    def __init__(
        self,
        position: tuple[int, int],
        environment: Environment,
        goal: tuple[int, int],
    ) -> None:
        """Initialize the agent internal state representation."""
        self.position = position
        self.environment = environment
        self.goal = goal


class GoalTest:
    """Goal-test component: checks whether current state achieves the objective."""

    def __init__(self, environment: Environment) -> None:
        """Initialize with target environment."""
        self.environment = environment

    def __call__(self, position: tuple[int, int]) -> bool:
        """Evaluate if given position satisfies the goal condition."""
        return position == self.environment.goal


# --- Search Planner (Decision Component) ---


@dataclass
class SearchResult:
    """Result of a search planner invocation."""

    found: bool
    path: list[tuple[int, int]]
    path_length: int
    expanded_nodes: int
    runtime: float
    algorithm: str


def _reconstruct_path(
    parents: dict[tuple[int, int], tuple[int, int] | None],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Reconstruct path from start to goal using parent pointers."""
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parents.get(cur)
    path.reverse()
    return path


def bfs_search(
    env: Environment,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> SearchResult:
    """BFS: optimal for unit-cost grid, guarantees shortest path.

    Uses a FIFO queue, closed set (visited), and parent pointers.
    """
    t0 = time.perf_counter()
    if start == goal:
        return SearchResult(
            found=True,
            path=[start],
            path_length=0,
            expanded_nodes=0,
            runtime=time.perf_counter() - t0,
            algorithm="BFS",
        )

    visited: set[tuple[int, int]] = {start}
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    expanded = 0

    while queue:
        current = queue.popleft()
        expanded += 1
        if current == goal:
            path = _reconstruct_path(parents, goal)
            return SearchResult(
                found=True,
                path=path,
                path_length=len(path) - 1,
                expanded_nodes=expanded,
                runtime=time.perf_counter() - t0,
                algorithm="BFS",
            )
        r, c = current
        for dr, dc in ACTION_DELTAS:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if env.is_valid_move(nxt) and nxt not in visited:
                visited.add(nxt)
                parents[nxt] = current
                queue.append(nxt)

    return SearchResult(
        found=False,
        path=[],
        path_length=-1,
        expanded_nodes=expanded,
        runtime=time.perf_counter() - t0,
        algorithm="BFS",
    )


def dfs_search(
    env: Environment,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> SearchResult:
    """DFS comparison baseline; not guaranteed optimal."""
    t0 = time.perf_counter()
    if start == goal:
        return SearchResult(
            found=True,
            path=[start],
            path_length=0,
            expanded_nodes=0,
            runtime=time.perf_counter() - t0,
            algorithm="DFS",
        )

    visited: set[tuple[int, int]] = set()
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    stack: list[tuple[int, int]] = [start]
    expanded = 0

    while stack:
        current = stack.pop()
        expanded += 1
        if current == goal:
            path = _reconstruct_path(parents, goal)
            return SearchResult(
                found=True,
                path=path,
                path_length=len(path) - 1,
                expanded_nodes=expanded,
                runtime=time.perf_counter() - t0,
                algorithm="DFS",
            )
        if current in visited:
            continue
        visited.add(current)
        r, c = current
        # Reverse order so first action explored is Up (consistent orientation)
        for dr, dc in reversed(ACTION_DELTAS):
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if env.is_valid_move(nxt) and nxt not in visited:
                parents[nxt] = current
                stack.append(nxt)

    return SearchResult(
        found=False,
        path=[],
        path_length=-1,
        expanded_nodes=expanded,
        runtime=time.perf_counter() - t0,
        algorithm="DFS",
    )


def astar_search(
    env: Environment,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> SearchResult:
    """A* search with Manhattan distance heuristic; optimal baseline for scaling."""
    t0 = time.perf_counter()
    if start == goal:
        return SearchResult(
            found=True,
            path=[start],
            path_length=0,
            expanded_nodes=0,
            runtime=time.perf_counter() - t0,
            algorithm="A*",
        )

    def manhattan(p: tuple[int, int]) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_heap: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (manhattan(start), 0, start))
    g_score: dict[tuple[int, int], int] = {start: 0}
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    expanded = 0
    visited: set[tuple[int, int]] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        expanded += 1
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = _reconstruct_path(parents, goal)
            return SearchResult(
                found=True,
                path=path,
                path_length=len(path) - 1,
                expanded_nodes=expanded,
                runtime=time.perf_counter() - t0,
                algorithm="A*",
            )
        for dr, dc in ACTION_DELTAS:
            nr, nc = current[0] + dr, current[1] + dc
            nxt = (nr, nc)
            if env.is_valid_move(nxt):
                tentative_g = g_score[current] + 1
                if nxt not in g_score or tentative_g < g_score[nxt]:
                    g_score[nxt] = tentative_g
                    parents[nxt] = current
                    f_score = tentative_g + manhattan(nxt)
                    heapq.heappush(open_heap, (f_score, tentative_g, nxt))

    return SearchResult(
        found=False,
        path=[],
        path_length=-1,
        expanded_nodes=expanded,
        runtime=time.perf_counter() - t0,
        algorithm="A*",
    )


class SearchPlanner:
    """Decision-making component that selects actions via search algorithm."""

    def __init__(self, algorithm: str = "BFS") -> None:
        """Initialize search planner with chosen algorithm."""
        self.algorithm = algorithm

    def plan(
        self,
        env: Environment,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> SearchResult:
        """Execute search planning from start to goal."""
        if self.algorithm == "BFS":
            return bfs_search(env, start, goal)
        if self.algorithm == "DFS":
            return dfs_search(env, start, goal)
        if self.algorithm == "A*":
            return astar_search(env, start, goal)
        raise ValueError(f"Unknown algorithm: {self.algorithm}")


class GoalBasedAgent:
    """Goal-based intelligent agent for warehouse navigation.

    Components (visible in docstrings and structure):
      - Environment: discrete grid with walls (shelving units).
      - State (AgentState): current position + objective.
      - Goal (GoalTest): check if position == G.
      - Actions (Up/Down/Left/Right): validated by wall check.
      - Decision (SearchPlanner): BFS planner that selects path to goal.
    """

    def __init__(self, environment: Environment) -> None:
        """Initialize goal-based agent with its environment and components."""
        self.environment = environment
        self.goal = environment.goal
        self.state = AgentState(
            position=environment.start,
            environment=environment,
            goal=environment.goal,
        )
        self.goal_test = GoalTest(environment)
        self.planner = SearchPlanner(algorithm="BFS")

    def decide(self) -> SearchResult:
        """Agent decision component: runs planner and returns route."""
        return self.planner.plan(
            self.environment,
            self.environment.start,
            self.environment.goal,
        )

    def explain_algorithm(self) -> str:
        """Explain why BFS was selected as the search planner algorithm."""
        return (
            "Breadth-First Search (BFS) was selected because each movement step "
            "has equal unit cost (1). BFS expands nodes in order of path cost, "
            "guaranteeing the shortest collision-free path to the goal while "
            "maintaining completeness on finite grid graphs."
        )
