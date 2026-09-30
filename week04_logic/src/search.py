# CS F407 Lab, Week 4 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""BFS planner: finds shortest plan from initial state to goal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from week04_logic.src.planner import Action, State, applicable, apply_action


@dataclass
class PlanResult:
    """Result of a planning search."""

    found: bool
    plan: list[str]
    states: list[list[str]]  # state after each action (sorted for readability)
    expanded: int


def bfs_plan(
    initial: State,
    goal: State,
    actions: list[Action],
) -> PlanResult:
    """Breadth-first search planner.

    Returns the shortest plan (fewest actions) achieving *goal* from *initial*.
    Uses a visited set of frozenset states to avoid re-expansion.
    """
    if goal <= initial:
        return PlanResult(True, [], [sorted(initial)], 0)

    queue: deque[tuple[State, list[Action]]] = deque([(initial, [])])
    visited: set[State] = {initial}
    expanded = 0

    while queue:
        state, path = queue.popleft()
        expanded += 1

        for action in actions:
            if not applicable(state, action):
                continue
            new_state = apply_action(state, action)
            if new_state in visited:
                continue
            visited.add(new_state)
            new_path = path + [action]
            if goal <= new_state:
                states = _trace_states(initial, new_path)
                return PlanResult(
                    found=True,
                    plan=[a.name for a in new_path],
                    states=states,
                    expanded=expanded,
                )
            queue.append((new_state, new_path))

    return PlanResult(found=False, plan=[], states=[], expanded=expanded)


def _trace_states(initial: State, actions: list[Action]) -> list[list[str]]:
    """Replay actions from initial, returning sorted state after each step."""
    states = [sorted(initial)]
    s = initial
    for a in actions:
        s = apply_action(s, a)
        states.append(sorted(s))
    return states
