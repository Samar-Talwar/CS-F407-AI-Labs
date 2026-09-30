# CS F407 Lab, Week 4 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Independent plan validator — does NOT share code with the BFS planner."""

from __future__ import annotations

from dataclasses import dataclass

from week04_logic.src.planner import Action, State


@dataclass
class ValidationResult:
    """Result of validating a plan."""

    valid: bool
    states: list[list[str]]           # S0, S1, …, Sn
    error: str | None = None          # description of first violation
    failed_step: int | None = None    # 0-indexed step that failed
    goal_reached: bool = False


def validate_plan(
    initial: State,
    goal: State,
    plan: list[Action],
) -> ValidationResult:
    """Replay *plan* from *initial*, checking preconditions at every step.

    This is an INDEPENDENT validator: it re-implements the precondition check
    and state update inline, without calling applicable() or apply_action().
    """
    states: list[list[str]] = [sorted(initial)]
    current = initial

    for i, action in enumerate(plan):
        # Independent precondition check
        missing_pos = action.pos_pre - current
        violated_neg = action.neg_pre & current
        if missing_pos or violated_neg:
            parts: list[str] = []
            if missing_pos:
                parts.append(
                     f"positive preconditions not in state: {sorted(missing_pos)}"
                 )
            if violated_neg:
                parts.append(
                     f"negative preconditions violated (present in state): {sorted(violated_neg)}"
                 )
            return ValidationResult(
                valid=False,
                states=states,
                error=f"Step {i} ({action.name}): {'; '.join(parts)}",
                failed_step=i,
            )
        # Independent state update: remove neg_eff, add pos_eff
        current = (current - action.neg_eff) | action.pos_eff
        states.append(sorted(current))

    goal_reached = goal <= current
    if not goal_reached:
        return ValidationResult(
            valid=False,
            states=states,
            error=(
    f"Plan completed but goal {sorted(goal)} not reached "
    f"in final state {sorted(current)}"
),
            goal_reached=False,
        )

    return ValidationResult(valid=True, states=states, goal_reached=True)
