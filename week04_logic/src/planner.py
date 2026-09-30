# CS F407 Lab, Week 4 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Core planning primitives: State, Action, applicable(), apply_action()."""

from __future__ import annotations

from dataclasses import dataclass

# A state is a frozenset of ground proposition strings, e.g. "At(Robot,A)".
State = frozenset[str]


@dataclass(frozen=True)
class Action:
    """A STRIPS-style action with positive/negative preconditions and effects."""

    name: str
    pos_pre: frozenset[str]   # must be IN state
    neg_pre: frozenset[str]   # must NOT be in state
    pos_eff: frozenset[str]   # added after application
    neg_eff: frozenset[str]   # removed after application


def applicable(state: State, action: Action) -> bool:
    """Check if *action* is applicable in *state*.

    S |= Preconditions(a) iff pos_pre ⊆ S and neg_pre ∩ S = ∅.
    """
    return action.pos_pre <= state and action.neg_pre.isdisjoint(state)


def missing_preconditions(state: State, action: Action) -> dict[str, list[str]]:
    """Return the preconditions that are NOT satisfied, grouped by kind."""
    missing: dict[str, list[str]] = {}
    absent_pos = sorted(action.pos_pre - state)
    if absent_pos:
        missing["positive_missing"] = absent_pos
    present_neg = sorted(action.neg_pre & state)
    if present_neg:
        missing["negative_violated"] = present_neg
    return missing


def apply_action(state: State, action: Action) -> State:
    """Apply *action* to *state*: remove neg_eff, then add pos_eff."""
    return (state - action.neg_eff) | action.pos_eff
