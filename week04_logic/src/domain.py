# CS F407 Lab, Week 4 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Warehouse planning domain: initial state, goal, grounded actions."""

from __future__ import annotations

from week04_logic.src.planner import Action, State

LOCATIONS = ["A", "B", "C"]

# Connectivity: A-B and B-C (bidirectional), no direct A-C link.
CONNECTIONS = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]

INITIAL_STATE: State = frozenset({"At(Robot,A)", "At(Package,A)"})
GOAL: State = frozenset({"At(Package,C)"})


def warehouse_actions() -> list[Action]:
    """Return the full set of grounded warehouse actions."""
    actions: list[Action] = []

    # Move actions
    for src, dst in CONNECTIONS:
        actions.append(Action(
            name=f"Move({src},{dst})",
            pos_pre=frozenset({f"At(Robot,{src})"}),
            neg_pre=frozenset(),
            pos_eff=frozenset({f"At(Robot,{dst})"}),
            neg_eff=frozenset({f"At(Robot,{src})"}),
        ))

    # PickUp actions
    for loc in LOCATIONS:
        actions.append(Action(
            name=f"PickUp(Package,{loc})",
            pos_pre=frozenset({f"At(Robot,{loc})", f"At(Package,{loc})"}),
            neg_pre=frozenset(),
            pos_eff=frozenset({"Holding(Package)"}),
            neg_eff=frozenset({f"At(Package,{loc})"}),
        ))

    # Drop actions
    for loc in LOCATIONS:
        actions.append(Action(
            name=f"Drop(Package,{loc})",
            pos_pre=frozenset({f"At(Robot,{loc})", "Holding(Package)"}),
            neg_pre=frozenset(),
            pos_eff=frozenset({f"At(Package,{loc})"}),
            neg_eff=frozenset({"Holding(Package)"}),
        ))

    return actions


def warehouse_actions_no_pickup() -> list[Action]:
    """Warehouse actions with PickUp removed (Test B)."""
    return [a for a in warehouse_actions() if not a.name.startswith("PickUp")]


def warehouse_actions_with_irrelevant() -> list[Action]:
    """Warehouse actions plus irrelevant extra actions (Test C).

    Adds irrelevant actions such as moving to a side-room D or cleaning the floor,
    which expand the search space without helping to deliver the package.
    """
    extras = [
        Action(
            name="Move(A,D)",
            pos_pre=frozenset({"At(Robot,A)"}),
            neg_pre=frozenset(),
            pos_eff=frozenset({"At(Robot,D)"}),
            neg_eff=frozenset({"At(Robot,A)"}),
        ),
        Action(
            name="Move(D,A)",
            pos_pre=frozenset({"At(Robot,D)"}),
            neg_pre=frozenset(),
            pos_eff=frozenset({"At(Robot,A)"}),
            neg_eff=frozenset({"At(Robot,D)"}),
        ),
        Action(
            name="CleanFloor(A)",
            pos_pre=frozenset({"At(Robot,A)"}),
            neg_pre=frozenset(),
            pos_eff=frozenset({"Clean(A)"}),
            neg_eff=frozenset(),
        ),
    ]
    return warehouse_actions() + extras


# ---------- Synthetic problem exercising negative preconditions ----------

SYNTH_INITIAL: State = frozenset({"HasKey", "DoorLocked"})
SYNTH_GOAL: State = frozenset({"Inside"})

SYNTH_ACTIONS: list[Action] = [
    Action(
        name="UnlockDoor",
        pos_pre=frozenset({"HasKey", "DoorLocked"}),
        neg_pre=frozenset(),
        pos_eff=frozenset(),                    # just removes locked
        neg_eff=frozenset({"DoorLocked"}),       # negative effect: removes DoorLocked
    ),
    Action(
        name="Enter",
        pos_pre=frozenset(),
        neg_pre=frozenset({"DoorLocked"}),       # negative precondition: door must NOT be locked
        pos_eff=frozenset({"Inside"}),
        neg_eff=frozenset(),
    ),
]
