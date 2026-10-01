# CS F407 Lab, Week 4 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Command-line interface to reproduce all results and print the plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from week04_logic.src import (
    domain,
    prolog_runner,
    search,
    validator,
)
from week04_logic.src.planner import (
    Action,
    applicable,
    apply_action,
    missing_preconditions,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def task0_applicability() -> dict:
    """Task 0: record which actions are applicable in the initial state."""
    initial_state = domain.INITIAL_STATE
    goal = domain.GOAL
    actions = domain.warehouse_actions()

    pickup = next(a for a in actions if a.name == "PickUp(Package,A)")
    drop = next(a for a in actions if a.name == "Drop(Package,C)")

    return {
        "initial_state": sorted(initial_state),
        "goal": sorted(goal),
        "pickup_applicable": applicable(initial_state, pickup),
        "drop_applicable": applicable(initial_state, drop),
        "pickup_missing_preconditions": missing_preconditions(initial_state, pickup),
        "drop_missing_preconditions": missing_preconditions(initial_state, drop),
    }


def manual_plan() -> dict:
    """Task 1: manually constructed plan with state after each action."""
    # The correct shortest plan: PickUp(A), Move(A,B), Move(B,C), Drop(C)
    plan_names = ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]
    actions_by_name = {a.name: a for a in domain.warehouse_actions()}
    plan = [actions_by_name[n] for n in plan_names]
    states = [sorted(domain.INITIAL_STATE)]
    s = domain.INITIAL_STATE
    for a in plan:
        s = apply_action(s, a)
        states.append(sorted(s))
    return {
        "plan": plan_names,
        "states": states,
    }


def run_test(label: str, actions_func) -> dict:
    """Run a planning test and record the five required fields."""
    initial_state = domain.INITIAL_STATE
    goal = domain.GOAL
    actions = actions_func()
    result = search.bfs_plan(initial_state, goal, actions)

    plan_actions = []
    if result.plan:
        action_by_name = {a.name: a for a in actions}
        for name in result.plan:
            a = action_by_name[name]
            plan_actions.append(Action(
                name=a.name,
                pos_pre=a.pos_pre,
                neg_pre=a.neg_pre,
                pos_eff=a.pos_eff,
                neg_eff=a.neg_eff,
            ))

    validation = validator.validate_plan(initial_state, goal, plan_actions)

    return {
        "initial_state": sorted(initial_state),
        "goal": sorted(goal),
        "plan_found": result.found,
        "plan": result.plan,
        "states_expanded": result.expanded,
        "valid": validation.valid,
        "validation_error": validation.error,
        "goal_reached": validation.goal_reached,
    }


def test_a() -> dict:
    """Test A: original warehouse problem."""
    return run_test("A", domain.warehouse_actions)


def test_b() -> dict:
    """Test B: impossible (no PickUp)."""
    return run_test("B", domain.warehouse_actions_no_pickup)


def test_c() -> dict:
    """Test C: irrelevant actions."""
    return run_test("C", domain.warehouse_actions_with_irrelevant)


def invalid_example_plan() -> dict:
    """Task 1 trap: validate the sheet's example sequence (should fail)."""
    sheet_plan_names = ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)", "Drop(Package,C)"]
    actions_by_name = {a.name: a for a in domain.warehouse_actions()}
    sheet_plan = [actions_by_name[n] for n in sheet_plan_names]
    validation = validator.validate_plan(
        domain.INITIAL_STATE, domain.GOAL, sheet_plan
    )
    return {
        "plan": sheet_plan_names,
        "valid": validation.valid,
        "error": validation.error,
        "failed_step": validation.failed_step,
        "states": validation.states,
    }


def negative_preconditions() -> dict:
    """Exercise negative preconditions and effects with a synthetic problem."""
    initial_state = domain.SYNTH_INITIAL
    goal = domain.SYNTH_GOAL
    actions = domain.SYNTH_ACTIONS
    result = search.bfs_plan(initial_state, goal, actions)

    plan_actions = []
    if result.plan:
        action_by_name = {a.name: a for a in actions}
        for name in result.plan:
            a = action_by_name[name]
            plan_actions.append(Action(
                name=a.name,
                pos_pre=a.pos_pre,
                neg_pre=a.neg_pre,
                pos_eff=a.pos_eff,
                neg_eff=a.neg_eff,
            ))

    validation = validator.validate_plan(initial_state, goal, plan_actions)

    return {
        "initial_state": sorted(initial_state),
        "goal": sorted(goal),
        "plan_found": result.found,
        "plan": result.plan,
        "states": result.states,
        "states_expanded": result.expanded,
        "valid": validation.valid,
        "validation_error": validation.error,
    }


def prolog_results(plan: list[str]) -> dict:
    """Run Prolog queries and write week04_logic/results/prolog_results.json."""
    results_dir = RESULTS_DIR
    return prolog_runner.write_prolog_results(results_dir, plan)


def regenerate_all() -> None:
    """Re-generate every result file into week04_logic/results/ and print summaries."""
    results_dir = RESULTS_DIR
    _ensure_dir(results_dir)

    # Task 0
    task0 = task0_applicability()
    (results_dir / "task0_applicability.json").write_text(
        json.dumps(task0, indent=2) + "\n", encoding="utf-8"
    )

    # Task 1 manual plan
    manual = manual_plan()
    (results_dir / "manual_plan.json").write_text(
        json.dumps(manual, indent=2) + "\n", encoding="utf-8"
    )

    # Test A, B, C
    for label, func in [("test_a", test_a), ("test_b", test_b), ("test_c", test_c)]:
        data = func()
        (results_dir / f"{label}.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )

    # Invalid example plan
    invalid = invalid_example_plan()
    (results_dir / "invalid_example_plan.json").write_text(
        json.dumps(invalid, indent=2) + "\n", encoding="utf-8"
    )

    # Negative preconditions
    neg = negative_preconditions()
    (results_dir / "negative_preconditions.json").write_text(
        json.dumps(neg, indent=2) + "\n", encoding="utf-8"
    )

    # Prolog results (after we have the plan from test_a)
    plan_a = test_a().get("plan", [])
    prolog_results(plan_a)


def print_summary() -> None:
    """Print human-readable summaries matching the lab sheet expectations."""
    results_dir = RESULTS_DIR
    print("\n=== TASK 0: APPLICABILITY IN INITIAL STATE ===")
    t0 = json.loads((results_dir / "task0_applicability.json").read_text())
    print(f"Initial state: {{{', '.join(t0['initial_state'])}}}")
    print(f"Goal: {{{', '.join(t0['goal'])}}}")
    print(f"PickUp(Package,A) applicable: {t0['pickup_applicable']}")
    if not t0["pickup_applicable"]:
        print(f"  Missing preconditions: {t0['pickup_missing_preconditions']}")
    print(f"Drop(Package,C) applicable: {t0['drop_applicable']}")
    if not t0["drop_applicable"]:
        print(f"  Missing preconditions: {t0['drop_missing_preconditions']}")

    print("\n=== TASK 1: MANUAL PLAN ===")
    manual = json.loads((results_dir / "manual_plan.json").read_text())
    print("Plan:", " -> ".join(manual["plan"]))
    for i, state in enumerate(manual["states"]):
        print(f"S{i}: {{{', '.join(state)}}}")

    print("\n=== TEST A (SOLVABLE) ===")
    ta = json.loads((results_dir / "test_a.json").read_text())
    print(f"Plan found: {ta['plan_found']}")
    print(f"Plan: {ta['plan']}")
    print(f"States expanded: {ta['states_expanded']}")
    print(f"Valid plan: {ta['valid']}")
    if not ta["valid"]:
        print(f"  Validation error: {ta['validation_error']}")

    print("\n=== TEST B (NO PICKUP) ===")
    tb = json.loads((results_dir / "test_b.json").read_text())
    print(f"Plan found: {tb['plan_found']}")
    print(f"Plan: {tb['plan']}")
    print(f"States expanded: {tb['states_expanded']}")
    print(f"Valid plan: {tb['valid']}")

    print("\n=== TEST C (IRRELEVANT ACTIONS) ===")
    tc = json.loads((results_dir / "test_c.json").read_text())
    print(f"Plan found: {tc['plan_found']}")
    print(f"Plan: {tc['plan']}")
    print(f"States expanded: {tc['states_expanded']}")
    print(f"Valid plan: {tc['valid']}")
    if tc["plan_found"] and tc["plan"]:
        print("Note: robot reaches C but package stays at A unless picked up.")

    print("\n=== INVALID EXAMPLE PLAN (FROM SHEET) ===")
    ie = json.loads((results_dir / "invalid_example_plan.json").read_text())
    print(f"Plan: {ie['plan']}")
    print(f"Valid: {ie['valid']}")
    if not ie["valid"]:
        print(f"Failed at step {ie['failed_step']}: {ie['error']}")
        print("States:")
        for i, state in enumerate(ie["states"]):
            print(f"  S{i}: {{{', '.join(state)}}}")

    print("\n=== NEGATIVE PRECONDITIONS SYNTHETIC PROBLEM ===")
    np = json.loads((results_dir / "negative_preconditions.json").read_text())
    print(f"Initial: {{{', '.join(np['initial_state'])}}}")
    print(f"Goal: {{{', '.join(np['goal'])}}}")
    print(f"Plan found: {np['plan_found']}")
    print(f"Plan: {np['plan']}")
    print(f"States expanded: {np['states_expanded']}")
    print(f"Valid: {np['valid']}")
    if not np["valid"]:
        print(f"  Error: {np['validation_error']}")

    print("\n=== PROLOG RESULTS ===")
    prolog_path = results_dir / "prolog_results.json"
    if prolog_path.exists():
        pr = json.loads(prolog_path.read_text())
        if not pr.get("swipl_found", True):
            print("SWI-Prolog not found; Prolog tests skipped.")
        else:
            print("can_move(a,b):", pr.get("can_move_a_b"))
            print("can_move(a,c):", pr.get("can_move_a_c"))
            print("valid_move(a,b):", pr.get("valid_move_a_b"))
            print("valid_move(b,c):", pr.get("valid_move_b_c"))
            print("valid_move(a,c):", pr.get("valid_move_a_c"))
            print("reduce_speed:", pr.get("reduce_speed"))
            print("animal(polly):", pr.get("animal_polly"))
            print("Cross-check (Python plan moves accepted by Prolog):", pr.get("cross_check"))
    else:
        print("Prolog results not yet generated.")


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if argv and argv[0] == "--regenerate":
        regenerate_all()
        print("\nAll results regenerated in week04_logic/results/")
    elif argv and argv[0] == "--summary":
        print_summary()
    else:
        # default: regenerate then print
        regenerate_all()
        print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())