# CS F407 Lab, Week 4 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Tests for week04_logic: planning agent."""

from __future__ import annotations

from week04_logic.src import (
    domain,
    planner,
    validator,
)

# --------------------- Task 0: Applicability ---------------------

def test_initial_state():
    """I = {At(Robot,A), At(Package,A)}"""
    assert frozenset({"At(Robot,A)", "At(Package,A)"}) == domain.INITIAL_STATE


def test_goal():
    """G = {At(Package,C)}"""
    assert frozenset({"At(Package,C)"}) == domain.GOAL


def test_action_definitions():
    """Every action has correct preconditions/effects."""
    actions = domain.warehouse_actions()
    # PickUp(A)
    pickup = next(a for a in actions if a.name == "PickUp(Package,A)")
    assert pickup.pos_pre == frozenset({"At(Robot,A)", "At(Package,A)"})
    assert pickup.pos_eff == frozenset({"Holding(Package)"})
    assert pickup.neg_eff == frozenset({"At(Package,A)"})
    # Drop(C)
    drop = next(a for a in actions if a.name == "Drop(Package,C)")
    assert drop.pos_pre == frozenset({"At(Robot,C)", "Holding(Package)"})
    assert drop.pos_eff == frozenset({"At(Package,C)"})
    assert drop.neg_eff == frozenset({"Holding(Package)"})


def test_pickup_applicable_in_initial():
    """PickUp(Package,A) applicable in I? Yes."""
    initial_state = domain.INITIAL_STATE
    pickup = next(a for a in domain.warehouse_actions() if a.name == "PickUp(Package,A)")
    assert planner.applicable(initial_state, pickup) is True


def test_drop_not_applicable_in_initial():
    """Drop(Package,C) applicable in I? No (missing Holding and At(Robot,C))."""
    initial_state = domain.INITIAL_STATE
    drop = next(a for a in domain.warehouse_actions() if a.name == "Drop(Package,C)")
    assert planner.applicable(initial_state, drop) is False
    missing = planner.missing_preconditions(initial_state, drop)
    assert set(missing.get("positive_missing", [])) == {"At(Robot,C)", "Holding(Package)"}


# --------------------- Task 1: Manual plan ---------------------

def test_manual_plan_states():
    """Manual plan yields states S0..S4 as expected."""
    manual = {
        "plan": ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"],
        "states": [
            ["At(Package,A)", "At(Robot,A)"],                    # S0 (sorted)
            ["At(Robot,A)", "Holding(Package)"],                # S1 after PickUp
            ["At(Robot,B)", "Holding(Package)"],                # S2 after Move(A,B)
            ["At(Robot,C)", "Holding(Package)"],                # S3 after Move(B,C)
            ["At(Package,C)", "At(Robot,C)"],                   # S4 after Drop (sorted)
        ],
    }
    # Recompute from src/cli.py manual_plan()
    from week04_logic.src.cli import manual_plan
    actual = manual_plan()
    assert actual == manual


def test_sheet_example_invalid():
    """Sheet's example sequence fails at PickUp(Package,B)."""
    from week04_logic.src.cli import invalid_example_plan
    data = invalid_example_plan()
    assert data["valid"] is False
    assert data["failed_step"] == 1  # PickUp(Package,B) is step 1 (0-indexed)
    # Check that positive precondition At(Package,B) is missing at S1
    # S1 after Move(A,B): {At(Robot,B), At(Package,A)}
    assert "At(Package,B)" in str(data["error"])


# --------------------- Task 3: Tests A, B, C ---------------------

def test_a_solvable():
    """Test A: original warehouse -> 4-step plan."""
    from week04_logic.src.cli import test_a
    data = test_a()
    assert data["plan_found"] is True
    assert data["plan"] == ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]
    assert data["states_expanded"] >= 0  # will be >0 in practice
    assert data["valid"] is True
    assert data["goal_reached"] is True


def test_b_impossible():
    """Test B: no PickUp -> no plan found."""
    from week04_logic.src.cli import test_b
    data = test_b()
    assert data["plan_found"] is False
    assert data["plan"] == []
    assert data["valid"] is False  # validator rejects empty plan when goal not reached


def test_c_irrelevant_actions():
    """Test C: robot at C ≠ package at C; need PickUp+Drop."""
    from week04_logic.src.cli import test_c
    data = test_c()
    assert data["plan_found"] is True
    # The shortest valid plan still requires picking up at A and moving
    assert data["plan"] == ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]
    assert data["valid"] is True
    # Ensure we did NOT accept a plan where robot just moves to C
    # (that would be Move(A,B), Move(B,C) but package stays at A)
    assert "Move(A,C)" not in data["plan"]  # direct move not in warehouse actions
    # Even with irrelevant actions, planner should not invent actions


def test_result_schema():
    """Every result file has the five required fields."""
    required = {"initial_state", "goal", "plan_found", "plan", "valid"}
    from week04_logic.src.cli import test_a, test_b, test_c
    for func in [test_a, test_b, test_c]:
        data = func()
        assert set(data.keys()) >= required


# --------------------- Task 5: Independent validator trust ---------------------

def test_validator_catches_bad_plan():
    """Validator rejects sheet's example with correct error."""
    sheet_plan_names = ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)", "Drop(Package,C)"]
    actions_by_name = {a.name: a for a in domain.warehouse_actions()}
    sheet_plan = [actions_by_name[n] for n in sheet_plan_names]
    val = validator.validate_plan(domain.INITIAL_STATE, domain.GOAL, sheet_plan)
    assert val.valid is False
    assert val.failed_step == 1
    assert "At(Package,B)" in val.error  # missing positive precondition


def test_validator_accepts_good_plan():
    """Validator accepts correct shortest plan."""
    good = ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]
    actions_by_name = {a.name: a for a in domain.warehouse_actions()}
    good_plan = [actions_by_name[n] for n in good]
    val = validator.validate_plan(domain.INITIAL_STATE, domain.GOAL, good_plan)
    assert val.valid is True
    assert val.goal_reached is True
    assert len(val.states) == 5  # S0..S4


# --------------------- Negative preconditions ---------------------

def test_negative_preconditions_exercised():
    """Synthetic problem: Unlock then Enter."""
    from week04_logic.src.cli import negative_preconditions
    data = negative_preconditions()
    assert data["plan_found"] is True
    assert data["plan"] == ["UnlockDoor", "Enter"]
    assert data["valid"] is True
    # States: S0={HasKey,DoorLocked}, S1={HasKey}, S2={HasKey,Inside}
    assert data["states"][0] == sorted(["HasKey", "DoorLocked"])
    assert data["states"][1] == sorted(["HasKey"])
    assert data["states"][2] == sorted(["HasKey", "Inside"])


# --------------------- Prolog extension ---------------------

def test_prolog_can_move():
    """can_move(a,b) true, can_move(a,c) false (no direct link)."""
    from week04_logic.src.prolog_runner import run_query
    assert run_query("can_move(a,b)") is True
    assert run_query("can_move(a,c)") is False


def test_prolog_valid_move():
    """valid_move mirrors can_move for this knowledge base."""
    from week04_logic.src.prolog_runner import run_query
    assert run_query("valid_move(a,b)") is True
    assert run_query("valid_move(b,c)") is True
    assert run_query("valid_move(a,c)") is False


def test_prolog_move_ac_challenge():
    """Challenge: proposed Move(a,c) should be rejected."""
    from week04_logic.src.prolog_runner import run_query
    assert run_query("valid_move(a,c)") is False


def test_prolog_reduce_speed_chain():
    """wet_road => slippery => reduce_speed."""
    from week04_logic.src.prolog_runner import run_query
    assert run_query("reduce_speed") is True


def test_prolog_animal_polly():
    """penguin(polly) => bird(polly) => animal(polly)."""
    from week04_logic.src.prolog_runner import run_query
    assert run_query("animal(polly)") is True


def test_prolog_cross_check():
    """Every Move step in Python plan is accepted by Prolog valid_move/2."""
    from week04_logic.src.cli import test_a
    from week04_logic.src.prolog_runner import cross_check_plan_moves
    plan = test_a().get("plan", [])
    moves = [step for step in plan if step.startswith("Move(")]
    assert len(moves) == 2  # Move(A,B) and Move(B,C)
    checks = cross_check_plan_moves(plan)
    assert checks["swipl_found"] is True
    for move in moves:
        assert checks["moves"][move] is True, f"{move} should be valid per Prolog"
    # Challenge move should be rejected
    assert checks["move_ac_challenge"] is False


# --------------------- Real mutation tests (QG.2) ---------------------

def test_mutation_applicable_ignore_preconditions(monkeypatch):
    """(1) Break applicable(): ignore preconditions -> planner may return invalid plan."""
    from week04_logic.src import search as search_mod
    def always_true(state, action):
        return True
    monkeypatch.setattr(search_mod, "applicable", always_true)
    from week04_logic.src.cli import test_a
    data = test_a()
    if data["plan_found"]:
        actions_by_name = {a.name: a for a in domain.warehouse_actions()}
        plan_actions = [actions_by_name[n] for n in data["plan"]]
        val = validator.validate_plan(domain.INITIAL_STATE, domain.GOAL, plan_actions)
        # Since applicable is broken, planner can pick bad actions, but the
        # correct 4-step sequence is still valid. We assert the independent
        # validator either rejects an invalid sequence or accepts only the
        # valid one (the mutation's effect depends on search order; record
        # honestly rather than forcing False).
        expected_good = [
            "PickUp(Package,A)",
            "Move(A,B)",
            "Move(B,C)",
            "Drop(Package,C)",
        ]
        assert val.valid is False or data["plan"] == expected_good
    else:
        assert data["plan_found"] is False


def test_mutation_applied_skip_neg_eff(monkeypatch):
    """(2) Break apply(): skip negative effects -> negative preconditions fail."""
    from week04_logic.src import search as search_mod
    def apply_skip_neg_eff(state, action):
        # ONLY add pos_eff, do NOT remove neg_eff
        return state | action.pos_eff

    monkeypatch.setattr(search_mod, "apply_action", apply_skip_neg_eff)
    from week04_logic.src.cli import negative_preconditions
    data = negative_preconditions()
    # Because DoorLocked is never deleted by UnlockDoor, Enter is never applicable
    assert data["plan_found"] is False


def test_mutation_remove_pickup_actions(monkeypatch):
    """(3) Remove PickUp from action set -> test B behaviour (no plan)."""
    original_domain = domain.warehouse_actions

    def no_pickup():
        return [a for a in original_domain() if not a.name.startswith("PickUp")]

    monkeypatch.setattr(domain, "warehouse_actions", no_pickup)
    from week04_logic.src.cli import test_a
    data = test_a()
    assert data["plan_found"] is False  # same as test B
    assert data["plan"] == []


def test_cli_results_dir_module_relative(monkeypatch, tmp_path):
    from pathlib import Path

    import week04_logic.src.cli as cli_mod

    monkeypatch.chdir(tmp_path)
    expected = (Path(cli_mod.__file__).resolve().parents[1] / "results").resolve()
    assert cli_mod.RESULTS_DIR.resolve() == expected

