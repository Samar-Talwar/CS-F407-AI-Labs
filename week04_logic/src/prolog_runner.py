# CS F407 Lab, Week 4 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Prolog extension: run queries against planner.pl via swipl subprocess."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

PROLOG_FILE = Path(__file__).resolve().parent.parent / "prolog" / "planner.pl"


def swipl_available() -> bool:
    """Return True if swipl is on PATH."""
    return shutil.which("swipl") is not None


def run_query(query: str) -> bool | None:
    """Run a Prolog query and return True/False, or None if swipl is absent."""
    if not swipl_available():
        return None
    # swipl -g "query" -t halt returns exit 0 if query succeeds, 1 if it fails
    result = subprocess.run(
        ["swipl", "-f", str(PROLOG_FILE), "-g", query, "-t", "halt"],
        capture_output=True, text=True, timeout=10,
    )
    return result.returncode == 0


def run_all_prolog_queries() -> dict:
    """Run every query from the lab sheet and return a results dict."""
    if not swipl_available():
        return {"swipl_found": False, "error": "swipl not found on PATH"}

    queries = {
        "can_move_a_b": "can_move(a,b)",
        "can_move_a_c": "can_move(a,c)",
        "valid_move_a_b": "valid_move(a,b)",
        "valid_move_b_c": "valid_move(b,c)",
        "valid_move_a_c": "valid_move(a,c)",
        "reduce_speed": "reduce_speed",
        "animal_polly": "animal(polly)",
    }
    results: dict = {"swipl_found": True}
    for key, q in queries.items():
        results[key] = run_query(q)
    return results


def cross_check_plan_moves(plan: list[str]) -> dict:
    """Check that every Move step in the Python plan is accepted by Prolog.

    Also check that the illegal Move(a,c) is rejected.
    """
    if not swipl_available():
        return {"swipl_found": False}

    checks: dict = {"swipl_found": True, "moves": {}}
    for step in plan:
        if step.startswith("Move("):
            # Parse Move(X,Y) -> valid_move(x,y)
            inner = step[5:-1]  # "A,B"
            src, dst = inner.split(",")
            query = f"valid_move({src.strip().lower()},{dst.strip().lower()})"
            checks["moves"][step] = run_query(query)

    # Check illegal move
    checks["move_ac_challenge"] = run_query("valid_move(a,c)")
    return checks


def write_prolog_results(results_dir: Path, plan: list[str]) -> dict:
    """Run all Prolog queries and write results/prolog_results.json."""
    data = run_all_prolog_queries()
    data["cross_check"] = cross_check_plan_moves(plan)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "prolog_results.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    return data
