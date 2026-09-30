# CS F407 Lab, Week 3 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Command-line interface to reproduce all search experiments and display benchmarks."""

from __future__ import annotations

import sys

from week03_search.src.environment import SHEET_MAP_RAW, Environment
from week03_search.src.experiments import (
    run_all_experiments,
)
from week03_search.src.search import astar_search


def print_theoretical_explanation() -> None:
    """Print explanation of A*, Manhattan admissibility, and consistency."""
    print("=" * 78)
    print("THEORETICAL FOUNDATIONS OF A* & MANHATTAN HEURISTIC")
    print("=" * 78)
    print(
        "1. A* Evaluation Function:\n"
        "   f(n) = g(n) + h(n)\n"
        "   - g(n): exact known path cost accumulated from start s0 to current node n.\n"
        "   - h(n): estimated future cost from current node n to goal state G.\n"
        "   - f(n): estimated total cost of cheapest solution passing through n.\n\n"
        "2. Admissibility of Manhattan Distance on 4-Connected Unit-Cost Grid:\n"
        "   - h(n) = |r - r_G| + |c - c_G|\n"
        "   - In a 4-connected grid with unit movement cost c = 1, any step can change\n"
        "     either row or column by at most 1, reducing Manhattan distance by at most 1.\n"
        "   - Obstacles (shelves) can only force detours, never shortcuts.\n"
        "   - Therefore, h(n) <= h*(n) (true optimal remaining cost) everywhere (Admissible).\n\n"
        "3. Consistency (Monotonicity):\n"
        "   - For any successor n' generated from n with step cost c(n, n') = 1:\n"
        "     h(n) <= c(n, n') + h(n')\n"
        "   - Triangle inequality holds: |dr| + |dc| <= 1 + |dr'| + |dc'|.\n"
        "   - Consistency guarantees f(n) is non-decreasing along any path, ensuring that\n"
        "     when a node is expanded from the priority queue, its optimal g-cost is found.\n"
        "     No node needs re-expansion, and graph-search A* is provably optimal.\n"
    )
    print("=" * 78)


def main() -> int:
    """Execute all benchmarks and format results for terminal display."""
    print("\n[CS F407 Lab - Week 3: Search (A* & Heuristic Search)]")
    print("Executing full experimental suite...\n")

    print_theoretical_explanation()

    results = run_all_experiments()

    # Task 3: Systematic Test Suite
    print("\n" + "=" * 78)
    print("TASK 3: SYSTEMATIC TEST RESULTS")
    print("=" * 78)
    t1 = results["test1_sheet_map"]
    t2 = results["test2_trivial"]
    t3 = results["test3_no_solution"]
    t4 = results["test4_alternatives"]

    print(
        f"Test 1 (Sheet 9x17): Found={t1['found']}, "
        f"Len={t1['path_length']}, Exp={t1['states_expanded']}"
    )
    print(
        f"Test 2 (Trivial):    Found={t2['found']}, "
        f"Len={t2['path_length']}, Exp={t2['states_expanded']}"
    )
    print(
        f"Test 3 (No Soln):    Found={t3['found']}, "
        f"Len={t3['path_length']}, Exp={t3['states_expanded']}"
    )
    print(
        f"Test 4 (Alt Paths):  Found={t4['found']}, "
        f"Len={t4['path_length']}, Exp={t4['states_expanded']}"
    )

    # Task 5: BFS vs A* Comparison
    print("\n" + "=" * 78)
    print("TASK 5: BLIND (BFS) VS INFORMED (A*) SEARCH COMPARISON (Sheet Map)")
    print("=" * 78)
    comp = results["bfs_vs_astar"]
    print(f"{'Measure':<25} | {'BFS':<15} | {'A* (Manhattan)':<15}")
    print("-" * 60)
    print(
        f"{'Solution Found':<25} | "
        f"{str(comp['bfs']['found']):<15} | {str(comp['astar']['found']):<15}"
    )
    print(
        f"{'Path Length (steps)':<25} | {comp['bfs']['path_length']:<15} | "
        f"{comp['astar']['path_length']:<15}"
    )
    print(
        f"{'Path Cost':<25} | {comp['bfs']['cost']:<15.1f} | "
        f"{comp['astar']['cost']:<15.1f}"
    )
    print(
        f"{'States Expanded':<25} | {comp['bfs']['states_expanded']:<15} | "
        f"{comp['astar']['states_expanded']:<15}"
    )

    # Task 6: Heuristic Study (Sheet Map)
    print("\n" + "=" * 78)
    print("TASK 6: HEURISTIC COMPARISON ON OFFICIAL WAREHOUSE MAP")
    print("=" * 78)
    h_data = results["heuristic_study"]["heuristics"]
    print(f"{'Heuristic':<20} | {'Found':<6} | {'Length':<8} | {'Expanded':<10} | {'Optimal?':<8}")
    print("-" * 62)
    for name, m in h_data.items():
        print(
            f"{name:<20} | {str(m['found']):<6} | {m['path_length']:<8} | "
            f"{m['states_expanded']:<10} | {str(m['optimal']):<8}"
        )

    # Task 6: Random Grid Stress Study (200 Grids)
    print("\n" + "=" * 78)
    print("TASK 6: STRESS STUDY ON 200 RANDOM 15x15 GRIDS (~25% Walls, Seed 0)")
    print("=" * 78)
    stress = results["stress_study"]["metrics"]
    print(
        f"{'Heuristic':<20} | {'Suboptimal':<12} | {'Optimal Rate':<12} | "
        f"{'Mean Expanded':<18} | {'Rel. Expansion':<14}"
    )
    print("-" * 84)
    for name, m in stress.items():
        subopt = f"{m['suboptimal_path_count']} / {m['total_grids']}"
        opt_rate = f"{m['optimal_path_rate']*100:.1f}%"
        mean_exp = f"{m['mean_states_expanded']:.2f} +/- {m['std_states_expanded']:.2f}"
        rel_exp = f"{m['mean_expansion_ratio_vs_manhattan']:.3f}x"
        print(
            f"{name:<20} | {subopt:<12} | {opt_rate:<12} | "
            f"{mean_exp:<18} | {rel_exp:<14}"
        )

    print("\nRendered sheet map solution path:")
    env = Environment(SHEET_MAP_RAW)
    res = astar_search(env)
    print(env.render_with_path(res.path))
    print("\n[All experiments completed successfully and results saved to results/]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
