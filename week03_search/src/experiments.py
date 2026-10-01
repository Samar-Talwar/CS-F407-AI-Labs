# CS F407 Lab, Week 3 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Benchmark experiments: sheet map, boundary tests, BFS vs A*, and random stress study."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

from week03_search.src.environment import (
    SHEET_MAP_RAW,
    TEST2_TRIVIAL_RAW,
    TEST3_NO_SOLUTION_RAW,
    TEST4_ALTERNATIVE_PATHS_RAW,
    Environment,
    GridState,
)
from week03_search.src.heuristics import (
    euclidean_distance,
    manhattan_distance,
    scaled_manhattan,
    zero_heuristic,
)
from week03_search.src.search import (
    astar_search,
    bfs_search,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def ensure_results_dir() -> Path:
    """Ensure results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def run_sheet_map_test() -> dict[str, Any]:
    """Test 1: Run A* on the official 9x17 warehouse map."""
    env = Environment(SHEET_MAP_RAW)
    result = astar_search(env, heuristic=manhattan_distance, heuristic_name="manhattan")
    rendered_map = env.render_with_path(result.path)

    data = {
        "test_name": "Test 1: Original Warehouse (9x17)",
        "found": result.found,
        "path": result.path,
        "path_length": result.path_length,
        "cost": result.cost,
        "states_expanded": result.states_expanded,
        "algorithm": result.algorithm,
        "heuristic": result.heuristic,
        "grid_rows": env.rows,
        "grid_cols": env.cols,
        "start": env.start,
        "goal": env.goal,
    }

    results_dir = ensure_results_dir()
    with open(results_dir / "original_warehouse.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(results_dir / "sheet_map_path.txt", "w", encoding="utf-8") as f:
        f.write(rendered_map + "\n")

    return data


def run_test2_trivial() -> dict[str, Any]:
    """Test 2: Trivial adjacent goal (3x5)."""
    env = Environment(TEST2_TRIVIAL_RAW)
    result = astar_search(env, heuristic=manhattan_distance, heuristic_name="manhattan")

    data = {
        "test_name": "Test 2: Trivial Case (Adjacent Goal)",
        "found": result.found,
        "path": result.path,
        "path_length": result.path_length,
        "cost": result.cost,
        "states_expanded": result.states_expanded,
        "algorithm": result.algorithm,
        "heuristic": result.heuristic,
        "start": env.start,
        "goal": env.goal,
    }

    results_dir = ensure_results_dir()
    with open(results_dir / "test2_trivial.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def run_test3_no_solution() -> dict[str, Any]:
    """Test 3: Unreachable goal (5x7)."""
    env = Environment(TEST3_NO_SOLUTION_RAW)
    result = astar_search(env, heuristic=manhattan_distance, heuristic_name="manhattan")

    data = {
        "test_name": "Test 3: No Solution (Unreachable Goal)",
        "found": result.found,
        "path": result.path,
        "path_length": result.path_length,
        "cost": result.cost,
        "states_expanded": result.states_expanded,
        "algorithm": result.algorithm,
        "heuristic": result.heuristic,
        "start": env.start,
        "goal": env.goal,
    }

    results_dir = ensure_results_dir()
    with open(results_dir / "test3_no_solution.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def run_test4_alternatives() -> dict[str, Any]:
    """Test 4: Alternative paths (two equal shortest paths + decoy longer path)."""
    env = Environment(TEST4_ALTERNATIVE_PATHS_RAW)
    result = astar_search(env, heuristic=manhattan_distance, heuristic_name="manhattan")

    data = {
        "test_name": "Test 4: Alternative Paths (Equal Shortest + Longer Decoy)",
        "found": result.found,
        "path": result.path,
        "path_length": result.path_length,
        "cost": result.cost,
        "states_expanded": result.states_expanded,
        "algorithm": result.algorithm,
        "heuristic": result.heuristic,
        "start": env.start,
        "goal": env.goal,
    }

    results_dir = ensure_results_dir()
    with open(results_dir / "test4_alternatives.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def run_bfs_vs_astar() -> dict[str, Any]:
    """Task 5: Compare BFS with A* on the official warehouse map."""
    env = Environment(SHEET_MAP_RAW)
    bfs_res = bfs_search(env)
    astar_res = astar_search(env, heuristic=manhattan_distance, heuristic_name="manhattan")

    data = {
        "bfs": {
            "found": bfs_res.found,
            "path_length": bfs_res.path_length,
            "cost": bfs_res.cost,
            "states_expanded": bfs_res.states_expanded,
            "path": bfs_res.path,
        },
        "astar": {
            "found": astar_res.found,
            "path_length": astar_res.path_length,
            "cost": astar_res.cost,
            "states_expanded": astar_res.states_expanded,
            "path": astar_res.path,
        },
        "both_found": bfs_res.found and astar_res.found,
        "same_path_length": bfs_res.path_length == astar_res.path_length,
        "fewer_states_expanded": (
            "astar" if astar_res.states_expanded < bfs_res.states_expanded else "equal_or_bfs"
        ),
        "expansion_reduction_percent": (
            (bfs_res.states_expanded - astar_res.states_expanded) / bfs_res.states_expanded * 100.0
            if bfs_res.states_expanded > 0
            else 0.0
        ),
    }

    results_dir = ensure_results_dir()
    with open(results_dir / "bfs_vs_astar.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def run_heuristic_study() -> dict[str, Any]:
    """Task 6: Compare heuristics on the official warehouse map."""
    env = Environment(SHEET_MAP_RAW)
    bfs_ref = bfs_search(env)
    optimal_length = bfs_ref.path_length

    heuristics = [
        ("manhattan", manhattan_distance, "Manhattan |x - x_G| + |y - y_G| (admissible)"),
        ("zero", zero_heuristic, "Zero h(n) = 0 (blind search / UCS, admissible)"),
        ("euclidean", euclidean_distance, "Euclidean sqrt(dx^2 + dy^2) (admissible)"),
        ("scaled_manhattan_2x", scaled_manhattan(2.0), "2x Manhattan (inadmissible)"),
    ]

    results: dict[str, Any] = {}
    for key, h_fn, desc in heuristics:
        res = astar_search(env, heuristic=h_fn, heuristic_name=key)
        results[key] = {
            "description": desc,
            "found": res.found,
            "path_length": res.path_length,
            "cost": res.cost,
            "states_expanded": res.states_expanded,
            "optimal": res.path_length == optimal_length if res.found else False,
            "path": res.path,
        }

    output_data = {
        "map": "Original Warehouse (9x17)",
        "reference_optimal_length": optimal_length,
        "heuristics": results,
    }

    results_dir = ensure_results_dir()
    with open(results_dir / "heuristic_study.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    return output_data


def generate_random_grid(
    size: int,
    wall_prob: float,
    rng: random.Random,
) -> tuple[str, GridState, GridState]:
    """Generate a random grid map of given size with border walls."""
    lines: list[list[str]] = []
    for r in range(size):
        row: list[str] = []
        for c in range(size):
            if r == 0 or r == size - 1 or c == 0 or c == size - 1:
                row.append("#")
            else:
                row.append("#" if rng.random() < wall_prob else ".")
        lines.append(row)

    start = (1, 1)
    goal = (size - 2, size - 2)
    lines[start[0]][start[1]] = "S"
    lines[goal[0]][goal[1]] = "G"

    raw_map = "\n".join("".join(row) for row in lines)
    return raw_map, start, goal


def run_stress_study(
    num_grids: int = 200,
    grid_size: int = 15,
    wall_prob: float = 0.25,
    seed: int = 0,
) -> dict[str, Any]:
    """Task 6 Stress Study: 200 random 15x15 solvable grids to evaluate admissibility.

    Compares Manhattan, h=0, Euclidean, and 2x Manhattan against independent BFS/Dijkstra oracle.
    """
    rng = random.Random(seed)
    solvable_grids: list[str] = []

    # Generate exact number of solvable grids
    while len(solvable_grids) < num_grids:
        raw_map, _, _ = generate_random_grid(grid_size, wall_prob, rng)
        env = Environment(raw_map)
        oracle_res = bfs_search(env)
        if oracle_res.found:
            solvable_grids.append(raw_map)

    heuristic_defs = [
        ("manhattan", manhattan_distance),
        ("zero", zero_heuristic),
        ("euclidean", euclidean_distance),
        ("scaled_manhattan_2x", scaled_manhattan(2.0)),
    ]

    # Collect per-grid data
    expansions: dict[str, list[int]] = {k: [] for k, _ in heuristic_defs}
    suboptimal_counts: dict[str, int] = {k: 0 for k, _ in heuristic_defs}
    path_lengths: dict[str, list[int]] = {k: [] for k, _ in heuristic_defs}

    for raw_map in solvable_grids:
        env = Environment(raw_map)
        oracle_res = bfs_search(env)
        oracle_len = oracle_res.path_length

        for key, h_fn in heuristic_defs:
            res = astar_search(env, heuristic=h_fn, heuristic_name=key)
            expansions[key].append(res.states_expanded)
            path_lengths[key].append(res.path_length)
            if res.path_length > oracle_len:
                suboptimal_counts[key] += 1

    summary_metrics: dict[str, Any] = {}
    manhattan_expansions = expansions["manhattan"]

    for key, _ in heuristic_defs:
        exp_list = expansions[key]
        ratios = [
            e / m if m > 0 else 1.0
            for e, m in zip(exp_list, manhattan_expansions, strict=True)
        ]
        summary_metrics[key] = {
            "total_grids": num_grids,
            "suboptimal_path_count": suboptimal_counts[key],
            "optimal_path_rate": (num_grids - suboptimal_counts[key]) / num_grids,
            "mean_states_expanded": statistics.mean(exp_list),
            "std_states_expanded": statistics.stdev(exp_list) if len(exp_list) > 1 else 0.0,
            "min_states_expanded": min(exp_list),
            "max_states_expanded": max(exp_list),
            "mean_expansion_ratio_vs_manhattan": statistics.mean(ratios),
        }

    output_data = {
        "configuration": {
            "num_solvable_grids": num_grids,
            "grid_size": grid_size,
            "wall_probability": wall_prob,
            "random_seed": seed,
        },
        "metrics": summary_metrics,
    }

    results_dir = ensure_results_dir()
    with open(results_dir / "stress_study.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    return output_data


def run_all_experiments() -> dict[str, Any]:
    """Execute the full experimental suite and save all results."""
    return {
        "test1_sheet_map": run_sheet_map_test(),
        "test2_trivial": run_test2_trivial(),
        "test3_no_solution": run_test3_no_solution(),
        "test4_alternatives": run_test4_alternatives(),
        "bfs_vs_astar": run_bfs_vs_astar(),
        "heuristic_study": run_heuristic_study(),
        "stress_study": run_stress_study(),
    }
