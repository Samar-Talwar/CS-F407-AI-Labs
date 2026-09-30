# CS F407 Lab, Week 2 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Scaling experiment: generate 2x scaled map and compare search algorithms."""

from __future__ import annotations

import time
from typing import Any

from week02_agents.src.agent import astar_search, bfs_search, dfs_search
from week02_agents.src.environment import Environment, parse_map


def scale_map_2x(env: Environment) -> Environment:
    """Scale an environment 2x in both dimensions with identical topology.

    Each cell in the original H x W grid expands to a 2x2 block in the 2H x 2W grid:
    - Wall '#' becomes a 2x2 block of '#'.
    - Free space '.' becomes a 2x2 block of '.'.
    - Start 'S' becomes 'S' at (2*r, 2*c) and '.' in the other 3 cells.
    - Goal 'G' becomes 'G' at (2*r + 1, 2*c + 1) and '.' in the other 3 cells.

    Args:
        env: Source Environment.

    Returns:
        New Environment scaled 2x in height and width.
    """
    orig_lines = env.grid
    scaled_lines: list[list[str]] = [
        ["." for _ in range(env.width * 2)] for _ in range(env.height * 2)
    ]

    for r in range(env.height):
        for c in range(env.width):
            char = orig_lines[r][c]
            r0, r1 = 2 * r, 2 * r + 1
            c0, c1 = 2 * c, 2 * c + 1

            if char == "#":
                scaled_lines[r0][c0] = "#"
                scaled_lines[r0][c1] = "#"
                scaled_lines[r1][c0] = "#"
                scaled_lines[r1][c1] = "#"
            elif char == ".":
                scaled_lines[r0][c0] = "."
                scaled_lines[r0][c1] = "."
                scaled_lines[r1][c0] = "."
                scaled_lines[r1][c1] = "."
            elif char == "S":
                scaled_lines[r0][c0] = "S"
                scaled_lines[r0][c1] = "."
                scaled_lines[r1][c0] = "."
                scaled_lines[r1][c1] = "."
            elif char == "G":
                scaled_lines[r0][c0] = "."
                scaled_lines[r0][c1] = "."
                scaled_lines[r1][c0] = "."
                scaled_lines[r1][c1] = "G"

    joined = ["".join(row) for row in scaled_lines]
    return parse_map(joined)


def run_scaling_experiment(env: Environment) -> dict[str, Any]:
    """Run BFS, DFS, and A* on original and 2x scaled maps, collecting metrics.

    Args:
        env: Original warehouse environment.

    Returns:
        Dictionary containing comparative benchmarks at full precision.
    """
    env_2x = scale_map_2x(env)

    algorithms = [
        ("BFS", bfs_search),
        ("DFS", dfs_search),
        ("A*", astar_search),
    ]

    results: dict[str, Any] = {
        "original_map": {
            "dimensions": [env.height, env.width],
            "start": list(env.start),
            "goal": list(env.goal),
        },
        "scaled_2x_map": {
            "dimensions": [env_2x.height, env_2x.width],
            "start": list(env_2x.start),
            "goal": list(env_2x.goal),
        },
        "benchmarks": {},
    }

    # Multiple timing iterations for stable runtime measurement
    num_runs = 50

    for name, search_fn in algorithms:
        # Original map
        res_orig = search_fn(env, env.start, env.goal)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            search_fn(env, env.start, env.goal)
        avg_runtime_orig = (time.perf_counter() - t0) / num_runs

        # Scaled 2x map
        res_2x = search_fn(env_2x, env_2x.start, env_2x.goal)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            search_fn(env_2x, env_2x.start, env_2x.goal)
        avg_runtime_2x = (time.perf_counter() - t0) / num_runs

        nodes_ratio = (
            res_2x.expanded_nodes / res_orig.expanded_nodes
            if res_orig.expanded_nodes > 0
            else None
        )
        runtime_ratio = (
            avg_runtime_2x / avg_runtime_orig
            if avg_runtime_orig > 0
            else None
        )

        results["benchmarks"][name] = {
            "original": {
                "found": res_orig.found,
                "path_length": res_orig.path_length,
                "nodes_expanded": res_orig.expanded_nodes,
                "runtime_seconds": avg_runtime_orig,
            },
            "scaled_2x": {
                "found": res_2x.found,
                "path_length": res_2x.path_length,
                "nodes_expanded": res_2x.expanded_nodes,
                "runtime_seconds": avg_runtime_2x,
            },
            "scaling_ratios": {
                "nodes_expanded_ratio_2x_to_orig": nodes_ratio,
                "runtime_ratio_2x_to_orig": runtime_ratio,
            },
        }

    return results
