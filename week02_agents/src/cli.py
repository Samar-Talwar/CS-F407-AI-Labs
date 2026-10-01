# CS F407 Lab, Week 2 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Command-line interface to reproduce all Week 2 Agent results."""

from __future__ import annotations

import json
from pathlib import Path

from week02_agents.src.agent import GoalBasedAgent
from week02_agents.src.diagram import generate_agent_diagram
from week02_agents.src.environment import SHEET_MAP_STRING, Environment, parse_map
from week02_agents.src.scaling import run_scaling_experiment

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

NO_PATH_MAP_STRING = """\
#######
#S...##
#...###
###.###
###.#G#
#######\
"""

GOAL_ADJACENT_MAP = """\
#####
#SG.#
#...#
#####\
"""

DIAMOND_EQUAL_ROUTES_MAP = """\
#####
#S..#
#.#.#
#..G#
#####\
"""


def run_all() -> None:
    """Run all experiments and generate all result files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("CS F407 Week 2: Goal-Based Agent for Warehouse Navigation")
    print("=" * 60)

    # 1. Sheet Map Resolution
    print("\n[1/5] Solving Official Sheet Map via Goal-Based Agent (BFS)...")
    env = parse_map(SHEET_MAP_STRING)
    agent = GoalBasedAgent(env)
    res = agent.decide()

    rendered_map = env.render(res.path)
    print(rendered_map)
    print(
        f"Result: found={res.found}, steps={res.path_length}, "
        f"nodes_expanded={res.expanded_nodes}, time={res.runtime:.6f}s"
    )

    sheet_result_data = {
        "found": res.found,
        "path": [list(pos) for pos in res.path],
        "path_length": res.path_length,
        "num_coordinates": len(res.path),
        "expanded_nodes": res.expanded_nodes,
        "runtime_seconds": res.runtime,
        "algorithm": res.algorithm,
        "message": "Path found" if res.found else "No path exists",
    }
    with open(RESULTS_DIR / "sheet_map_result.json", "w", encoding="utf-8") as f:
        json.dump(sheet_result_data, f, indent=2)

    with open(RESULTS_DIR / "sheet_map_path.txt", "w", encoding="utf-8") as f:
        f.write(rendered_map + "\n")

    # 2. Unsolvable / No-Path Case
    print("\n[2/5] Evaluating Unsolvable Map (Exit-safe No-Path Behavior)...")
    env_unsolvable = parse_map(NO_PATH_MAP_STRING)
    agent_unsolvable = GoalBasedAgent(env_unsolvable)
    res_unsolvable = agent_unsolvable.decide()

    no_path_data = {
        "found": res_unsolvable.found,
        "path": res_unsolvable.path,
        "path_length": res_unsolvable.path_length,
        "expanded_nodes": res_unsolvable.expanded_nodes,
        "runtime_seconds": res_unsolvable.runtime,
        "algorithm": res_unsolvable.algorithm,
        "message": "No path exists" if not res_unsolvable.found else "Path found",
    }
    with open(RESULTS_DIR / "no_path_result.json", "w", encoding="utf-8") as f:
        json.dump(no_path_data, f, indent=2)
    print(f"Unsolvable result: found={res_unsolvable.found}, message={no_path_data['message']}")

    # 3. Trivial & Boundary Cases
    print("\n[3/5] Evaluating Trivial and Boundary Scenarios...")
    trivial_data = {}

    # 3a. Start equals Goal
    env_same = Environment(grid=("###", "#S#", "###"), start=(1, 1), goal=(1, 1))
    agent_same = GoalBasedAgent(env_same)
    res_same = agent_same.decide()
    trivial_data["start_equals_goal"] = {
        "found": res_same.found,
        "path": [list(p) for p in res_same.path],
        "path_length": res_same.path_length,
        "expanded_nodes": res_same.expanded_nodes,
    }

    # 3b. Goal adjacent to Start
    env_adj = parse_map(GOAL_ADJACENT_MAP)
    agent_adj = GoalBasedAgent(env_adj)
    res_adj = agent_adj.decide()
    trivial_data["goal_adjacent_to_start"] = {
        "found": res_adj.found,
        "path": [list(p) for p in res_adj.path],
        "path_length": res_adj.path_length,
        "expanded_nodes": res_adj.expanded_nodes,
    }

    # 3c. Multiple equal-length routes
    env_diamond = parse_map(DIAMOND_EQUAL_ROUTES_MAP)
    agent_diamond = GoalBasedAgent(env_diamond)
    res_diamond = agent_diamond.decide()
    trivial_data["multiple_equal_routes"] = {
        "found": res_diamond.found,
        "path": [list(p) for p in res_diamond.path],
        "path_length": res_diamond.path_length,
        "expanded_nodes": res_diamond.expanded_nodes,
    }

    with open(RESULTS_DIR / "trivial_cases.json", "w", encoding="utf-8") as f:
        json.dump(trivial_data, f, indent=2)
    print("Trivial cases recorded.")

    # 4. Scaling Experiment (Original vs 2x Map: BFS vs DFS vs A*)
    print("\n[4/5] Running Scaling Experiment (Original vs 2x Map for BFS, DFS, A*)...")
    scaling_data = run_scaling_experiment(env)
    with open(RESULTS_DIR / "scaling.json", "w", encoding="utf-8") as f:
        json.dump(scaling_data, f, indent=2)

    for algo in ["BFS", "DFS", "A*"]:
        bench = scaling_data["benchmarks"][algo]
        orig = bench["original"]
        scaled = bench["scaled_2x"]
        ratio = bench["scaling_ratios"]
        print(
            f"  {algo:4s} | "
            f"Orig: steps={orig['path_length']}, "
            f"expanded={orig['nodes_expanded']:3d} | "
            f"2x: steps={scaled['path_length']}, "
            f"expanded={scaled['nodes_expanded']:3d} | "
            f"Expansion Ratio: "
            f"{ratio['nodes_expanded_ratio_2x_to_orig']:.2f}x"
        )

    # 5. Generate Agent Architecture Block Diagram
    print("\n[5/5] Generating Agent Architecture Diagram...")
    diagram_path = RESULTS_DIR / "agent_diagram.png"
    generate_agent_diagram(diagram_path)
    print(f"Diagram saved to: {diagram_path}")

    print("\nAll Week 2 results generated successfully.")


if __name__ == "__main__":
    run_all()
