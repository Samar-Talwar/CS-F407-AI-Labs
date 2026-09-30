# CS F407 Lab, Week 2 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Comprehensive test suite for Goal-Based Agent warehouse navigation."""

from __future__ import annotations

import heapq

import pytest

from week02_agents.src.agent import (
    Action,
    AgentState,
    GoalBasedAgent,
    GoalTest,
    SearchPlanner,
    apply_action,
    astar_search,
    bfs_search,
    dfs_search,
)
from week02_agents.src.environment import (
    SHEET_MAP_STRING,
    Environment,
    parse_map,
)
from week02_agents.src.scaling import scale_map_2x

# --- Independent Oracle Implementation (Dijkstra) ---


def _independent_dijkstra_oracle(
    grid_lines: tuple[str, ...],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[int, list[tuple[int, int]]]:
    """Independent Dijkstra shortest path oracle.

    Built separately from the agent codebase using graph adjacency and priority queue.
    """
    height = len(grid_lines)
    width = len(grid_lines[0])
    distances: dict[tuple[int, int], int] = {start: 0}
    predecessors: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    pq: list[tuple[int, tuple[int, int]]] = [(0, start)]
    visited: set[tuple[int, int]] = set()

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while pq:
        dist, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u == goal:
            # Reconstruct path
            path: list[tuple[int, int]] = []
            curr: tuple[int, int] | None = goal
            while curr is not None:
                path.append(curr)
                curr = predecessors.get(curr)
            path.reverse()
            return dist, path

        ur, uc = u
        for dr, dc in deltas:
            vr, vc = ur + dr, uc + dc
            v = (vr, vc)
            if 0 <= vr < height and 0 <= vc < width and grid_lines[vr][vc] != "#":
                new_dist = dist + 1
                if new_dist < distances.get(v, float("inf")):
                    distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(pq, (new_dist, v))

    return -1, []


# --- Tests ---


class TestSheetMapSolution:
    """Tests evaluating the agent on the official laboratory warehouse map."""

    def test_path_exists_and_reaches_goal(self) -> None:
        """Verify agent finds a path starting at S and ending at G."""
        env = parse_map(SHEET_MAP_STRING)
        agent = GoalBasedAgent(env)
        result = agent.decide()

        assert result.found is True
        assert len(result.path) > 0
        assert result.path[0] == env.start
        assert result.path[-1] == env.goal
        assert result.path_length == len(result.path) - 1

    def test_path_step_validity_and_obstacle_avoidance(self) -> None:
        """Verify every move is exactly 1 step (Manhattan distance 1) and avoids walls."""
        env = parse_map(SHEET_MAP_STRING)
        agent = GoalBasedAgent(env)
        result = agent.decide()

        for i in range(len(result.path) - 1):
            p1 = result.path[i]
            p2 = result.path[i + 1]
            manhattan_dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            assert manhattan_dist == 1, f"Step from {p1} to {p2} is not unit distance."
            assert env.is_valid_move(p2), f"Step {p2} is inside an obstacle or out of bounds."

    def test_optimality_against_independent_dijkstra_oracle(self) -> None:
        """Verify agent BFS path length exactly equals independent Dijkstra oracle."""
        env = parse_map(SHEET_MAP_STRING)
        agent = GoalBasedAgent(env)
        result = agent.decide()

        oracle_length, oracle_path = _independent_dijkstra_oracle(
            env.grid, env.start, env.goal
        )

        assert oracle_length != -1
        assert result.path_length == oracle_length
        assert result.path_length == 20
        assert len(result.path) == 21


class TestUnsolvableAndBoundaryCases:
    """Tests for edge cases, trivial instances, and unsolvable maps."""

    def test_unsolvable_map_exit_safe(self) -> None:
        """Verify agent gracefully reports no path without exceptions or infinite loops."""
        unsolvable_map = """\
#######
#S...##
#...###
###.###
###.#G#
#######\
"""
        env = parse_map(unsolvable_map)
        agent = GoalBasedAgent(env)
        result = agent.decide()

        assert result.found is False
        assert result.path == []
        assert result.path_length == -1
        assert result.expanded_nodes > 0

    def test_start_equals_goal(self) -> None:
        """Verify behavior when start location is also the goal."""
        env = Environment(grid=("###", "#S#", "###"), start=(1, 1), goal=(1, 1))
        agent = GoalBasedAgent(env)
        result = agent.decide()

        assert result.found is True
        assert result.path == [(1, 1)]
        assert result.path_length == 0
        assert result.expanded_nodes == 0

    def test_goal_adjacent_to_start(self) -> None:
        """Verify single-step route when goal is directly adjacent to start."""
        map_str = """\
#####
#SG.#
#...#
#####\
"""
        env = parse_map(map_str)
        agent = GoalBasedAgent(env)
        result = agent.decide()

        assert result.found is True
        assert result.path_length == 1
        assert len(result.path) == 2
        assert result.path == [(1, 1), (1, 2)]

    def test_multiple_equal_routes(self) -> None:
        """Verify optimal path found in diamond grid with multiple shortest paths."""
        diamond_map = """\
#####
#S..#
#.#.#
#..G#
#####\
"""
        env = parse_map(diamond_map)
        agent = GoalBasedAgent(env)
        result = agent.decide()

        oracle_length, _ = _independent_dijkstra_oracle(
            env.grid, env.start, env.goal
        )
        assert result.found is True
        assert result.path_length == oracle_length
        assert result.path_length == 4


class TestMapValidation:
    """Tests for strict map format validation."""

    def test_missing_start_rejected(self) -> None:
        raw = "#####\n#...#\n#..G#\n#####"
        with pytest.raises(ValueError, match="exactly one start 'S'"):
            parse_map(raw)

    def test_missing_goal_rejected(self) -> None:
        raw = "#####\n#S..#\n#...#\n#####"
        with pytest.raises(ValueError, match="exactly one goal 'G'"):
            parse_map(raw)

    def test_multiple_starts_rejected(self) -> None:
        raw = "#####\n#S.S#\n#..G#\n#####"
        with pytest.raises(ValueError, match="exactly one start 'S'"):
            parse_map(raw)

    def test_multiple_goals_rejected(self) -> None:
        raw = "#####\n#S.G#\n#..G#\n#####"
        with pytest.raises(ValueError, match="exactly one goal 'G'"):
            parse_map(raw)

    def test_ragged_rows_rejected(self) -> None:
        raw = "######\n#S..#\n#...G#\n######"
        with pytest.raises(ValueError, match="Ragged map detected"):
            parse_map(raw)

    def test_invalid_character_rejected(self) -> None:
        raw = "#####\n#S.X#\n#..G#\n#####"
        with pytest.raises(ValueError, match="Invalid character 'X'"):
            parse_map(raw)

    def test_empty_map_rejected(self) -> None:
        with pytest.raises(ValueError, match="Map cannot be empty"):
            parse_map("")


class TestRealMutation:
    """Real mutation tests that break actual production code."""

    def test_mutation_ignoring_walls_fails_wall_avoidance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Monkeypatch Environment.is_wall to return False and verify invalid path."""
        env = parse_map(SHEET_MAP_STRING)
        orig_is_wall = Environment.is_wall

        # On the unmutated environment, the wall at (1, 6) prevents straight traversal
        assert orig_is_wall(env, (1, 6)) is True

        # Monkeypatch is_wall on Environment to always return False
        monkeypatch.setattr(Environment, "is_wall", lambda self, pos: False)

        mutated_agent = GoalBasedAgent(env)
        mutated_result = mutated_agent.decide()

        # The mutated search cuts directly through the wall at (1, 6)
        straight_path_contains_wall = (1, 6) in mutated_result.path
        assert straight_path_contains_wall is True
        assert mutated_result.path_length == 18  # Straight line Manhattan distance is shorter

        # Now verify that when evaluated against original unmutated wall logic, this step is invalid
        assert orig_is_wall(env, (1, 6)) is True
        assert env.grid[1][6] == "#"


class TestScalingAndAlgorithms:
    """Tests for 2x map scaling and comparative search baselines (BFS, DFS, A*)."""

    def test_scale_map_2x_properties(self) -> None:
        env = parse_map(SHEET_MAP_STRING)
        scaled = scale_map_2x(env)

        assert scaled.height == env.height * 2
        assert scaled.width == env.width * 2
        assert scaled.is_valid_move(scaled.start) is True
        assert scaled.is_valid_move(scaled.goal) is True

    def test_search_algorithms_comparison(self) -> None:
        env = parse_map(SHEET_MAP_STRING)

        res_bfs = bfs_search(env, env.start, env.goal)
        res_dfs = dfs_search(env, env.start, env.goal)
        res_astar = astar_search(env, env.start, env.goal)

        # BFS and A* with admissible heuristic must produce optimal path length
        assert res_bfs.found is True
        assert res_astar.found is True
        assert res_bfs.path_length == 20
        assert res_astar.path_length == 20

        # DFS finds a path, but is not guaranteed optimal
        assert res_dfs.found is True
        assert res_dfs.path_length >= res_bfs.path_length


class TestAgentComponentsVisibility:
    """Verify architectural components are visible, documented, and modular."""

    def test_agent_components_and_actions(self) -> None:
        env = parse_map(SHEET_MAP_STRING)
        agent = GoalBasedAgent(env)

        # State component
        assert isinstance(agent.state, AgentState)
        assert agent.state.position == env.start

        # Goal component
        assert isinstance(agent.goal_test, GoalTest)
        assert agent.goal_test(env.goal) is True
        assert agent.goal_test(env.start) is False

        # Action component
        assert len(Action) == 4
        next_pos = apply_action((1, 1), Action.RIGHT, env)
        assert next_pos == (1, 2)

        # Decision component
        assert isinstance(agent.planner, SearchPlanner)
        assert isinstance(agent.explain_algorithm(), str)
        assert "Breadth-First Search" in agent.explain_algorithm()
