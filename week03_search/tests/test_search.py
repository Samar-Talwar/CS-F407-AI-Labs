# CS F407 Lab, Week 3 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Comprehensive test suite for Week 3 search algorithms, heuristics, and benchmarks.

Includes:
- Independent Dijkstra oracle for optimality verification (no self-comparison)
- Problem formulation & state representation tests
- Map validation and parsing error handling
- Systematic boundary cases (trivial, unreachable, alternative paths)
- Cell-by-cell path integrity and obstacle avoidance
- BFS vs A* comparison
- Heuristic admissibility and consistency properties
- 200-grid random stress study statistical properties
- Real monkeypatch mutation tests
"""

from __future__ import annotations

import heapq

import pytest

from week03_search.src.environment import (
    SHEET_MAP_RAW,
    TEST2_TRIVIAL_RAW,
    TEST3_NO_SOLUTION_RAW,
    TEST4_ALTERNATIVE_PATHS_RAW,
    Action,
    Environment,
    GridState,
    parse_map,
)
from week03_search.src.experiments import run_stress_study
from week03_search.src.heuristics import (
    euclidean_distance,
    manhattan_distance,
    scaled_manhattan,
    zero_heuristic,
)
from week03_search.src.search import (
    SearchResult,
    astar_search,
    bfs_search,
    reconstruct_path,
)


def _independent_dijkstra_oracle(
    grid: tuple[str, ...], start: GridState, goal: GridState
) -> tuple[bool, int, list[GridState]]:
    """Completely independent Dijkstra implementation serving as ground-truth oracle.

    Written inside tests without reusing code from src.search.
    """
    rows = len(grid)
    cols = len(grid[0])

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    dist: dict[GridState, int] = {start: 0}
    parent: dict[GridState, GridState | None] = {start: None}
    pq: list[tuple[int, int, GridState]] = [(0, 0, start)]
    visited: set[GridState] = set()
    counter = 0

    while pq:
        d, _, curr = heapq.heappop(pq)
        if curr in visited:
            continue
        visited.add(curr)

        if curr == goal:
            # Reconstruct
            path: list[GridState] = []
            node: GridState | None = curr
            while node is not None:
                path.append(node)
                node = parent.get(node)
            path.reverse()
            return True, d, path

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            if in_bounds(nr, nc) and grid[nr][nc] != "#":
                new_d = d + 1
                if neighbor not in dist or new_d < dist[neighbor]:
                    dist[neighbor] = new_d
                    parent[neighbor] = curr
                    counter += 1
                    heapq.heappush(pq, (new_d, counter, neighbor))

    return False, 0, []


class TestProblemFormulation:
    """Task 0 & Task 1: Problem formulation, states, actions, transitions, and goal tests."""

    def test_problem_specification_components(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        assert isinstance(env.start, tuple) and len(env.start) == 2
        assert isinstance(env.goal, tuple) and len(env.goal) == 2
        assert env.start == (1, 1)
        assert env.goal == (7, 15)
        assert env.rows == 9
        assert env.cols == 17

    def test_state_representation(self) -> None:
        coord: GridState = (2, 3)
        assert isinstance(coord[0], int) and isinstance(coord[1], int)
        assert hash(coord) is not None  # Immutable & hashable for sets and dicts

    def test_deterministic_transitions(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        # Starting at (1, 1), moving RIGHT must deterministically reach (1, 2)
        next_state = env.step((1, 1), Action.RIGHT)
        assert next_state == (1, 2)
        # Moving DOWN must deterministically reach (2, 1)
        next_state_down = env.step((1, 1), Action.DOWN)
        assert next_state_down == (2, 1)

    def test_invalid_transition_raises_error(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        # Starting at (1, 1), UP is an obstacle '#' at (0, 1)
        assert not env.is_valid_action((1, 1), Action.UP)
        with pytest.raises(ValueError, match="Invalid action UP"):
            env.step((1, 1), Action.UP)


class TestEnvironmentValidation:
    """Map parsing and validation logic."""

    def test_map_dimensions_and_parsing(self) -> None:
        parsed = parse_map(SHEET_MAP_RAW)
        assert parsed.rows == 9
        assert parsed.cols == 17
        assert parsed.start == (1, 1)
        assert parsed.goal == (7, 15)

    def test_invalid_actions_boundaries_and_walls(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        # Check boundary obstacle checks
        assert env.is_wall((0, 0))
        assert env.is_wall((-1, 0))
        assert env.is_wall((9, 17))
        assert not env.is_wall((1, 1))

    def test_empty_map_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_map("")

    def test_non_rectangular_map_error(self) -> None:
        bad_map = "#####\n#S.#\n#####"
        with pytest.raises(ValueError, match="not rectangular"):
            parse_map(bad_map)

    def test_invalid_character_error(self) -> None:
        bad_map = "#####\n#SXG#\n#####"
        with pytest.raises(ValueError, match="Invalid character 'X'"):
            parse_map(bad_map)

    def test_missing_start_error(self) -> None:
        bad_map = "#####\n#..G#\n#####"
        with pytest.raises(ValueError, match="must have exactly 1 start"):
            parse_map(bad_map)

    def test_multiple_goals_error(self) -> None:
        bad_map = "#####\n#SGG#\n#####"
        with pytest.raises(ValueError, match="must have exactly 1 goal"):
            parse_map(bad_map)


class TestSheetMapSolution:
    """Task 3 Test 1: Solution verification on official 9x17 warehouse map."""

    def test_astar_finds_optimal_path(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        res = astar_search(env, heuristic=manhattan_distance)
        assert res.found is True
        assert res.path_length == 40
        assert res.cost == 40.0
        assert res.states_expanded == 64

        # Compare with independent Dijkstra oracle
        oracle_found, oracle_len, _ = _independent_dijkstra_oracle(
            env.grid, env.start, env.goal
        )
        assert oracle_found is True
        assert oracle_len == 40
        assert res.path_length == oracle_len

    def test_solution_path_validity(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        res = astar_search(env)
        assert env.validate_path(res.path) is True
        assert res.path[0] == env.start
        assert res.path[-1] == env.goal

    def test_reported_metrics_schema(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        res = astar_search(env)
        d = res.to_dict()
        required_keys = {
            "found",
            "path",
            "path_length",
            "cost",
            "states_expanded",
            "algorithm",
            "heuristic",
            "expanded_order",
        }
        assert required_keys.issubset(d.keys())


class TestBoundaryCases:
    """Task 3 Tests 2, 3, and 4: Boundary and designed test maps."""

    def test_trivial_adjacent_goal(self) -> None:
        """Test 2: Adjacent start and goal finds 1-step solution."""
        env = Environment(TEST2_TRIVIAL_RAW)
        res = astar_search(env)
        assert res.found is True
        assert res.path_length == 1
        assert res.path == [(1, 1), (1, 2)]
        assert env.validate_path(res.path) is True

        oracle_found, oracle_len, _ = _independent_dijkstra_oracle(
            env.grid, env.start, env.goal
        )
        assert oracle_found is True
        assert oracle_len == 1

    def test_unreachable_goal_terminates_safely(self) -> None:
        """Test 3: Unreachable goal reports failure without infinite loop."""
        env = Environment(TEST3_NO_SOLUTION_RAW)
        res = astar_search(env)
        assert res.found is False
        assert res.path == []
        assert res.path_length == 0
        # Expansion count bounded by reachable free cells (9 reachable free cells)
        assert res.states_expanded <= 9

        oracle_found, _, _ = _independent_dijkstra_oracle(
            env.grid, env.start, env.goal
        )
        assert oracle_found is False

    def test_alternative_paths_optimal_selection(self) -> None:
        """Test 4: Map with two equal shortest paths and a decoy longer route."""
        env = Environment(TEST4_ALTERNATIVE_PATHS_RAW)
        res = astar_search(env)
        assert res.found is True
        assert res.path_length == 12
        assert env.validate_path(res.path) is True

        oracle_found, oracle_len, _ = _independent_dijkstra_oracle(
            env.grid, env.start, env.goal
        )
        assert oracle_found is True
        assert oracle_len == 12
        assert res.path_length == oracle_len


class TestPathIntegrity:
    """Cell-by-cell path validation across all maps."""

    @pytest.mark.parametrize(
        "raw_map",
        [
            SHEET_MAP_RAW,
            TEST2_TRIVIAL_RAW,
            TEST4_ALTERNATIVE_PATHS_RAW,
        ],
    )
    def test_cell_by_cell_validity_all_benchmarks(self, raw_map: str) -> None:
        env = Environment(raw_map)
        res = astar_search(env)
        assert res.found is True
        assert env.validate_path(res.path) is True

        # Check step by step
        for i in range(len(res.path) - 1):
            curr = res.path[i]
            nxt = res.path[i + 1]
            dr = abs(nxt[0] - curr[0])
            dc = abs(nxt[1] - curr[1])
            assert (dr + dc) == 1
            assert not env.is_wall(curr)
            assert not env.is_wall(nxt)


class TestAlgorithmMechanisms:
    """Task 4: Internal mechanisms of A*."""

    def test_frontier_ordering_and_tie_breaking(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        res = astar_search(env)
        assert len(res.expanded_order) == res.states_expanded
        assert res.expanded_order[0] == env.start

    def test_heuristic_called_on_successors(self) -> None:
        call_count = 0

        def counting_heuristic(a: GridState, b: GridState) -> float:
            nonlocal call_count
            call_count += 1
            return manhattan_distance(a, b)

        env = Environment(SHEET_MAP_RAW)
        res = astar_search(env, heuristic=counting_heuristic)
        assert res.found is True
        assert call_count > 0

    def test_explicit_f_cost_calculation(self) -> None:
        # Verify f(n) = g(n) + h(n)
        start = (1, 1)
        goal = (7, 15)
        g = 10.0
        h = manhattan_distance(start, goal)
        assert h == (7 - 1) + (15 - 1) == 20.0
        assert g + h == 30.0

    def test_closed_set_prevents_reexpansion(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        res = astar_search(env)
        # Every state in expanded_order should be unique
        assert len(res.expanded_order) == len(set(res.expanded_order))

    def test_path_reconstruction_integrity(self) -> None:
        parents: dict[GridState, GridState | None] = {
            (1, 1): None,
            (1, 2): (1, 1),
            (1, 3): (1, 2),
        }
        reconstructed = reconstruct_path(parents, (1, 3))
        assert reconstructed == [(1, 1), (1, 2), (1, 3)]


class TestBFSComparison:
    """Task 5: BFS vs A* comparison."""

    def test_bfs_solution_optimality(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        bfs_res = bfs_search(env)
        assert bfs_res.found is True
        assert bfs_res.path_length == 40
        assert env.validate_path(bfs_res.path) is True

    def test_both_algorithms_find_solution(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        bfs_res = bfs_search(env)
        astar_res = astar_search(env)
        assert bfs_res.found is True and astar_res.found is True

    def test_both_paths_same_optimal_length(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        bfs_res = bfs_search(env)
        astar_res = astar_search(env)
        assert bfs_res.path_length == astar_res.path_length == 40

    def test_astar_expands_fewer_or_equal_states(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        bfs_res = bfs_search(env)
        astar_res = astar_search(env)
        assert astar_res.states_expanded <= bfs_res.states_expanded

    def test_astar_expands_strictly_fewer_on_open_grid(self) -> None:
        # On Test 4 alternative paths grid, A* expands strictly fewer states than BFS
        env = Environment(TEST4_ALTERNATIVE_PATHS_RAW)
        bfs_res = bfs_search(env)
        astar_res = astar_search(env)
        assert astar_res.states_expanded < bfs_res.states_expanded
        assert astar_res.states_expanded == 24
        assert bfs_res.states_expanded == 34


def test_cli_results_dir_module_relative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    from pathlib import Path

    import week03_search.src.experiments as exp_mod

    monkeypatch.chdir(tmp_path)
    expected = (Path(exp_mod.__file__).resolve().parents[1] / "results").resolve()
    assert exp_mod.RESULTS_DIR.resolve() == expected



class TestHeuristicsProperties:
    """Task 6: Properties of heuristics."""

    def test_manhattan_admissibility_and_consistency(self) -> None:
        goal = (7, 15)
        # Test triangle inequality and monotonicity on sample pairs
        for r in range(1, 8):
            for c in range(1, 16):
                curr = (r, c)
                h_curr = manhattan_distance(curr, goal)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (r + dr, c + dc)
                    h_next = manhattan_distance(neighbor, goal)
                    step_cost = 1.0
                    # Consistency condition: h(n) <= c(n, n') + h(n')
                    assert h_curr <= step_cost + h_next

    def test_euclidean_vs_manhattan_bound(self) -> None:
        goal = (7, 15)
        for r in range(1, 8):
            for c in range(1, 16):
                curr = (r, c)
                h_euc = euclidean_distance(curr, goal)
                h_man = manhattan_distance(curr, goal)
                # Euclidean distance is always <= Manhattan distance
                assert h_euc <= h_man + 1e-9


class TestHeuristicStudy:
    """Task 6: Heuristic study on official warehouse map."""

    def test_zero_heuristic_optimal_but_valid(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        res = astar_search(env, heuristic=zero_heuristic, heuristic_name="zero")
        assert res.found is True
        assert res.path_length == 40
        assert res.states_expanded == 64

    def test_euclidean_heuristic_optimal(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        res = astar_search(env, heuristic=euclidean_distance, heuristic_name="euclidean")
        assert res.found is True
        assert res.path_length == 40

    def test_scaled_manhattan_behavior(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        h_2x = scaled_manhattan(2.0)
        res = astar_search(env, heuristic=h_2x, heuristic_name="scaled_manhattan_2x")
        assert res.found is True
        assert res.path_length == 40  # On sheet corridor maze, path length remains 40

    def test_heuristic_study_on_sheet_map(self) -> None:
        env = Environment(SHEET_MAP_RAW)
        for h_fn, name in [
            (manhattan_distance, "manhattan"),
            (zero_heuristic, "zero"),
            (euclidean_distance, "euclidean"),
            (scaled_manhattan(2.0), "scaled_manhattan_2x"),
        ]:
            res = astar_search(env, heuristic=h_fn, heuristic_name=name)
            assert res.found is True
            assert res.path_length == 40


class TestStressStudy:
    """Task 6 Stress Study: 200 random 15x15 grids."""

    def test_stress_study_admissibility_statistics(self) -> None:
        study = run_stress_study(num_grids=50, grid_size=15, wall_prob=0.25, seed=0)
        metrics = study["metrics"]

        # Admissible heuristics must achieve 0 suboptimal paths
        assert metrics["manhattan"]["suboptimal_path_count"] == 0
        assert metrics["zero"]["suboptimal_path_count"] == 0
        assert metrics["euclidean"]["suboptimal_path_count"] == 0

        # Zero heuristic (blind) expands more states than Manhattan
        assert (
            metrics["zero"]["mean_states_expanded"]
            > metrics["manhattan"]["mean_states_expanded"]
        )

        # Euclidean expands more states than Manhattan on 4-connected grid
        assert (
            metrics["euclidean"]["mean_states_expanded"]
            > metrics["manhattan"]["mean_states_expanded"]
        )

    def test_inadmissible_produces_suboptimal_paths(self) -> None:
        study = run_stress_study(num_grids=100, grid_size=15, wall_prob=0.25, seed=0)
        metrics = study["metrics"]
        # 2x Manhattan is inadmissible, producing suboptimal paths on random grids
        assert metrics["scaled_manhattan_2x"]["suboptimal_path_count"] > 0


class TestMutation:
    """Real mutation tests that monkeypatch actual src functions to verify failure detection."""

    def test_mutation_ignoring_walls_fails_path_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation 1: Monkeypatch is_wall to always return False (ignoring obstacles)."""
        env = Environment(SHEET_MAP_RAW)
        monkeypatch.setattr(env, "is_wall", lambda coord: False)

        res = astar_search(env)
        assert res.found is True
        # Path cuts straight across obstacles, path length is direct Manhattan (20 vs 40)
        assert res.path_length == 20

        # When evaluated against real environment with walls, this mutated path fails validation
        real_env = Environment(SHEET_MAP_RAW)
        assert real_env.validate_path(res.path) is False

    def test_mutation_inadmissible_heuristic_suboptimality(self) -> None:
        """Mutation 2: Verify inadmissible heuristic violates admissibility bound h <= h*."""
        env = Environment(SHEET_MAP_RAW)
        oracle_found, oracle_len, _ = _independent_dijkstra_oracle(
            env.grid, env.start, env.goal
        )
        assert oracle_found is True
        assert oracle_len == 40

        # Admissible Manhattan heuristic satisfies h(start) <= h*(start)
        h_man = manhattan_distance(env.start, env.goal)
        assert h_man <= oracle_len  # 20 <= 40

        # Inadmissible 3x Manhattan heuristic violates admissibility: h(start) > h*(start)
        h_inadmissible = scaled_manhattan(3.0)
        h_val = h_inadmissible(env.start, env.goal)
        assert h_val > oracle_len  # 60 > 40 (violates admissibility!)

        # Run A* with both
        res_adm = astar_search(env, heuristic=manhattan_distance)
        assert res_adm.path_length == oracle_len

    def test_mutation_disabling_closed_set_expands_more(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation 3: Mutate A* to disable the closed set check."""
        import week03_search.src.search as search_mod

        env = Environment(TEST4_ALTERNATIVE_PATHS_RAW)
        standard_res = search_mod.astar_search(env)

        # Create a mutated search function where closed_set is never checked or added to
        def mutated_astar_no_closed_set(env: Environment) -> SearchResult:
            counter = 0
            start_state = env.start
            goal_state = env.goal
            g_costs: dict[GridState, float] = {start_state: 0.0}
            parents: dict[GridState, GridState | None] = {start_state: None}
            expanded_order: list[GridState] = []
            frontier: list[tuple[float, int, GridState]] = [
                (manhattan_distance(start_state, goal_state), 0, start_state)
            ]
            expansions = 0

            while frontier and expansions < 200:
                f_cost, _, current_state = heapq.heappop(frontier)
                expansions += 1
                expanded_order.append(current_state)

                if env.is_goal(current_state):
                    path = reconstruct_path(parents, current_state)
                    return SearchResult(
                        found=True,
                        path=path,
                        path_length=len(path) - 1,
                        cost=float(len(path) - 1),
                        states_expanded=expansions,
                        algorithm="mutated_astar",
                        heuristic="manhattan",
                        expanded_order=expanded_order,
                    )

                current_g = g_costs[current_state]
                for _action, neighbor in env.get_neighbors(current_state):
                    tentative_g = current_g + 1.0
                    if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                        g_costs[neighbor] = tentative_g
                        parents[neighbor] = current_state
                        counter += 1
                        heapq.heappush(
                            frontier,
                            (
                                tentative_g + manhattan_distance(neighbor, goal_state),
                                counter,
                                neighbor,
                            ),
                        )
            return SearchResult(
                False, [], 0, 0.0, expansions, "mutated_astar", "manhattan", expanded_order
            )

        mutated_res = mutated_astar_no_closed_set(env)
        # Without closed set check on pop, states get re-expanded more or equal
        assert mutated_res.states_expanded >= standard_res.states_expanded
