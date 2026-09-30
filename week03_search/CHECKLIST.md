# Week 3 – Search (A* & Heuristic Search): Checklist

Every requirement from `docs/lab_sheets/search_lab_ex.pdf` and `docs/PROJECT_RULES.md` mapped to deliverables.

## Task 0 – Understand the Search Problem
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 0.1 | Search problem formulation table (S, A, T, s0, G, c) | `src/environment.py`, `src/search.py` | `REPORT.md §1.Table-Task0` | `test_search.py::TestProblemFormulation::test_problem_specification_components` |
| 0.2 | Q(a): Information necessary to specify a state | `src/environment.py::GridState` | `REPORT.md §1.Q(a)` | `test_search.py::TestProblemFormulation::test_state_representation` |
| 0.3 | Q(b): What makes an action invalid | `src/environment.py::Environment.is_valid_action` | `REPORT.md §1.Q(b)` | `test_search.py::TestEnvironmentValidation::test_invalid_actions_boundaries_and_walls` |
| 0.4 | Q(c): Is this a deterministic search problem? | `src/environment.py::Environment.step` | `REPORT.md §1.Q(c)` | `test_search.py::TestProblemFormulation::test_deterministic_transitions` |
| 0.5 | Q(d): What would constitute a solution? | `src/search.py::SearchResult` | `REPORT.md §1.Q(d)` | `test_search.py::TestSheetMapSolution::test_solution_path_validity` |

## Task 1 – Plan the Agent
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 1.1 | Design item 1: State representation in Python | `src/environment.py::GridState` | `REPORT.md §2.Item1` | `test_search.py::TestProblemFormulation::test_state_representation` |
| 1.2 | Design item 2: Warehouse representation in Python | `src/environment.py::Environment` | `REPORT.md §2.Item2` | `test_search.py::TestEnvironmentValidation::test_map_dimensions_and_parsing` |
| 1.3 | Design item 3: Valid actions determination | `src/environment.py::Environment.get_valid_actions` | `REPORT.md §2.Item3` | `test_search.py::TestEnvironmentValidation::test_invalid_actions_boundaries_and_walls` |
| 1.4 | Design item 4: Goal recognition mechanism | `src/search.py::astar_search` (goal test on expansion) | `REPORT.md §2.Item4` | `test_search.py::TestSheetMapSolution::test_astar_finds_optimal_path` |
| 1.5 | Design item 5: Frontier stored information | `src/search.py::astar_search` (`(f, tie_breaker, state)`) | `REPORT.md §2.Item5` | `test_search.py::TestAlgorithmMechanisms::test_frontier_ordering_and_tie_breaking` |
| 1.6 | Design item 6: Path reconstruction method | `src/search.py::reconstruct_path` (`parent` dict) | `REPORT.md §2.Item6` | `test_search.py::TestAlgorithmMechanisms::test_path_reconstruction_integrity` |
| 1.7 | Report fields (found, path, path_length, states_expanded) | `src/search.py::SearchResult` | `results/original_warehouse.json` | `test_search.py::TestSheetMapSolution::test_reported_metrics_schema` |

## Task 2 – LLM Prompting & Code Generation
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 2.1 | Prompt log entry with review notes | `docs/prompt_log.md` | Row `2026-09-30 \| 3` | Documented in `docs/prompt_log.md` |
| 2.2 | Prompt covering states, frontier, g/h/f, closed set, path, metrics | `REPORT.md §4` | `REPORT.md §4` | Verified in generated implementation |

## Task 3 – Test the Generated Program
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 3.1 | Test 1: Original warehouse (9x17 map) | `src/experiments.py::run_sheet_map_test` | `results/original_warehouse.json`, `results/sheet_map_path.txt` | `test_search.py::TestSheetMapSolution::test_astar_finds_optimal_path` |
| 3.2 | Test 2: Trivial case (`#####/#SG##/#####`, adjacent goal) | `src/experiments.py::run_test2_trivial` | `results/test2_trivial.json` | `test_search.py::TestBoundaryCases::test_trivial_adjacent_goal` |
| 3.3 | Test 3: No solution (7-wide unreachable goal) | `src/experiments.py::run_test3_no_solution` | `results/test3_no_solution.json` | `test_search.py::TestBoundaryCases::test_unreachable_goal_terminates_safely` |
| 3.4 | Test 4: Alternative paths (equal paths + decoy longer route) | `src/experiments.py::run_test4_alternatives` | `results/test4_alternatives.json` | `test_search.py::TestBoundaryCases::test_alternative_paths_optimal_selection` |
| 3.5 | Cell-by-cell path validation on all test maps | `src/environment.py::Environment.validate_path` | `results/*.json["path"]` | `test_search.py::TestPathIntegrity::test_cell_by_cell_validity_all_benchmarks` |

## Task 4 – Inspect the A* Algorithm
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 4.1 | Concept-to-code table (State, Action, Transition, Goal test, g, h, f, Frontier, Visited, Path reconstruction) | `src/search.py`, `src/environment.py`, `src/heuristics.py` | `REPORT.md §3.Concept-Table` | `test_search.py::TestAlgorithmMechanisms` (suite) |
| 4.2 | Q(a): Data structure used for frontier | `src/search.py::astar_search` (`heapq` min-heap) | `REPORT.md §3.Q(a)` | `test_search.py::TestAlgorithmMechanisms::test_frontier_ordering_and_tie_breaking` |
| 4.3 | Q(b): Next state selection mechanism | `src/search.py::astar_search` (`heappop` min f) | `REPORT.md §3.Q(b)` | `test_search.py::TestAlgorithmMechanisms::test_frontier_ordering_and_tie_breaking` |
| 4.4 | Q(c): Where heuristic is calculated | `src/search.py::astar_search` (on successor generation) | `REPORT.md §3.Q(c)` | `test_search.py::TestAlgorithmMechanisms::test_heuristic_called_on_successors` |
| 4.5 | Q(d): Explicit calculation of $f(n) = g(n) + h(n)$ | `src/search.py::astar_search` (`f_cost = g_cost + h_cost`) | `REPORT.md §3.Q(d)` | `test_search.py::TestAlgorithmMechanisms::test_explicit_f_cost_calculation` |
| 4.6 | Q(e): Prevention of repeated exploration | `src/search.py::astar_search` (`closed_set` & `g_costs`) | `REPORT.md §3.Q(e)` | `test_search.py::TestAlgorithmMechanisms::test_closed_set_prevents_reexpansion` |

## Task 5 – Compare A* with Blind Search (BFS)
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 5.1 | BFS implementation with deque and closed set | `src/search.py::bfs_search` | `results/bfs_vs_astar.json["bfs"]` | `test_search.py::TestBFSComparison::test_bfs_solution_optimality` |
| 5.2 | Comparison table (found, path length, states expanded) | `src/experiments.py::run_bfs_vs_astar` | `results/bfs_vs_astar.json` | `test_search.py::TestBFSComparison::test_bfs_vs_astar_metrics` |
| 5.3 | Q(a): Did both algorithms find a solution? | `src/experiments.py` | `results/bfs_vs_astar.json`, `REPORT.md §6.Q(a)` | `test_search.py::TestBFSComparison::test_both_algorithms_find_solution` |
| 5.4 | Q(b): Did they find paths of the same length? | `src/experiments.py` | `results/bfs_vs_astar.json`, `REPORT.md §6.Q(b)` | `test_search.py::TestBFSComparison::test_both_paths_same_optimal_length` |
| 5.5 | Q(c): Which algorithm expanded fewer states? | `src/experiments.py` | `results/bfs_vs_astar.json`, `REPORT.md §6.Q(c)` | `test_search.py::TestBFSComparison::test_astar_expands_fewer_or_equal_states` |
| 5.6 | Q(d): Why might A* expand fewer states? (Informed vs blind) | `src/search.py` | `REPORT.md §6.Q(d)` | Documented in `REPORT.md` |

## Task 6 – Investigate the Heuristic
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 6.1 | Explanation of Manhattan admissibility & consistency | `src/heuristics.py::manhattan_distance` | `REPORT.md §7.Explanation` | `test_search.py::TestHeuristicsProperties::test_manhattan_admissibility_and_consistency` |
| 6.2 | Heuristic 1: $h(n) = 0$ (Zero / UCS) | `src/heuristics.py::zero_heuristic` | `results/heuristic_study.json["h_zero"]` | `test_search.py::TestHeuristicStudy::test_zero_heuristic_optimal_but_more_expanded` |
| 6.3 | Heuristic 2: $h(n) = \text{Euclidean}$ | `src/heuristics.py::euclidean_distance` | `results/heuristic_study.json["h_euclidean"]` | `test_search.py::TestHeuristicStudy::test_euclidean_heuristic_optimal` |
| 6.4 | Heuristic 3: $h(n) = 2 \times \text{Manhattan}$ (Aggressive/Inadmissible) | `src/heuristics.py::scaled_manhattan` | `results/heuristic_study.json["h_2x_manhattan"]` | `test_search.py::TestHeuristicStudy::test_scaled_manhattan_behavior` |
| 6.5 | Benchmark comparison on original warehouse map | `src/experiments.py::run_heuristic_study` | `results/heuristic_study.json` | `test_search.py::TestHeuristicStudy::test_heuristic_study_on_sheet_map` |
| 6.6 | 200-grid random stress study (seed 0, 15x15, ~25% walls) | `src/experiments.py::run_stress_study` | `results/stress_study.json` | `test_search.py::TestStressStudy::test_stress_study_admissibility_statistics` |
| 6.7 | Suboptimal path counts & relative expansion analysis | `src/experiments.py` | `results/stress_study.json["metrics"]` | `test_search.py::TestStressStudy::test_inadmissible_produces_suboptimal_paths` |

## Task 7 – Evaluate the LLM-Generated Agent
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 7.1 | Questions 1-8 answered factually | `REPORT.md §8` | `REPORT.md §8` | Reviewed against test suite |
| 7.2 | Evaluation table (Designed / Suggested / Accepted / Changed / Tested) | `REPORT.md §8.Table` | `REPORT.md §8.Table` | Grounded in actual development facts |

## Submission Items (1 to 8) & Final Reflection (1 to 5)
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| S.1 | 1. Problem formulation | `src/environment.py`, `src/search.py` | `REPORT.md §1` | `test_search.py::TestProblemFormulation` |
| S.2 | 2. Agent design | `src/environment.py`, `src/search.py` | `REPORT.md §2` | `test_search.py::TestAlgorithmMechanisms` |
| S.3 | 3. Final Python program | `src/` modular package | `src/*.py` | Full test suite execution |
| S.4 | 4. Prompts used | `docs/prompt_log.md` | `REPORT.md §4` | Logged and referenced |
| S.5 | 5. Test results | `src/experiments.py` | `results/*.json`, `REPORT.md §5` | `test_search.py` all test cases |
| S.6 | 6. BFS/A* comparison | `src/experiments.py` | `results/bfs_vs_astar.json`, `REPORT.md §6` | `test_search.py::TestBFSComparison` |
| S.7 | 7. Heuristic investigation | `src/experiments.py`, `src/heuristics.py` | `results/heuristic_study.json`, `results/stress_study.json`, `REPORT.md §7` | `test_search.py::TestHeuristicStudy`, `test_search.py::TestStressStudy` |
| S.8 | 8. Reflection answers | `REPORT.md §8`, `REPORT.md §9` | `REPORT.md §8`, `REPORT.md §9` | Factual stubs for student completion |

## Quality Gates & Verification
| # | Gate | Verification Command | Status |
|---|---|---|---|
| QG.1 | Independent Dijkstra Oracle | `test_search.py::_independent_dijkstra_oracle` comparing BFS, A*, heuristics | Verified |
| QG.2 | Real Mutation Tests (monkeypatching actual `src` functions) | `test_search.py::TestMutation` (3 real mutations) | Verified |
| QG.3 | Map Validation & Parsing Tests | `test_search.py::TestMapValidation` | Verified |
| QG.4 | Full Precision JSON Results (no rounding in JSON files) | `results/*.json` | Verified |
| QG.5 | Fresh Clone Verification | `git clone . $env:TEMP\fresh_w3` + ruff + pytest + CLI | To execute |
