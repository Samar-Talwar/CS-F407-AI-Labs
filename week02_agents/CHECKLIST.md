# Week 2 – Goal-Based Agent: Checklist

Every requirement from `docs/lab_sheets/agents_lab.pdf` and `docs/PROJECT_RULES.md` mapped to deliverables.

## Task 1 – Understanding the Problem (5 questions + Think-About-It)
| # | Requirement (sheet Q) | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 1.1 | What type of intelligent agent architecture? | `src/agent.py::GoalBasedAgent` | `REPORT.md §1.Q1` | `test_agent.py::TestAgentComponentsVisibility::test_agent_components_and_actions` |
| 1.2 | Key components needed? | `src/agent.py` (`AgentState`, `GoalTest`, `Action`, `SearchPlanner`) | `REPORT.md §1.Q2`, `results/agent_diagram.png` | `test_agent.py::TestAgentComponentsVisibility::test_agent_components_and_actions` |
| 1.3 | How environment and agent state represented? | `src/environment.py::Environment`, `src/agent.py::AgentState` | `REPORT.md §1.Q3` | `test_agent.py::TestSheetMapSolution::test_path_step_validity_and_obstacle_avoidance` |
| 1.4 | What search algorithm should decision component use & why? | `src/agent.py::bfs_search`, `src/agent.py::SearchPlanner` | `REPORT.md §1.Q4`, `results/sheet_map_result.json["algorithm"]` | `test_agent.py::TestSheetMapSolution::test_optimality_against_independent_dijkstra_oracle` |
| 1.5 | What data structures needed to track path? | `src/agent.py::bfs_search` (`deque`, `visited`, `parents`) | `REPORT.md §1.Q5` | `test_agent.py::TestSheetMapSolution::test_path_exists_and_reaches_goal` |
| TAI-1 | Think-About-It: Twice as large warehouse scaling? | `src/scaling.py::run_scaling_experiment` | `results/scaling.json["benchmarks"]`, `REPORT.md §1.Think-About-It` | `test_agent.py::TestScalingAndAlgorithms::test_scale_map_2x_properties` |

## Task 2 – Designing the Agent
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 2.1 | Block diagram of agent architecture | `src/diagram.py::generate_agent_diagram` | `results/agent_diagram.png` (300 DPI) | Diagram generated & referenced in `REPORT.md §2` |
| 2.2 | Component roles visible in code & docstrings | `src/agent.py` (`AgentState`, `GoalTest`, `Action`, `SearchPlanner`, `GoalBasedAgent`) | `REPORT.md §2` | `test_agent.py::TestAgentComponentsVisibility::test_agent_components_and_actions` |

## Task 3 – Prompt Engineering & Implementation (4 questions + prompt bullets)
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 3.1 | Prompt recorded with review changes | `docs/prompt_log.md` | Row `2026-09-30 \| 2` | Documented in `docs/prompt_log.md` |
| 3.2 | Did LLM code work on first attempt? | `REPORT.md §3.Q1` | Factual evaluation stub in `REPORT.md` | Verified via test iterations |
| 3.3 | How could prompt be improved? | `REPORT.md §3.Q2` | Structural & failure-mode tips in `REPORT.md` | Documented in `REPORT.md` |
| 3.4 | Why was specific algorithm chosen? | `src/agent.py::GoalBasedAgent.explain_algorithm` | `results/sheet_map_result.json["algorithm"]`, `REPORT.md §3.Q3` | `test_agent.py::TestAgentComponentsVisibility::test_agent_components_and_actions` |
| 3.5 | Summary of results (coordinates, steps, expanded, runtime) | `src/cli.py` | `results/sheet_map_result.json`, `results/sheet_map_path.txt` | `test_agent.py::TestSheetMapSolution::test_optimality_against_independent_dijkstra_oracle` |
| 3.6 | Clear "No path exists" message with exit-safe execution | `src/agent.py::bfs_search` | `results/no_path_result.json["found"] = false` | `test_agent.py::TestUnsolvableAndBoundaryCases::test_unsolvable_map_exit_safe` |
| 3.7 | Map validation (rectangular, single S/G, valid chars) | `src/environment.py::parse_map` | Clear `ValueError` on malformed inputs | `test_agent.py::TestMapValidation` (7 tests) |
| 3.8 | Real mutation test (breaking actual module) | `src/environment.py::Environment.is_wall` | Verified invalid path cuts through wall | `test_agent.py::TestRealMutation::test_mutation_ignoring_walls_fails_wall_avoidance` |
| 3.9 | Independent oracle verification (no self-comparison) | `tests/test_agent.py::_independent_dijkstra_oracle` | `oracle_length == 20 == result.path_length` | `test_agent.py::TestSheetMapSolution::test_optimality_against_independent_dijkstra_oracle` |
