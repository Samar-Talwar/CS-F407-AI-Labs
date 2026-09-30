# Week 4 – Logic / Planning: Checklist

Every requirement from `docs/lab_sheets/logic_lab_ex.pdf` mapped to deliverables.

## Task 0 – Understand the Planning Problem
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 0.1 | Initial state I | `src/domain.py::INITIAL_STATE` | `results/task0_applicability.json["initial_state"]` | `test_week04::test_initial_state` |
| 0.2 | Goal G | `src/domain.py::GOAL` | `results/task0_applicability.json["goal"]` | `test_week04::test_goal` |
| 0.3 | Actions with preconditions and effects | `src/domain.py::warehouse_actions()` | `REPORT.md §Task0` | `test_week04::test_action_definitions` |
| 0.4 | PickUp(Package,A) applicable in I | `src/planner.py::applicable()` | `results/task0_applicability.json["pickup_applicable"]` | `test_week04::test_pickup_applicable_in_initial` |
| 0.5 | Drop(Package,C) NOT applicable in I (missing preconditions) | `src/planner.py::applicable()` | `results/task0_applicability.json["drop_not_applicable"]` | `test_week04::test_drop_not_applicable_in_initial` |

## Task 1 – Construct a Plan by Hand
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 1.1 | Manual plan with state after every action (S0–S4) | `src/cli.py` | `results/manual_plan.json` | `test_week04::test_manual_plan_states` |
| 1.2 | Sheet's example sequence is INVALID (PickUp(Package,B) precondition fails) | `src/validator.py::validate_plan()` | `results/invalid_example_plan.json` | `test_week04::test_sheet_example_invalid` |

## Task 2 – Ask an LLM to Implement the Planner
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 2.1 | Prompt used with the LLM | `docs/prompt_log.md` | `REPORT.md §Task2` | Documented |
| 2.2 | Think-About-It: Preconditions → applicable() | `src/planner.py::applicable()` | `REPORT.md §Task2.TAI` | `test_week04::test_applicable_logic` |
| 2.3 | Think-About-It: Effects → apply() | `src/planner.py::apply_action()` | `REPORT.md §Task2.TAI` | `test_week04::test_apply_effects` |
| 2.4 | Think-About-It: Goal → BFS termination | `src/planner.py::bfs_plan()` | `REPORT.md §Task2.TAI` | `test_week04::test_goal_termination` |
| 2.5 | Think-About-It: BFS → deque exploration | `src/planner.py::bfs_plan()` | `REPORT.md §Task2.TAI` | `test_week04::test_bfs_explores_alternatives` |

## Task 3 – Test the Generated Planner
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 3A | Test A: Solvable (original warehouse) – plan found, valid | `src/planner.py::bfs_plan()` | `results/test_a.json` | `test_week04::test_a_solvable` |
| 3B | Test B: Impossible (no PickUp) – "No plan found" | `src/planner.py::bfs_plan()` | `results/test_b.json` | `test_week04::test_b_impossible` |
| 3C | Test C: Irrelevant actions – robot at C ≠ package at C | `src/planner.py::bfs_plan()` | `results/test_c.json` | `test_week04::test_c_irrelevant_actions` |
| 3.rec | Five fields per test: initial_state, goal, plan_found, plan, valid | all `results/test_*.json` | five keys each | `test_week04::test_result_schema` |

## Task 4 – Logic and Search
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 4.1 | Explain logic vs search roles | `REPORT.md §Task4` | — | — |
| 4.2 | Fill the blank: "If all preconditions satisfied → action is applicable" | `REPORT.md §Task4` | — | — |

## Task 5 – Can the LLM Verify Its Own Plan?
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| 5.1 | Trust executed transitions over LLM explanation, grounded in validator | `REPORT.md §Task5` | — | — |

## Reflection Questions 1–7
| # | Question | File |
|---|---|---|
| R1 | Why specify preconditions/effects before prompting? | `REPORT.md §R1` |
| R2 | Example error without precondition check | `REPORT.md §R2` |
| R3 | Why "looks reasonable" ≠ valid | `REPORT.md §R3` |
| R4 | What did the LLM contribute? | `REPORT.md §R4` (DRAFT stub) |
| R5 | What did you verify independently? | `REPORT.md §R5` (DRAFT stub) |
| R6 | Where is logical reasoning used? | `REPORT.md §R6` |
| R7 | How is planning related to search? | `REPORT.md §R7` |

## Prolog Extension (Tasks 6–8)
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| P6.1 | connected/2 facts, can_move/2 rule | `prolog/planner.pl` | `results/prolog_results.json["can_move"]` | `test_week04::test_prolog_can_move` |
| P6.2 | can_move(a,b) → true, can_move(a,c) → false | subprocess | `results/prolog_results.json` | `test_week04::test_prolog_can_move` |
| P6.3 | Questions a,b,c about can_move/connected | `REPORT.md §P6` | — | — |
| P7.1 | valid_move/2 rule | `prolog/planner.pl` | `results/prolog_results.json["valid_move"]` | `test_week04::test_prolog_valid_move` |
| P7.2 | valid_move(a,b)→T, valid_move(b,c)→T, valid_move(a,c)→F | subprocess | `results/prolog_results.json` | `test_week04::test_prolog_valid_move` |
| P7.3 | Challenge: Move(a,c) rejected by Prolog | subprocess | `results/prolog_results.json["move_ac_challenge"]` | `test_week04::test_prolog_move_ac_rejected` |
| P7.4 | Cross-check: Python plan's Move steps accepted by Prolog | `src/cli.py` | `results/prolog_results.json["cross_check"]` | `test_week04::test_prolog_cross_check` |
| P8.1 | wet_road/slippery/reduce_speed | `prolog/planner.pl` | `results/prolog_results.json["reduce_speed"]` | `test_week04::test_prolog_reduce_speed` |
| P8.2 | penguin/bird/animal chain, animal(polly) → true | `prolog/planner.pl` | `results/prolog_results.json["animal_polly"]` | `test_week04::test_prolog_animal_polly` |
| P8.3 | Logical reasoning chain explanation | `REPORT.md §P8` | — | — |

## Prolog Reflection 1–4
| # | Question | File |
|---|---|---|
| PR1 | Fact vs rule | `REPORT.md §PR1` |
| PR2 | Query ↔ KB entailment | `REPORT.md §PR2` |
| PR3 | Why verify with Prolog? | `REPORT.md §PR3` |
| PR4 | Advantage of independent verifier | `REPORT.md §PR4` |

## Submission Items (1–7)
| # | Item | File(s) |
|---|---|---|
| S1 | Planning problem specification | `REPORT.md §Task0`, `src/domain.py` |
| S2 | Manually constructed plan | `REPORT.md §Task1`, `results/manual_plan.json` |
| S3 | Prompt used with LLM | `REPORT.md §Task2`, `docs/prompt_log.md` |
| S4 | Generated Python program | `src/` package |
| S5 | Test results | `results/test_a.json`, `test_b.json`, `test_c.json` |
| S6 | Think About It answers | `REPORT.md` (§Task2.TAI, §Task4, §P7, §P8) |
| S7 | Reflection on LLM use | `REPORT.md §R4, §R5` (DRAFT stubs) |

## Negative Preconditions
| # | Requirement | Source File | Result File & Key | Test Function |
|---|---|---|---|---|
| NP1 | Synthetic problem exercising neg_pre and neg_eff | `src/domain.py` | `results/negative_preconditions.json` | `test_week04::test_negative_preconditions` |

## Quality Gates
| # | Gate | Verification | Status |
|---|---|---|---|
| QG1 | ruff check . passes | `ruff check .` | ☑ |
| QG2 | pytest -q passes (all weeks) | `pytest -q` | ☑ |
| QG3 | Fresh-clone gate | clone + ruff + pytest + CLI | ☑ |
| QG4 | Skeptical-grader audit | one pass | ☑ |
| QG5 | Conventional commit + push | `git push` | ☑ |
