# Week 03 — Search (A* & Heuristic Search)

**Author:** Samar Talwar | CS F407 | Not licensed for reuse or submission by others.

## How to Run

```bash
# Run full pytest suite for this week (from repo root, no pip install needed)
pytest week03_search/tests/ -q

# Regenerate all machine outputs (results/*.json, *.txt) and view benchmark summary
python -m week03_search.src.cli

# Quick import and verification checks
python -c "from week03_search.src.search import astar_search, bfs_search; print('Search module OK')"
python -c "from week03_search.src.environment import Environment, SHEET_MAP_RAW; env = Environment(SHEET_MAP_RAW); print('Map parsed. Start:', env.start, 'Goal:', env.goal)"
```

## Project Layout

```
week03_search/
├── src/                 # Importable modules + CLI entrypoint
│   ├── __init__.py
│   ├── environment.py   # Map parsing, validation, GridState, Action, Environment, path validation
│   ├── heuristics.py    # Manhattan, Euclidean, Zero (h=0/UCS), and Scaled Manhattan distance
│   ├── search.py        # A* (heapq priority queue, tie-breaker, closed set) & BFS (deque)
│   ├── experiments.py   # Benchmark pipelines: Tests 1-4, BFS vs A*, Heuristics, 200-grid stress
│   └── cli.py           # CLI entrypoint to display theoretical proofs and reproduce all results
├── tests/
│   ├── __init__.py
│   └── test_search.py   # 41 tests: independent Dijkstra oracle, cell-by-cell path checks, mutation tests
├── results/             # Machine-generated outputs at FULL precision (unrounded JSON files)
│   ├── original_warehouse.json
│   ├── sheet_map_path.txt
│   ├── test2_trivial.json
│   ├── test3_no_solution.json
│   ├── test4_alternatives.json
│   ├── bfs_vs_astar.json
│   ├── heuristic_study.json
│   └── stress_study.json
├── CHECKLIST.md         # Exhaustive mapping of all lab sheet tasks to source, results, and tests
├── REPORT.md            # Complete laboratory report covering Tasks 0-7, concept tables, and stubs
└── README.md            # This documentation file
```

## Summary of Empirical Results (from `results/` at Full Precision)

### 1. Systematic Benchmark Suite (Tasks 3 & 5)
- **Test 1 — Official Warehouse Map (9 × 17):**
  - **Status:** Path found successfully (`found: true`).
  - **Path Length:** 40 steps (41 coordinate states).
  - **Path Cost:** 40.0 (unit step cost $c=1$).
  - **States Expanded:** 64 states.
  - **BFS Comparison:** Identical path length (40 steps), identical path cost (40.0), and identical expansions (64 states) due to the single-corridor warehouse layout with zero open dead-end branchings.
- **Test 2 — Trivial Adjacent Goal (3 × 5):**
  - **Status:** Path found (`found: true`), path length 1 step, 2 states expanded.
- **Test 3 — Unreachable Goal (5 × 7):**
  - **Status:** Safely terminated with no solution (`found: false`), path length 0, 9 reachable states expanded without infinite loop.
- **Test 4 — Alternative Paths Grid (9 × 11):**
  - **Status:** Path found (`found: true`), optimal path length 12 steps (avoiding 16-step decoy route), 24 states expanded by A* (vs 34 states expanded by BFS).

### 2. Heuristic Study & 200-Grid Random Stress Benchmark (Task 6)
Evaluated across **200 random 15 × 15 solvable grids** (~25% wall density, fixed seed 0) against an independent Dijkstra shortest-path oracle:

| Heuristic | Admissible? | Suboptimal Paths | Optimal Rate | Mean States Expanded | Relative Expansion vs Manhattan |
|---|---|---|---|---|---|
| **Manhattan Distance** | Yes (Consistent) | 0 / 200 | 100.0% | 85.68 ± 17.34 | 1.000× (Baseline) |
| **Zero Heuristic ($h=0$)** | Yes (UCS / Blind) | 0 / 200 | 100.0% | 122.91 ± 10.98 | 1.485× |
| **Euclidean Distance** | Yes (Admissible) | 0 / 200 | 100.0% | 97.78 ± 16.04 | 1.154× |
| **Scaled Manhattan ($2\times$)** | No (Inadmissible) | 55 / 200 | 72.5% | 34.30 ± 10.19 | 0.417× |

### Key Takeaways
1. **Admissibility Guarantees Optimality:** Manhattan, Euclidean, and Zero heuristics achieved a 100.0% optimality rate across all 200 random grids with zero suboptimal routes.
2. **Heuristic Dominance:** Manhattan heuristic dominates Euclidean distance on 4-connected unit grids ($h_{\text{Manhattan}}(n) \ge h_{\text{Euclidean}}(n)$ everywhere), expanding ~13.4% fewer nodes (85.68 vs 97.78) while retaining provable optimality.
3. **Inadmissible Heuristic Trade-off:** $2\times \text{Manhattan}$ achieves greedy speedup (expanding only 41.7% as many nodes as standard Manhattan), but sacrifices optimality, returning suboptimal paths on 27.5% (55/200) of branching grids.
