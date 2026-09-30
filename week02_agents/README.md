# Week 02 — Goal-Based Agent for Warehouse Navigation

**Author:** Samar Talwar | CS F407 | Not licensed for reuse or submission by others.

**How to run**

```bash
# Run full pytest suite for this week (from repo root, no pip install needed)
pytest week02_agents/ -q

# Regenerate all machine outputs (results/*.json, *.txt, *.png)
python -m week02_agents.src.cli

# Inspect architecture diagram
open week02_agents/results/agent_diagram.png
```

**Quick validation**

```bash
python -c "from week02_agents.src.agent import GoalBasedAgent; print('Import OK')"
python -c "from week02_agents.src.environment import parse_map, SHEET_MAP_STRING; env = parse_map(SHEET_MAP_STRING); print('Map parsed. Start:', env.start, 'Goal:', env.goal)"
```

**Project layout**

```
week02_agents/
├── src/                 # importable modules + CLI entry
│   ├── __init__.py
│   ├── environment.py   # map parsing / validation (rectangular, single S/G, symbols # . S G)
│   ├── agent.py         # Action, AgentState, GoalTest, SearchPlanner, GoalBasedAgent, BFS/DFS/A*
│   ├── scaling.py       # 2x scale experiment and runtime measurements
│   ├── diagram.py       # matplotlib architecture block diagram
│   └── cli.py           # reproduces every result file
├── tests/
│   └── test_agent.py    # 18 tests: oracle verification, boundary, mutation, scaling, visibility
├── results/             # machine-generated outputs at FULL precision (no rounding in files)
│   ├── sheet_map_result.json
│   ├── sheet_map_path.txt
│   ├── no_path_result.json
│   ├── trivial_cases.json
│   ├── scaling.json
│   └── agent_diagram.png
├── CHECKLIST.md         # maps every task / question to file + result key + test
├── REPORT.md            # submitted answers using data from results/
└── README.md            # this file
```

**Results summary (from results/ at full precision)**

| Experiment | Algorithm | Path Length | Nodes Expanded | Runtime (s) |
|---|---|---|---|---|
| Sheet map (7x21) | BFS | 20 | 59 | 0.000365 |
| Sheet map (7x21) | A* | 20 | 23 | 0.000163 |
| Unsolvable (8x7) | BFS | -1 (none) | 9 | ~0.0001 |
| Trivial: start==goal | BFS | 0 | 0 | 0 |
| Trivial: adjacent | BFS | 1 | 3 | ~0.0001 |
| Trivial: diamond (2 routes) | BFS | 4 | 8 | ~0.0001 |
| Scaled 2x (14x42) | BFS | 40 | 239 | 0.000967 |
| Scaled 2x (14x42) | A* | 40 | 56 | 0.000541 |

**Notes**
- BFS is the agent's primary planner (guarantees shortest path on uniform-cost grids).
- A* and DFS are comparison baselines only (Task 2 design / Think-About-It scaling).
- All JSON results contain unrounded floats (e.g., `runtime_seconds` preserved to full float64).
- No open-source license is added (NOTICE.md governs reuse); every source file starts with the required authorship header.
