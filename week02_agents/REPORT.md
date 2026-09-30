# CS F407: Artificial Intelligence — Lab Report (Week 02)
**Topic:** Goal-Based Agent for Warehouse Navigation  
**Author:** Samar Talwar | BITS Pilani, K. K. Birla Goa Campus  
**Course Repository:** `CS-F407-AI-Labs/week02_agents`  
**License:** Not licensed for reuse or submission by others.

---

## Executive Summary

This report documents the architectural design, implementation, empirical evaluation, and theoretical analysis of a **Goal-Based Intelligent Agent** deployed for autonomous warehouse pathfinding. The agent operates in a discrete 2D grid environment populated with shelving obstacles (`#`), navigates from designated start location `S` to pickup goal `G`, and strictly guarantees collision-free, optimal navigation.

### Key Results Summary (from `results/` at full precision)
- **Official Warehouse Map (7 × 21):**
  - **Status:** Path found successfully (`found: true`).
  - **Path Length (Steps):** 20 steps (21 coordinate waypoints).
  - **Nodes Expanded (BFS):** 59 nodes.
  - **Runtime:** 0.000370 s.
  - **Optimality Verification:** 20 steps, confirmed identical by an independent Dijkstra shortest-path oracle.
- **Unsolvable Map Benchmark (6 × 7):**
  - **Status:** Gracefully handled (`found: false`), path length -1, 9 nodes expanded, exit-safe termination without infinite looping or uncaught exceptions.
- **Scaling Analysis (2× Scale Grid, 14 × 42):**
  - **BFS:** Path length 40 steps, 239 nodes expanded (4.05× expansion ratio).
  - **DFS:** Path length 142 steps, 286 nodes expanded (5.02× expansion ratio, highly suboptimal).
  - **A\* (Manhattan):** Path length 40 steps, 56 nodes expanded (2.43× expansion ratio, 76.6% fewer expansions than BFS).

---

## 1. Task 1: Conceptual Questions & Theoretical Formulation

### Question 1: What type of intelligent agent architecture is appropriate for this navigation task?
**Answer:**
A **Goal-Based Agent** (Russell & Norvig, Chapter 2) is the appropriate architecture for warehouse grid navigation.
- **Why Simple Reflex Agents Fail:** A simple reflex agent selects actions based solely on current sensory percepts (e.g., condition-action rules like *"if front is wall, turn right"*). In a maze-like warehouse with cul-de-sacs and concave obstacle configurations, a reflex agent cannot maintain history or plan future sequences, inevitably succumbing to infinite oscillation or dead-end traps.
- **Why Model-Based Reflex Agents Are Insufficient:** While a model-based reflex agent maintains internal state to track unobserved world aspects, it still lacks an explicit objective to evaluate which state transitions lead toward the warehouse goal.
- **Why Goal-Based Agents Succeed:** A goal-based agent couples internal world representation with an **explicit goal specification** (`position == goal`) and a **search planner**. It deliberatively simulates action sequences to select a trajectory that achieves the target state.
- **Why Utility-Based Agents Are Not Strictly Necessary:** Utility-based agents evaluate continuous preference trade-offs among multiple conflicting goals. For single-target destination reaching with uniform movement cost, goal satisfaction is binary, making the goal-based formulation optimal and computationally efficient.

---

### Question 2: What are the key components needed to implement this agent?
**Answer:**
The agent architecture consists of five core components:
1. **Sensors / Percepts:** Reads current agent grid coordinate $(r, c)$ from the environment.
2. **Internal State Representation (`AgentState`):** Tracks the agent's current position, topological map memory, and closed set of explored coordinates.
3. **Goal Specification & Goal Test (`GoalTest`):** Explicit predicate `is_goal(pos)` returning `True` if and only if $\text{pos} = G$.
4. **Actions & Actuators (`Action`, `apply_action`):** Four discrete orthogonal transitions: $\text{Up } (-1, 0)$, $\text{Down } (+1, 0)$, $\text{Left } (0, -1)$, and $\text{Right } (0, 1)$, bounded by boundary and obstacle collision checks.
5. **Decision Component (`SearchPlanner`):** Graph search planner (Breadth-First Search) that explores state transitions and constructs an optimal collision-free action sequence.

---

### Question 3: How should the warehouse environment and the agent's state be represented?
**Answer:**
- **Environment Representation:** Represented as a discrete 2D immutable grid matrix of size $H \times W$ (tuples of strings):
  - `'#'` denotes non-traversable obstacle boundaries and shelving units.
  - `'.'` denotes traversable aisle corridors.
  - `'S'` denotes unique starting dispatch location.
  - `'G'` denotes unique destination goal location.
- **Agent State Representation:** Represented as an immutable tuple $(r, c) \in \{0, \dots, H-1\} \times \{0, \dots, W-1\}$. State transitions are defined deterministically by $s' = s + \Delta_a$ where $\Delta_a$ is the action offset, valid iff $0 \le r' < H$, $0 \le c' < W$, and $\text{grid}[r'][c'] \ne \text{'\#'}$.

---

### Question 4: What search algorithm should the decision-making component use, and why?
**Answer:**
The decision-making component uses **Breadth-First Search (BFS)**.
- **Uniform Cost Optimality:** On an unweighted 4-connected grid graph where every orthogonal movement step incurs identical unit cost ($c(s, a, s') = 1$), BFS is mathematically guaranteed to return a path of minimal total length (optimal step count).
- **Completeness:** On finite graphs with branching factor $b \le 4$, BFS is complete: if a traversable path exists, BFS is guaranteed to terminate and return it; if no path exists, it exhausts the finite reachable component and terminates safely.
- **Time and Space Complexity:** Time complexity is $O(V + E) = O(b^d)$ and space complexity is $O(V) = O(b^d)$, where $V \le H \times W$ and $d$ is the goal depth. Because the warehouse grid has bounded dimensions ($7 \times 21 = 147$ total states), the entire state space easily fits in memory.

---

### Question 5: What data structures are needed to track the agent's path from start to goal?
**Answer:**
Three essential data structures are employed:
1. **Frontier / Open Queue:** A First-In-First-Out (FIFO) queue (`collections.deque[tuple[int, int]]`) to expand state nodes in non-decreasing order of depth.
2. **Explored Set / Closed Set:** A hash set (`set[tuple[int, int]]`) containing visited state coordinates to prevent redundant re-expansions and avoid infinite cycles.
3. **Parent Pointers (`came_from` / `predecessors`):** A hash map (`dict[tuple[int, int], tuple[int, int] | None]`) mapping each newly discovered child coordinate to its immediate predecessor state. Once the goal node is dequeued, the path is reconstructed by traversing parent pointers backward from $G$ to $S$ and reversing the resulting sequence.

---

### Think-About-It: What would change in the agent's decision-making component if the warehouse was twice as large?
**Answer (supported by empirical data from `results/scaling.json`):**

When the warehouse dimensions double in both height and width ($2\times$ scale: $7 \times 21 \to 14 \times 42$, state space growing $4\times$ from 147 to 588 cells):

1. **State Space and Path Length Growth:**
   - On the $2\times$ scaled map, the optimal path length doubles linearly from **20 steps to 40 steps**.
2. **Exponential / Super-Linear Growth of Uninformed BFS:**
   - In unguided BFS, the frontier expands uniformly in all directions. Node expansions increased from **59 nodes to 239 nodes** (an expansion ratio of **4.05×**), and runtime increased from **0.00037 s to 0.00097 s** (2.65×).
   - In large industrial warehouses ($1000 \times 1000 = 10^6$ cells), BFS memory requirements ($O(b^d)$) quickly become prohibitive.
3. **Transition to Informed Heuristic Search (A\*):**
   - In the $2\times$ experiment, **A\* Search** with an admissible Manhattan distance heuristic $h(n) = |r_n - r_G| + |c_n - c_G|$ required only **56 node expansions** (vs 239 for BFS — a **76.6% reduction in search effort** while guaranteeing identical 40-step optimal path length).
   - **Conclusion:** As warehouse dimensions scale up, the agent architecture remains structurally identical (same environment, state, and goal components), but the `SearchPlanner` decision component must transition from uninformed BFS to informed heuristic search ($A^*$ with Manhattan heuristic or Hierarchical Pathfinding $HPA^*$) to prune irrelevant state exploration.

```
Scaling Comparison Table (Empirical Data from results/scaling.json):
Algorithm  | Original (7x21) Exp. | Scaled 2x (14x42) Exp. | Expansion Ratio | Path Length (Orig -> 2x)
-----------|----------------------|------------------------|-----------------|-------------------------
BFS        | 59 nodes             | 239 nodes              | 4.05x           | 20 -> 40 (Optimal)
DFS        | 57 nodes             | 286 nodes              | 5.02x           | 36 -> 142 (Suboptimal)
A*         | 23 nodes             | 56 nodes               | 2.43x           | 20 -> 40 (Optimal)
```

---

## 2. Task 2: Agent Architecture Design & Block Diagram

### Architectural Breakdown
```
 +-----------------------------------------------------------------------------+
 |                               GOAL-BASED AGENT                              |
 |                                                                             |
 |  +---------------------+        +--------------------+                      |
 |  |  Sensors / Percept  |------->| Agent Internal     |                      |
 |  |  Current Pos (r, c) | update | State (AgentState) |                      |
 |  +---------------------+        +--------------------+                      |
 |             ^                              |                                |
 |             |                              | Current State                  |
 |             |                              v                                |
 |             |                   +--------------------+                      |
 |             |                   | Decision Component |<-----+               |
 |             |                   |  (SearchPlanner)   |      | Target Goal   |
 |             |                   +--------------------+      | is_goal == T  |
 |             |                              |                |               |
 |             |                              | Action Plan    |               |
 |             |                              v                |               |
 |             |                   +--------------------+  +-----------------+ |
 |             |                   | Actuators /        |  | Goal Test       | |
 |             |                   | Actions (Up/Dn/L/R)|  | (GoalTest)      | |
 |             |                   +--------------------+  +-----------------+ |
 |             |                              |                                |
 +-------------|------------------------------|--------------------------------+
               | Percept                      | Execute Action
               |                              v
 +-----------------------------------------------------------------------------+
 |                       ENVIRONMENT (Warehouse 2D Grid)                       |
 |  Grid Matrix: Shelving Units ('#'), Corridors ('.'), Start ('S'), Goal ('G')|
 +-----------------------------------------------------------------------------+
```

### Component Details
1. **`Environment` (`week02_agents/src/environment.py`):**
   - Encapsulates grid dimensions ($H, W$), wall locations, and designated start/goal coordinates.
   - Provides safe query method `is_valid_move(pos) -> bool` that verifies coordinates are within bounds and not occupied by shelving obstacles.
2. **`AgentState` (`week02_agents/src/agent.py`):**
   - Maintains agent current coordinate $s = (r, c)$ and internal map awareness.
3. **`GoalTest` (`week02_agents/src/agent.py`):**
   - Validates whether any proposed coordinate matches $G$: `goal_test(pos) == (pos == env.goal)`.
4. **`Action` (`week02_agents/src/agent.py`):**
   - Strict enumeration `Action.UP`, `Action.DOWN`, `Action.LEFT`, `Action.RIGHT`.
   - `apply_action(pos, action, env)` executes transitions with collision rejection.
5. **`SearchPlanner` (`week02_agents/src/agent.py`):**
   - Decision component running graph BFS to produce a `SearchResult` containing path coordinates, total steps, expansion count, and runtime.

*Architecture block diagram generated at `results/agent_diagram.png` using matplotlib.*

---

## 3. Task 3: Implementation, Prompt Strategy, & Analysis

### Suggested Prompt Structure & Implementation Strategy
To generate clean, production-grade agent software using an AI assistant, the prompt was structured into five explicit functional requirements:
1. **Domain Definition & Map Format:** Provided the exact 7 × 21 ASCII grid, specifying symbols (`#`, `.`, `S`, `G`) and coordinate convention $(row, col)$.
2. **Architectural Roles:** Requested explicit Python classes matching Russell & Norvig's goal-based agent paradigm (`AgentState`, `GoalTest`, `Action`, `SearchPlanner`, `GoalBasedAgent`).
3. **Algorithmic Invariants:** Specified BFS with closed set and parent pointers for unit-cost shortest-path optimality.
4. **Robustness & Edge-Case Handling:** Required rectangular map validation, checks for duplicate/missing terminals, and exit-safe execution on unsolvable maps.
5. **Comparative Baselines & Measurement:** Requested instrumentation for path length, nodes expanded, and comparison with A* and DFS.

---

### Task 3 Questions & Answers

#### Question 1: Did the LLM-generated code work on the first attempt?
<!-- DRAFT - rewrite in own words -->
**DRAFT - Student Experience Stub (Grounded in Facts):**
*The initial implementation generated the core BFS queue and graph search logic accurately. However, minor refinements were required during the first test execution:*
- *Strict Map Parsing:* The initial parser lacked explicit validation for ragged lines and invalid characters, which was subsequently added to reject malformed input cleanly.
- *Action Representation:* The original draft passed raw coordinate tuples directly; this was refactored into a formal `Action(Enum)` with explicit delta properties to strictly conform to agent architecture design requirements.
- *Mutation Scope:* During unit testing, global monkeypatching of `Environment.is_wall` required preserving the unmutated class method reference to independently evaluate the mutated agent's output path against the real map.

---

#### Question 2: How could you improve the prompt for better results?
<!-- DRAFT - rewrite in own words -->
**DRAFT - Student Experience Stub (Grounded in Facts):**
*Prompt quality can be significantly enhanced by providing concrete structural specifications rather than high-level descriptions:*
- *Explicit Class Hierarchy:* Pre-defining exact class signatures (`GoalBasedAgent(env)`, `SearchPlanner.plan(env, start, goal) -> SearchResult`) eliminates ambiguity in naming and interface contracts.
- *Failure Mode Specifications:* Explicitly prompting for negative testing requirements (e.g., *"return SearchResult(found=False, path_length=-1) when no path exists instead of throwing IndexError"*) prevents unhandled edge-case crashes.
- *Comparative Baseline Demands:* Explicitly requesting baseline algorithms (DFS, A*) in the initial prompt rather than as a subsequent iteration saves time and ensures consistent data structures across comparisons.

---

#### Question 3: Why was the specific search algorithm chosen?
**Answer:**
Breadth-First Search (BFS) was selected as the primary decision engine because:
1. **Exact Optimality on Unit Graphs:** Every movement between adjacent grid cells has identical step cost ($g(n) = \text{depth}(n)$). On graphs with uniform edge costs, BFS is guaranteed to find the shortest path.
2. **Minimal Algorithmic Overhead:** Unlike Dijkstra or A*, which maintain $O(\log N)$ priority queues (heaps), BFS operates with $O(1)$ amortized push/pop operations using a standard FIFO deque.
3. **Guaranteed Termination:** On finite, closed grids, the closed set prevents infinite loops and guarantees termination in $O(|V|)$ operations.

---

#### Question 4: Summary of Results on Official Warehouse Map
**Path Map Visualization (`results/sheet_map_path.txt`):**
```
#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

**Step-by-Step Path Coordinates (20 steps, 21 waypoints):**
1. `(1, 1)` [Start]
2. `(1, 2)`
3. `(1, 3)`
4. `(1, 4)`
5. `(2, 4)` [Dips down to bypass shelving obstacle at (1, 5)-(1, 6)]
6. `(2, 5)`
7. `(2, 6)`
8. `(2, 7)`
9. `(1, 7)` [Returns to upper corridor]
10. `(1, 8)`
11. `(1, 9)`
12. `(1, 10)`
13. `(1, 11)`
14. `(1, 12)`
15. `(1, 13)`
16. `(1, 14)`
17. `(1, 15)`
18. `(1, 16)`
19. `(1, 17)`
20. `(1, 18)`
21. `(1, 19)` [Goal reached]

**Performance Metrics:**
- **Path Length:** 20 steps
- **Nodes Expanded:** 59
- **Runtime:** 0.000370 s
- **Independent Dijkstra Oracle Match:** Exactly 20 steps.

---

## 4. Boundary, Robustness, and Mutation Test Evidence

All test cases are implemented in `week02_agents/tests/test_agent.py` and pass 100%:

1. **Independent Oracle Verification:**
   - Compared BFS output against `_independent_dijkstra_oracle` written independently using priority queue graph adjacency.
   - Result: Both yield identical 20-step optimal path.
2. **Unsolvable Map (`results/no_path_result.json`):**
   - Tested on disconnected grid where goal is surrounded by walls.
   - Result: Agent cleanly returns `found: false`, `path: []`, `path_length: -1`, expanding 9 nodes without crashing.
3. **Boundary Cases (`results/trivial_cases.json`):**
   - *Start Equals Goal:* Returns path `[(1, 1)]`, length 0, 0 expanded nodes.
   - *Goal Directly Adjacent:* Returns length 1, 3 expanded nodes.
   - *Diamond Equal Paths:* Correctly identifies optimal 4-step path among symmetric alternatives.
4. **Strict Format Validation:**
   - Missing start, missing goal, multiple starts, multiple goals, ragged rows, and invalid characters are caught and raise descriptive `ValueError` exceptions.
5. **Real Mutation Testing:**
   - Monkeypatched `Environment.is_wall` to return `False`.
   - The mutated agent generated an invalid 18-step straight-line path passing through the wall at `(1, 6)`.
   - The test verified that this mutated path is caught and rejected by the unmutated obstacle checker.

---

## 5. Lab Checklist Verification Mapping

| Requirement | Implementation File | Results File & Key | Test Verification |
|---|---|---|---|
| Map Parsing & Strict Validation | `environment.py:parse_map` | `sheet_map_result.json` | `test_agent.py::TestMapValidation` |
| Goal-Based Agent Components | `agent.py:GoalBasedAgent` | `sheet_map_result.json` | `test_agent.py::TestAgentComponentsVisibility` |
| BFS Shortest Path Search | `agent.py:bfs_search` | `sheet_map_result.json["path_length"]` = 20 | `test_agent.py::test_optimality_against_independent_dijkstra_oracle` |
| No-Path Exit-Safe Behavior | `agent.py:GoalBasedAgent.decide` | `no_path_result.json["found"]` = false | `test_agent.py::test_unsolvable_map_exit_safe` |
| Trivial & Boundary Cases | `agent.py:bfs_search` | `trivial_cases.json` | `test_agent.py::TestUnsolvableAndBoundaryCases` |
| 2x Scaling & Baselines (DFS/A*) | `scaling.py:run_scaling_experiment` | `scaling.json["benchmarks"]` | `test_agent.py::TestScalingAndAlgorithms` |
| Architecture Block Diagram | `diagram.py:generate_agent_diagram` | `results/agent_diagram.png` | Verified image output at 300 DPI |
| Real Mutation Testing | `tests/test_agent.py` | N/A (Test suite) | `test_agent.py::test_mutation_ignoring_walls_fails_wall_avoidance` |
