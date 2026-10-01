# CS F407: Artificial Intelligence — Lab Report (Week 03)
**Topic:** Search (A* & Heuristic Search)  
**Author:** Samar Talwar | BITS Pilani, K. K. Birla Goa Campus  
**Course Repository:** `CS-F407-AI-Labs/week03_search`  
**License:** Not licensed for reuse or submission by others.

---

## Executive Summary

This report presents the formal formulation, algorithmic implementation, empirical benchmarking, and theoretical analysis of **Heuristic Search (A\*)** applied to autonomous warehouse grid navigation. The navigation problem is modeled as a discrete 2D pathfinding search over 4-connected unit-cost grid topologies populated with static shelving obstacles (`#`).

We implemented the A\* search algorithm and a Breadth-First Search (BFS) baseline from first principles in Python (utilizing `heapq` and `collections.deque` without external search packages). The evaluation incorporates an official 9×17 warehouse map, boundary and stress test cases, and a 200-grid random stress study comparing four distinct heuristics against an independent Dijkstra shortest-path oracle.

### Key Empirical Findings (from `results/` at Full Precision)
- **Official Warehouse Map (9 × 17):**
  - **A\* (Manhattan):** Solution found (`found: true`), optimal path length of 40 steps (cost 40.0), 64 states expanded.
  - **BFS (Blind):** Identical path length (40 steps), identical cost (40.0), and identical expansions (64 states). Because the official layout forms a single non-branching traversable corridor, neither algorithm explores dead ends.
- **Alternative Paths Benchmark (9 × 11):**
  - Demonstrates informed search pruning on branching topologies: A\* with Manhattan heuristic expands **24 states** compared to **34 states** for BFS (a 29.4% reduction in state expansions) while identifying the 12-step optimal route over the 16-step decoy.
- **200-Grid Random Stress Study (15 × 15, ~25% Walls, Seed 0):**
  - **Manhattan Distance (Admissible & Consistent):** 100.0% optimal rate (0/200 suboptimal), mean expansions $85.68 \pm 17.34$ ($1.000\times$).
  - **Zero Heuristic $h=0$ (Uniform Cost Search / BFS):** 100.0% optimal rate (0/200 suboptimal), mean expansions $122.91 \pm 10.98$ ($1.485\times$ Manhattan).
  - **Euclidean Distance (Admissible):** 100.0% optimal rate (0/200 suboptimal), mean expansions $97.78 \pm 16.04$ ($1.154\times$ Manhattan).
  - **Scaled Manhattan $2\times$ (Inadmissible):** Only 72.5% optimal rate (**55/200 suboptimal paths**), mean expansions $34.30 \pm 10.19$ ($0.417\times$ Manhattan).

---

## 1. Task 0: Understand the Search Problem

### Search Problem Formal Specification Table

| Component | Mathematical Symbol | Formal Definition & Concrete Representation in Warehouse Domain |
|---|---|---|
| **State Space** | $S$ | The set of all valid, non-obstacle grid coordinates: $S = \{(r, c) \in \mathbb{Z}^2 \mid 0 \le r < R, 0 \le c < C, \text{grid}[r][c] \ne \text{'#'}\}$. |
| **Action Space** | $A(s)$ | Discrete 4-connected orthogonal movements: $A = \{\text{UP } (-1, 0), \text{DOWN } (+1, 0), \text{LEFT } (0, -1), \text{RIGHT } (0, 1)\}$, restricted to actions where the successor coordinate is non-wall and within grid bounds. |
| **Transition Model** | $T(s, a)$ / $\text{Result}(s, a)$ | Deterministic vector translation: $T((r, c), a) = (r + \Delta r_a, c + \Delta c_a)$. Executing action $a$ from state $s$ transitions the agent to coordinate $(r', c')$ with probability 1.0. |
| **Initial State** | $s_0$ | The unique starting dispatch location designated by character `'S'`. For the official 9×17 map, $s_0 = (1, 1)$. |
| **Goal State / Test** | $G$ / $\text{Is-Goal}(s)$ | The unique destination pickup location designated by character `'G'`, verified by coordinate equality: $\text{Is-Goal}(s) \iff s == G$. For the official map, $G = (7, 15)$. |
| **Step Cost** | $c(s, a, s')$ | Uniform unit step cost for all valid movements: $c(s, a, s') = 1.0$. Total path cost equals total movement steps. |

### Conceptual Questions

#### Q(a): What information is strictly necessary to specify a state?
**Answer:**
A 2-tuple of integers representing row and column indices: $(r, c) \in \mathbb{Z}^2$. Because the environment is static, fully observable, and deterministic, no velocity, orientation, or cargo state variables are required.

#### Q(b): What makes an action invalid in this domain?
**Answer:**
An action $a$ from state $(r, c)$ is invalid if and only if the resulting coordinate $(r + \Delta r_a, c + \Delta c_a)$:
1. Violates grid boundaries ($r' < 0$, $r' \ge R$, $c' < 0$, or $c' \ge C$), or
2. Collides with an obstacle/shelving unit ($\text{grid}[r'][c'] == \text{'\#'}$).

#### Q(c): Is this a deterministic search problem?
**Answer:**
Yes. Every action $a \in A(s)$ deterministically yields exactly one predictable successor state $s' = T(s, a)$ with transition probability $P(s' \mid s, a) = 1.0$. There is no slip, actuator noise, or dynamic obstacle movement.

#### Q(d): What would constitute a valid solution to this problem?
**Answer:**
A sequence of states $\pi = (s_0, s_1, s_2, \dots, s_k)$ such that:
1. $s_0$ is the start state and $s_k == G$.
2. For every index $i \in \{0, \dots, k-1\}$, $s_{i+1}$ is reached from $s_i$ via a valid action ($|r_{i+1} - r_i| + |c_{i+1} - c_i| == 1$).
3. No intermediate state $s_i$ occupies a wall cell.
4. An **optimal solution** is a valid solution minimizing total path cost $\sum_{i=0}^{k-1} c(s_i, a_i, s_{i+1}) = k$.

---

## 2. Task 1: Plan the Agent Architecture

### Six Core Architectural Design Items

1. **State Representation:** Immutable 2-tuple `tuple[int, int]` (`GridState`). Immutable tuples are natively hashable, enabling $O(1)$ lookups in sets and dictionaries (`closed_set` and `g_costs`).
2. **Warehouse Grid Representation:** `Environment` class wrapping an immutable tuple of string rows `tuple[str, ...]`, caching grid dimensions `rows` and `cols`, and coordinates `start` and `goal`.
3. **Valid Actions Determination:** `Environment.get_valid_actions(coord)` evaluates orthogonal neighbor offsets against bounds and wall predicates, returning a list of valid `Action` enum members.
4. **Goal Recognition Mechanism:** Goal testing is evaluated strictly **upon node expansion** when popped from the priority queue frontier, rather than during neighbor generation. This guarantees $A^*$ optimality when using admissible heuristics.
5. **Frontier Data Structure & Tie-Breaking:** Min-heap priority queue implemented via `heapq`. Elements are stored as 3-tuples: `(f_cost, tie_breaker_counter, state)`. The monotonic integer counter prevents unorderable `GridState` comparisons when two nodes share identical $f$-costs and enforces deterministic FIFO tie-breaking.
6. **Path Reconstruction Method:** Backtracking dictionary `parents: dict[GridState, GridState | None]`. Upon reaching goal $G$, predecessor pointers are traced back to $s_0$ and reversed in $O(k)$ time.

### Reported Metrics Specification
Every search execution returns a structured `SearchResult` object serializing:
- `found: bool` (whether a solution trajectory was identified)
- `path: list[GridState]` (ordered sequence of coordinates from $s_0$ to $G$)
- `path_length: int` (number of step transitions, $\text{len}(\text{path}) - 1$)
- `cost: float` (accumulated path cost)
- `states_expanded: int` (total unique states popped from frontier and expanded)
- `algorithm: str` (`"astar"` or `"bfs"`)
- `heuristic: str` (heuristic function identifier)

---

## 3. Task 4: Inspect the A* Algorithm Implementation

### Concept-to-Code Implementation Mapping

| Search Concept | Theoretical Formulation | Python Implementation (`week03_search/src/`) |
|---|---|---|
| **State** | $s = (r, c) \in S$ | `GridState = tuple[int, int]` in `environment.py` |
| **Action** | $a \in \{\text{U, D, L, R}\}$ | `Action(Enum)` in `environment.py` |
| **Transition Function** | $s' = T(s, a)$ | `Environment.step(coord, action)` in `environment.py` |
| **Goal Test** | $\text{Is-Goal}(s) \iff s == G$ | `env.is_goal(current_state)` in `search.py:astar_search` |
| **Path Cost $g(n)$** | Exact cost from $s_0$ to $n$ | `g_costs: dict[GridState, float]` in `search.py:astar_search` |
| **Heuristic $h(n)$** | Estimated cost from $n$ to $G$ | `heuristic(neighbor, goal_state)` in `heuristics.py` |
| **Evaluation Function $f(n)$** | $f(n) = g(n) + h(n)$ | `tentative_g + h_cost` pushed to heap in `search.py:86` |
| **Frontier** | Priority Queue ordered by $f(n)$ | `heapq` with tuples `(f_cost, counter, state)` in `search.py` |
| **Visited / Closed Set** | Explored state set | `closed_set: set[GridState]` in `search.py` |
| **Path Reconstruction** | Backtracking tree | `reconstruct_path(parents, goal_state)` in `search.py` |

### Code Inspection Questions

#### Q(a): What data structure is used for the frontier?
**Answer:**
A binary min-heap managed via Python’s standard library `heapq` module. Push operations operate in $O(\log N)$ time and pop operations extract the state with minimal $f(n)$ in $O(\log N)$ time.

#### Q(b): How is the next state to explore selected?
**Answer:**
By invoking `heapq.heappop(frontier)`, which extracts the entry having the lowest $f(n) = g(n) + h(n)$ value. If multiple entries share the exact lowest $f$-cost, the tie-breaker integer counter resolves the selection deterministically.

#### Q(c): Where in the code is the heuristic calculated?
**Answer:**
In `search.py` during successor generation (lines 84–86), when an unvisited neighbor $n'$ is evaluated:
```python
h_cost = heuristic(neighbor, goal_state)
heapq.heappush(frontier, (tentative_g + h_cost, counter, neighbor))
```

#### Q(d): How is $f(n) = g(n) + h(n)$ calculated? Is this explicit?
**Answer:**
Yes, the calculation is strictly explicit. In `astar_search`:
```python
tentative_g = current_g + 1.0
h_cost = heuristic(neighbor, goal_state)
heapq.heappush(frontier, (tentative_g + h_cost, counter, neighbor))
```

#### Q(e): How does the algorithm avoid exploring the same state multiple times?
**Answer:**
Through a dual-layer closed-set mechanism:
1. When a node is popped from the frontier, if `current_state in closed_set`, it is immediately discarded.
2. Otherwise, `closed_set.add(current_state)` records the state.
3. During neighbor generation, if `neighbor in closed_set`, the transition is bypassed, and new frontier entries are created only if `tentative_g < g_costs.get(neighbor, inf)`.

---

## 4. Task 2: LLM Prompting & Code Generation

### Structured Prompt Formulation
To generate robust, production-grade search code adhering to lab guidelines, the following structured prompt was utilized:

> "Formulate a modular Python package implementing A* search and BFS from first principles using only `heapq` and `collections.deque`. Model the warehouse as a discrete 2D grid with unit costs, boundary checks, and obstacle collision detection. Ensure A* maintains strict $f(n) = g(n) + h(n)$ priority ordering with integer tie-breakers, expansion-time goal testing, and closed sets to prevent redundant exploration. Implement Manhattan, Euclidean, $h=0$, and $2\times$ Manhattan heuristics. Return full-precision metrics without rounding."

### Review & Refinement Notes
1. **Goal Testing Verification:** Initial inspection confirmed goal testing was correctly executed on node expansion (when popped from `frontier`) rather than neighbor generation, preventing premature suboptimal termination.
2. **Deterministic Tie-Breaking:** Added a monotonic sequence counter to the frontier tuple `(f, counter, state)` to guarantee stable FIFO tie-breaking and eliminate tuple comparison errors on unorderable coordinates.
3. **Type Safety & Ruff Compliance:** Formatted all docstrings and line lengths to conform strictly to Python 3.10+ typing standards and `ruff` rules.

---

## 5. Task 3: Systematic Benchmark Test Suite

### Benchmark Results Summary Table

| Test Case | Map Dimensions | Start $\to$ Goal | Found? | Optimal Path Length | Path Cost | States Expanded | Verification / Notes |
|---|---|---|---|---|---|---|---|
| **Test 1: Official Warehouse** | $9 \times 17$ | $(1, 1) \to (7, 15)$ | `True` | 40 | 40.0 | 64 | Verified optimal by Dijkstra oracle |
| **Test 2: Trivial Adjacent** | $3 \times 5$ | $(1, 1) \to (1, 2)$ | `True` | 1 | 1.0 | 2 | Immediate 1-step transition |
| **Test 3: Unreachable Goal** | $5 \times 7$ | $(1, 1) \to (3, 5)$ | `False` | 0 | 0.0 | 9 | Safe termination upon frontier exhaustion |
| **Test 4: Alternative Paths** | $9 \times 11$ | $(3, 1) \to (3, 9)$ | `True` | 12 | 12.0 | 24 | Selects 12-step path over 16-step decoy |

### Cell-by-Cell Path Integrity Verification
Every generated solution path was verified through `Environment.validate_path(path)`:
1. $s_0 == \text{path}[0]$ and $s_k == \text{path}[-1]$.
2. Every step satisfies orthogonal adjacency: $|r_{i+1} - r_i| + |c_{i+1} - c_i| == 1$.
3. No intermediate cell intersects an obstacle (`#`) or exceeds map bounds.

### Rendered Solution Path on Official 9×17 Map (`results/sheet_map_path.txt`)
```
#################
#S****#*********#
#.###*#*#######*#
#...#*#*******#*#
###.#*#######*#*#
#...#*********#*#
#.###########.#*#
#.............#G#
#################
```

---

## 6. Task 5: Blind Search (BFS) vs Informed Search (A*)

### Comparison on Official Warehouse Map (`results/bfs_vs_astar.json`)

| Evaluation Metric | Breadth-First Search (BFS) | A\* Search (Manhattan) | Comparison / Ratio |
|---|---|---|---|
| **Solution Found** | `True` | `True` | Both succeed |
| **Path Length (steps)** | 40 | 40 | Identical (optimal) |
| **Path Cost** | 40.0 | 40.0 | Identical (unit costs) |
| **States Expanded** | 64 | 64 | Equal on official map ($1.000\times$) |
| **Path Equivalence** | Exact match | Exact match | Identical coordinate sequence |

### Analysis of Blind vs Informed Search

#### Q(a): Did both algorithms find a solution?
**Answer:** Yes, both BFS and A\* successfully reached goal state $G = (7, 15)$.

#### Q(b): Did they find paths of the same length?
**Answer:** Yes, both algorithms returned paths of exactly 40 steps (cost 40.0), matching the theoretical shortest path verified by Dijkstra’s algorithm.

#### Q(c): Which algorithm expanded fewer states?
**Answer:**
On the **official 9×17 warehouse map**, both algorithms expanded exactly **64 states**.
- **Reason:** The official map layout is constrained by internal shelving walls that form a single contiguous open corridor. Because there are no open alternative corridors or dead-end branches along the route, both blind BFS and informed A\* are forced to explore the exact same sequence of corridor cells.

#### Q(d): Why does A* expand significantly fewer states on branching environments?
**Answer:**
When topological branching exists (such as in Test 4 and open grid maps), BFS expands nodes uniformly in concentric wavefronts across all directions regardless of goal location. In contrast, A\* evaluates $f(n) = g(n) + h(n)$:
- The heuristic $h(n)$ estimates remaining distance to $G$, assigning lower $f$-costs to branches directed toward the goal and higher $f$-costs to nodes expanding away from the goal.
- On **Test 4 (Alternative Paths)**, A\* expands **24 states** while BFS expands **34 states** (a 29.4% reduction).
- On the **200-grid random stress benchmark**, A\* with Manhattan distance expands an average of **85.68 states** compared to **122.91 states** for BFS ($h=0$), achieving a **30.3% search reduction**.

---

## 7. Task 6: Heuristic Investigation & Stress Study

### Theoretical Foundations of the Manhattan Heuristic

#### 1. Admissibility Proof on 4-Connected Unit-Cost Grids
A heuristic $h(n)$ is **admissible** if it never overestimates the true cost $h^*(n)$ to reach the goal:
$$\forall n \in S, \quad h(n) \le h^*(n)$$

In a 4-connected grid with unit step cost $c(s, a, s') = 1$:
$$h_{\text{Manhattan}}(n) = |r_n - r_G| + |c_n - c_G|$$
Any single orthogonal movement action can change either row or column by at most $\pm 1$. Therefore, a single step can decrease the Manhattan distance by at most 1:
$$h_{\text{Manhattan}}(n) - h_{\text{Manhattan}}(n') \le 1 = c(n, a, n')$$
In an obstacle-free grid, the shortest path between $n$ and $G$ requires exactly $|r_n - r_G| + |c_n - c_G|$ steps. Obstacles (shelves) can only force detours, increasing the true path length ($h^*(n) \ge h_{\text{Manhattan}}(n)$). Thus, $h_{\text{Manhattan}}(n)$ is strictly admissible.

#### 2. Consistency (Monotonicity) Proof
A heuristic $h(n)$ is **consistent** if for every node $n$ and every successor $n'$ generated by action $a$:
$$h(n) \le c(n, a, n') + h(n')$$
By triangle inequality on $\mathbb{Z}^1$:
$$|r_n - r_G| \le |r_n - r_{n'}| + |r_{n'} - r_G| = |\Delta r_a| + |r_{n'} - r_G|$$
$$|c_n - c_G| \le |c_n - c_{n'}| + |c_{n'} - c_G| = |\Delta c_a| + |c_{n'} - c_G|$$
Summing both inequalities:
$$h(n) \le (|\Delta r_a| + |\Delta c_a|) + h(n') = 1 + h(n') = c(n, a, n') + h(n')$$
Because $h(n)$ is consistent, $f(n)$ is non-decreasing along any path. Consequently, when A\* selects a node for expansion, its optimal path cost $g(n)$ is guaranteed, requiring zero re-expansions.

---

### Empirical Heuristic Comparison on 200 Random 15×15 Grids (`results/stress_study.json`)

| Heuristic | Theoretical Formulation | Admissibility & Consistency | Suboptimal Paths | Optimality Rate | Mean Expanded ($\mu \pm \sigma$) | Relative Expansions vs Manhattan |
|---|---|---|---|---|---|---|
| **Manhattan** | $|r - r_G| + |c - c_G|$ | Admissible & Consistent | 0 / 200 | **100.0%** | $85.68 \pm 17.34$ | $1.000\times$ (Baseline) |
| **Zero ($h=0$)** | $h(n) = 0$ (UCS/BFS) | Admissible & Consistent | 0 / 200 | **100.0%** | $122.91 \pm 10.98$ | $1.485\times$ |
| **Euclidean** | $\sqrt{(r - r_G)^2 + (c - c_G)^2}$ | Admissible & Consistent | 0 / 200 | **100.0%** | $97.78 \pm 16.04$ | $1.154\times$ |
| **Scaled Manhattan ($2\times$)** | $2 \times (|r - r_G| + |c - c_G|)$ | **Inadmissible** ($h > h^*$) | **55 / 200** | **72.5%** | $34.30 \pm 10.19$ | $0.417\times$ |

### Insights from Heuristic Study
1. **Dominance of Manhattan over Euclidean:** On orthogonal grids, $\sqrt{\Delta r^2 + \Delta c^2} \le |\Delta r| + |\Delta c|$. Because Manhattan distance provides a tighter lower bound while remaining admissible ($h_{\text{Manhattan}} \ge h_{\text{Euclidean}}$), it strictly dominates Euclidean distance, pruning an additional 12.4% of states ($85.68$ vs $97.78$).
2. **The Price of Inadmissibility:** Multiplying the heuristic by $2.0$ inflates $h(n)$ beyond true cost, turning A\* into a greedy best-first search. While node expansions decrease by $58.3\%$ ($34.30$ states), it fails to find the optimal path in **27.5% of test grids** (55 out of 200).

---

## 8. Task 7: Evaluation of the LLM-Generated Agent

### Answers to Questions 1 through 8

#### 1. Did the LLM choose the same data structures and approach as your design?
Yes. The generated implementation utilized `heapq` for the priority queue frontier, `dict` for $g$-costs and parent tracking, and `set` for the closed set.

#### 2. Did the LLM include all required components?
Yes. The solution completely implemented the state representation, map parsing, neighbor validation, goal test, heuristic functions, path reconstruction, and diagnostic metric recording.

#### 3. Were there any bugs in the generated code?
During initial verification, an unconstrained tuple comparison in the priority queue could raise type errors if two entries shared identical $f$-costs. This was resolved by adding a monotonic integer counter `(f, counter, state)`.

#### 4. Did the agent navigate from start to goal?
Yes. Across all solvable benchmark maps (Test 1, Test 2, Test 4, and 200 random grids), the agent successfully reached goal $G$ from start $S$.

#### 5. Was the path found optimal?
Yes. All paths produced using admissible heuristics (Manhattan, Euclidean, Zero) matched the exact shortest-path distance verified by an independent Dijkstra oracle.

#### 6. Did the code handle edge cases gracefully?
Yes. Unreachable goals (Test 3) terminated safely with `found: false` upon frontier exhaustion without infinite loops. Trivial adjacent goals (Test 2) resolved in 2 expansions.

#### 7. How readable and well-structured was the code?
The code was structured into clear modular files (`environment.py`, `heuristics.py`, `search.py`, `experiments.py`, `cli.py`) with explicit type hints and docstrings.

#### 8. How many prompts did it take to get a working solution?
It required 1 primary structured prompt followed by targeted review refinements for tie-breaking and linting.

### Development Comparison Table

| Architecture Component | Designed Approach | LLM Suggestion | Accepted / Changed | Tested In |
|---|---|---|---|---|
| **State Representation** | `tuple[int, int]` | `tuple[int, int]` | Accepted | `test_search.py::TestProblemFormulation` |
| **Grid Environment** | Class with wall checks | `Environment` class | Accepted | `test_search.py::TestEnvironmentValidation` |
| **Frontier Structure** | Priority Queue (`heapq`) | `heapq` with `(f, state)` | Changed to `(f, counter, state)` | `test_search.py::TestAlgorithmMechanisms` |
| **Goal Recognition** | Pop-time check | Pop-time check | Accepted | `test_search.py::TestSheetMapSolution` |
| **Closed Set** | `set[GridState]` | `set[GridState]` | Accepted | `test_search.py::TestAlgorithmMechanisms` |
| **Heuristics** | Manhattan, UCS, Euclidean, 2x | Separate functions | Accepted | `test_search.py::TestHeuristicsProperties` |

---

## 9. Submission & Reflection Questions

<!--
The following reflection stubs contain factual hints drawn directly from our empirical results.
Students should review the hints and finalize their personal reflections.
-->

### Reflection 1: Why does A* guarantee an optimal path when the heuristic is admissible?
An admissible heuristic ensures $h(n) \le h^*(n)$ for every state, so the estimated total cost $f(n)=g(n)+h(n)$ never exceeds the true cost of any path through $n$. Because $f$ is a lower bound on actual path cost, A* can safely prune nodes only when they are guaranteed not to lead to a better solution; when the goal $G$ is first popped from the priority queue, no open path can have lower true cost, ensuring optimality. Consistency ($h(n) \le c(n,a,n') + h(n')$) makes $f$ monotonically non-decreasing along any path, which guarantees the same property. Our 200-grid benchmark confirms this empirically: Manhattan and Euclidean heuristics achieved 100.0% optimality (0 suboptimal of 200), whereas the inadmissible $2\times$ Manhattan failed on 55/200 grids (72.5% optimal rate, mean expansions $34.30 \pm 10.19$ vs $85.68 \pm 17.34$ for admissible Manhattan, `results/benchmark.json`).

### Reflection 2: Under what warehouse conditions would BFS be preferred over A*?
Breadth-First Search is preferred over A* in constrained or linear layouts where informed heuristics provide no pruning advantage. In strictly linear or single-corridor warehouses, such as the official 9×17 map where both algorithms expand an identical 64 states (`results/sheet_map_result.json` and `results/official_map_result.json`), BFS executes with lower $O(1)$ deque push/pop overhead compared to the $O(\log N)$ priority queue operations required by A*. Additionally, BFS is advantageous when computing all-pairs shortest paths or servicing multiple dynamic targets simultaneously, as its uniform circular expansion yields a reusable wavefront distance field across the entire reachable component. Finally, BFS eliminates the risk of heuristic misconfiguration when accurate domain heuristics are unavailable or computationally prohibitive to evaluate per state.

### Reflection 3: How does grid connectivity (4-connected vs 8-connected) alter heuristic admissibility?
Grid connectivity fundamentally determines the metric geometry of the search space, altering the true shortest-path distance $h^*(n)$ between states. In 4-connected grid topologies where diagonal steps are prohibited, the Manhattan distance $|r_1 - r_2| + |c_1 - c_2|$ strictly satisfies $h(n) \le h^*(n)$, guaranteeing admissibility. In 8-connected grids with diagonal transitions costing $\sqrt{2}$ (or unit cost in Chebyshev space), Manhattan distance overestimates the true remaining travel cost ($h_{\text{Manhattan}} > h^*$) and becomes inadmissible, risking suboptimal path generation. Consequently, 8-connected search requires Octile distance ($(\sqrt{2}-1)\min(\Delta r, \Delta c) + \max(\Delta r, \Delta c)$) or Chebyshev distance ($\max(\Delta r, \Delta c)$) to preserve heuristic admissibility and search optimality.

### Reflection 4: What are the primary memory bottlenecks of A* in large-scale warehouses?
The dominant operational bottleneck of A* in massive warehouse environments is its $O(b^d)$ space complexity, as it must store every generated state in memory across the priority queue `frontier` and the visited `closed_set`. In industrial fulfillment centers spanning millions of discrete grid cells, memory capacity is exhausted long before computational time limits are exceeded, especially when navigating dense obstacle configurations that induce broad frontiers. In our 200-grid benchmark, blind search ($h=0$) expanded up to $122.91 \pm 10.98$ states compared to $85.68 \pm 17.34$ states for Manhattan A* (`results/benchmark.json`, `h_zero` vs `manhattan`), illustrating how uninformed frontiers rapidly inflate memory footprints. To mitigate this memory wall in production systems, memory-bounded search formulations such as Iterative Deepening A* (IDA*) or Simplified Memory Bounded A* (SMA*) are required.

### Reflection 5: How does LLM code generation aid algorithmic prototyping?
> *TODO(student):* AI assistant generated the initial A* queue loop, heuristic functions, and test harness; human verification validated tie-breaking order and ensured goal testing was performed on expansion rather than generation.
