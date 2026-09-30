# CS F407 Lab, Week 4 | Author: Samar Talwar | Not licensed for reuse or submission by others.
## Prompt Log Entry

**Date**: 2026-09-30  
**Week**: 04  
**Prompt**: I want to implement a simple planning agent in Python.
Represent a state as a set of logical propositions.
Each action should contain:
• a name;
• positive preconditions;
• negative preconditions;
• positive effects;
• negative effects.
An action is applicable if all of its preconditions are satisfied by the current state.
When an action is applied:
1. remove its negative effects from the state;
2. add its positive effects;
Use breadth-first search to find a sequence of actions that achieves a specified goal.
The program should also:
• detect when no plan exists;
• print the resulting sequence of actions;
• print the states reached after each action.
Explain the implementation and identify any assumptions you make.
Run the generated program on the warehouse problem.

**What was changed after review**:
- Added `missing_preconditions()` helper for Task 0 diagnostics
- Added `neg_pre` and `neg_eff` fields to Action to fully support negative preconditions and effects
- Made `applicable()` and `apply_action()` exactly match the spec (remove neg_eff then add pos_eff)
- Added independent validator that re-implements applicable/apply logic inline (no code sharing)
- Added BFS planner that returns plan, states after each action, and nodes expanded
- Added grounding of warehouse actions (Move between A-B, B-A, B-C, C-B; PickUp/Drop at A,B,C)
- Added support for negative preconditions/effects via synthetic problem (UnlockDoor/Enter)
- Added Prolog integration via subprocess calls to swipl for independent verification
- Added cross-check: every Move step in Python plan must be accepted by Prolog valid_move/2
- Added proper handling of swipl absence (visible skip reason, not silent pass)

---
# CS F407 Lab, Week 8 | Author: Samar Talwar | Not licensed for reuse or submission by others.
**Date**: 2026-10-01
**Week**: 08
**Prompt**: Implement Bayesian Networks / autoregressive Markov language models per BN_lab.pdf. Build first-order and second-order Markov models on the 6-sentence canonical dataset with <START>/<END> tokens. Use only stdlib (collections, random, math, fractions, json). Implement greedy decoding (documented alphabetical tie-break), stochastic inverse-CDF sampling with seed control, cycle detection in greedy, unseen-context handling, chain-rule joint/log-probability evaluation, and a CLI. Write CHECKLIST.md mapping every requirement. Add pytest suite with independent fractions.Fraction oracles, normalization verification, 5+ hand-checked probabilities, N=20000 sampling convergence, mutation tests (monkeypatch actual src functions), and n-gram validity. Generate 8 unrounded JSON result files.

**What was changed after review**:
- Created CHECKLIST.md mapping all 14 questions to files/tests/keys
- Implemented dataset.py (padding tokens), markov_model.py (separate .counts/.cpt, predict_next, generate with cycle detection), metrics.py (model statistics), cli.py (pipeline)
- Added tests/test_markov_models.py (15 tests: exact Fraction oracle, normalization, hand-checked values, sampling convergence, greedy determinism, n-gram validity, 3 mutation tests)
- Generated results/ (counts_*.json, cpt_*.json, normalisation_checks.json, chain_rule_probabilities.json, generation.json, comparison.json) at full precision
- Wrote README.md (architecture, quickstart, key results) and REPORT.md (Questions 1-14 with real numbers from results/, reflection stubs)
- Updated root README.md status table; confirmed ruff check . and pytest -q pass; validated via fresh clone in $env:TEMP\fresh_w8

---

# CS F407 Lab, Week 5 | Author: Samar Talwar | Not licensed for reuse or submission by others.
**Date**: 2026-10-01
**Week**: 05
**Prompt**: Implement complete Week 5 Transformer lab from scratch (TinyGPT): causal self-attention in NumPy and PyTorch, single/multi-head projection, custom MHA with non-standard H=6/d_head=2/d_model=8, Pre-LN Transformer blocks, 4x ReLU FFN, 180-step AdamW training, temperature-controlled generation, 5 JSON result artifacts, 15-test pytest with real mutation tests, README/REPORT/checklist, fresh-clone gate, push.
**What was changed after review**:
- Created CHECKLIST.md mapping all notebook sections and requirements to files/tests/keys
- Added attention.py (NumPy attention oracle + PyTorch Head/MHA/CustomMHA with non-standard projection)
- Added model.py (TinyOneTokenLM baseline, FeedForward with 4x ReLU, Pre-LN TransformerBlock, TinyGPT)
- Added dataset.py (CharTokenizer, BigramModel with Laplace smoothing, repeated canonical corpus, get_batch)
- Added train.py (AdamW optimization loop, 180 steps, cross-entropy evaluation)
- Added generate.py (autoregressive token generation loop, temperature scaling softmax(z/tau))
- Added metrics.py (analytical parameter breakdown 60,313, variance scaling check)
- Added cli.py (standalone reproducibility generating all 5 unrounded JSON result artifacts)
- Added tests/test_transformers.py (15 tests: NumPy vs PyTorch oracle, analytical parameter oracle, causal invariance, variance scaling, temperature invariants, training convergence, 3 monkeypatch mutations)
- Generated results/*.json (unrounded machine outputs)
- Wrote README.md and REPORT.md (comprehensive mathematical derivations, tables, TODO(student) stubs)
- Updated root README.md; confirmed ruff check . and pytest -q pass; validated via fresh clone in $env:TEMP\fresh_w5
