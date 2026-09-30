# Week 8 Bayesian Networks & Autoregressive Models: Checklist

This checklist tracks every task, theoretical question, test, measurement, comparison, and deliverable required by `docs/lab_sheets/BN_lab.pdf` and `docs/PROJECT_RULES.md`.

## 1. Core Model Implementation & Constraints

- [x] Standard library ONLY in `week08_bayesian_networks/src/` (no numpy, torch, sklearn, nltk).
  - *Satisfied by:* `week08_bayesian_networks/src/markov_model.py`, `dataset.py`, `metrics.py`, `cli.py`.
- [x] Author header on all source files (`# CS F407 Lab, Week 8 | Author: Samar Talwar | Not licensed for reuse or submission by others.`).
  - *Satisfied by:* All `.py` files in `src/` and `tests/`.
- [x] Canonical 6-sentence dataset tokenization with `<START>` and `<END>` padding.
  - *Satisfied by:* `week08_bayesian_networks/src/dataset.py:CANONICAL_CORPUS`, `tokenize_corpus()`.
- [x] First-order Markov model $P(w_t \mid w_{t-1})$ with separate counts and normalized CPT.
  - *Satisfied by:* `week08_bayesian_networks/src/markov_model.py:FirstOrderMarkovModel`.
- [x] Second-order Markov model $P(w_t \mid w_{t-2}, w_{t-1})$ with explicit 2-token context and `<START>` padding.
  - *Satisfied by:* `week08_bayesian_networks/src/markov_model.py:SecondOrderMarkovModel`.
- [x] Generation support: Greedy (argmax with alphabetical tie-break) and Sampling (`random.Random(seed)` inverse-CDF).
  - *Satisfied by:* `FirstOrderMarkovModel.generate()`, `SecondOrderMarkovModel.generate()`.
- [x] Cycle detection in greedy decoding (halts infinite loops and flags repeated states).
  - *Satisfied by:* `FirstOrderMarkovModel.generate()`, `SecondOrderMarkovModel.generate()`.
- [x] Unseen context handling (returns None / explicit "unseen", never crashes or invents probability).
  - *Satisfied by:* `get_cpt()`, `predict_next()`, `sentence_probability()`.
- [x] Chain-rule factorisation and sentence log-probabilities computation.
  - *Satisfied by:* `sentence_probability()`, `results/chain_rule_probabilities.json`.
- [x] CLI entry point regenerating all results files and printing summaries.
  - *Satisfied by:* `python -m week08_bayesian_networks.src.cli`.

---

## 2. Lab Sheet Questions (1–14)

- [x] **Question 1:** Why is chain-rule factorisation useful for generating text?
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 1).
- [x] **Question 2:** What independence assumption is made by the first-order Markov network?
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 2), $P(X_t \mid X_1, \dots, X_{t-1}) = P(X_t \mid X_{t-1})$.
- [x] **Question 3:** Construct conditional distributions for `the`, `cat`, `dog`, `sat`, `ran` and identify zero-probability transitions.
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 3), `results/cpt_first_order.json`.
- [x] **Question 4:** Where in the program are the transition counts stored?
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 4), `markov_model.py:self.counts`.
- [x] **Question 5:** Where is $P(X_t \mid X_{t-1})$ computed?
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 5), `markov_model.py:train()` / `_normalize_counts()`.
- [x] **Question 6:** How does the program choose the next word? (Greedy vs Sampling).
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 6), `markov_model.py:predict_next()`.
- [x] **Question 7:** What happens if the program encounters an unseen context?
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 7), `markov_model.py:predict_next()`.
- [x] **Question 8:** If a context total is 0.87, what does this indicate about the implementation?
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 8), `results/normalisation_checks.json`.
- [x] **Question 9:** Model predictions vs human linguistic expectations.
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 9).
- [x] **Question 10:** Greedy vs Sampling generation comparison (variation & diversity).
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 10), `results/generation.json`.
- [x] **Question 11:** Differences between first-order and second-order models (graph structure, CPT, context, data requirement).
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 11), `results/comparison.json`.
- [x] **Question 12:** Why increasing context improves prediction but increases data sparsity (curse of dimensionality / CPT explosion).
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 12).
- [x] **Question 13:** Why Approach B (probabilistic specification) is preferable to naive prompting.
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 13).
- [x] **Question 14:** What thinking of language models as Bayesian networks adds (dependencies, factorisation, principled generation, invariants).
  - *File & Key:* `week08_bayesian_networks/REPORT.md` (Section: Question 14).

---

## 3. Results Files & Keys (`week08_bayesian_networks/results/`)

- [x] `counts_first_order.json`: Full bigram count dictionary.
- [x] `cpt_first_order.json`: Full bigram conditional probability table.
- [x] `counts_second_order.json`: Full trigram count dictionary.
- [x] `cpt_second_order.json`: Full trigram conditional probability table.
- [x] `normalisation_checks.json`: Per-context sum and max deviation from 1.0 for both models.
- [x] `chain_rule_probabilities.json`: Step-by-step factors, joint probabilities, and log-probabilities for all 6 training sentences under both models.
- [x] `generation.json`: Greedy generation (with cycle detection status) and 10 seeded samples for both models.
- [x] `comparison.json`: Structural and empirical comparison table (vocabulary size, observed contexts, parameter counts, zero-probability contexts, training log-likelihood, generation diversity, verbatim copy rate).

---

## 4. Test Suite Requirements (`week08_bayesian_networks/tests/`)

- [x] **Exact Oracle Test:** Recomputes every CPT from scratch with `fractions.Fraction` without using `src/` methods, asserts exact match.
  - *Test:* `test_first_order_exact_fraction_oracle`, `test_second_order_exact_fraction_oracle`.
- [x] **Normalization Test:** Asserts $\sum_v P(v \mid c) = 1.0 \pm 10^{-12}$ for all observed contexts; unseen contexts return None.
  - *Test:* `test_normalization_first_and_second_order`, `test_unseen_contexts_handling`.
- [x] **Hand-Checkable Values:** Asserts $\ge 5$ specific probabilities against known fractions ($P(\text{cat} \mid \text{the}) = 1/4$, $P(\text{dog} \mid \text{the}) = 1/4$, $P(\text{sat} \mid \text{cat}) = 2/3$, $P(\text{ran} \mid \text{cat}) = 1/3$, $P(\text{on} \mid \text{sat}) = 1.0$, $P(\text{cat} \mid \langle\text{START}\rangle, \text{the}) = 1/2$, $P(\text{mat} \mid \text{on}, \text{the}) = 1/2$).
  - *Test:* `test_five_hand_checked_probabilities`.
- [x] **Sampling Convergence & Seed Reproducibility:** $N = 20,000$ draws from context `'the'` matches CPT within 0.02; identical seed yields identical stream; different seed differs.
  - *Test:* `test_sampling_empirical_frequency_convergence`, `test_sampling_seed_reproducibility`.
- [x] **Greedy Determinism & Cycle Detection:** Greedy output is deterministic; cycle detector halts infinite loop on first-order model and synthetic cyclic models.
  - *Test:* `test_greedy_determinism_and_cycle_detection`, `test_custom_cycle_detection_model`, `test_second_order_greedy_generation`.
- [x] **N-gram Validity:** Generated tokens obey observed bigram (order 1) and trigram (order 2) transitions.
  - *Test:* `test_ngram_validity_of_generated_sentences`.
- [x] **Mutation Tests:**
  - Mutation 1: unnormalized CPT fails normalization test (`test_mutation_1_unnormalized_cpt`).
  - Mutation 2: second-order model ignoring $w_{t-2}$ fails oracle test (`test_mutation_2_second_order_ignores_context`).
  - Mutation 3: broken greedy tie-break fails deterministic generation test (`test_mutation_3_broken_greedy_tie_break`).

---

## 5. Submission & Gate Compliance

- [x] `README.md` with setup, CLI instructions, and architecture description.
- [x] `REPORT.md` addressing all 14 questions, results summary, and LLM reflection stub.
- [x] `docs/prompt_log.md` updated with Week 8 entry.
- [x] Root `README.md` status table updated for Week 8.
- [x] All repo tests pass (`ruff check .`, `pytest -q -rs`).
- [x] Fresh-clone gate passes in isolated directory `$env:TEMP\fresh_w8`.
