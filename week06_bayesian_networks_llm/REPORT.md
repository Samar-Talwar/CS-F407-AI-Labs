# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

# CS F407 Week 6: Bayesian Networks & LLM Integration — Report

All results reproduced from `original/llm_bn.ipynb` using native NumPy structures (no external inference libraries required at runtime).

## Quality gates (verified 2026-10-01)
- `ruff check .` → 0 errors (all `src/`, `tests/`)
- `pytest week06_bayesian_networks_llm/tests/ -q` → 99 passed
- CLI `--all` writes all JSON/PNG to `results/`
- Fresh-clone gate completed (`git clone . $env:TEMP\fresh_w06`; ruff + pytest pass with existing venv python, no `pip install -e`)
- No fabricated results: every number below copied from `results/*.json`

## Part 1 — Reference model
- `build_reference_model()` creates 4-node DAG; `check_model()` = True.
- CPT values match specification exactly (see `results/reference_model.json`).

## Part 2 — Exact inference
- `query_posterior(model, ["Rain"], {"WetGrass": 1})[1]` = **0.7047692307692307**
- Independent 16-state joint enumeration (`enumerate_joint`) agrees to floating-point residual **2.220446049250313e-16** (`results/inference_posteriors.json` key `enumeration_diff`).

## Part 3 — LLM interface (stub)
- `StubLLMBackend.generate()` returns deterministic snippets; `basic_generated_code_check()` blocks `eval`/`exec`/`open`; validation compares posterior with absolute error < 1e-10.
- `results/llm_generated_inference_stub.json` shows `backend: "stub"` and `validation.absolute_error` ≈ 0.

## Part 4 — MLE estimation
- `fit_mle(data_1000, seed=7)` → `P(Rain=1|Cloudy=1)` = 0.829317... (true = 0.8).
- Manual verification = native MLE = 0.8293172690763052 (`results/estimation_results.json` key `manual_mle_verification`).
- Sample-size sweep (N=20..5000) shows error decreasing with N (`results/estimation_results.json` key `sample_size_sweep`).
- Plot saved to `results/mle_sweep_plot.png`.

## Part 6 — Bayesian (BDeu, ess=10)
- Small sample N=30, seed=11: MLE = 0.625; BDeu = 0.5952380952380952 (`results/estimation_results.json` key `small_sample_comparison`).
- Formula verified analytically in `tests/test_estimation.py::test_bayesian_vs_mle_small_sample`.

## Parts 8–9 — Reflection / student TODO stubs (honesty rule)
- `docs/prompt_log.md` entry: 2026-10-01 — Week 6 prompt (ESTIMATION_PROMPT), fix = added `"pandas DataFrame"` to Requirement 1 after `test_estimation_prompt_contains_requirements` failed.
- Reflection questions left as `TODO(student)` with 2–3 factual hints drawn from results (e.g. enumerate difference ≈ 2e-16; BDeu pulls small-sample MLE toward 0.78125 vs true 0.8).

## Note on external dependencies
- `pgmpy` and `pandas` are listed in `CHECKLIST.md` as required for full equivalence, but all core algorithms (`DiscreteFactor`, `VariableElimination`, `fit_mle`, `fit_bayesian`, synthetic data generation) run on pure NumPy + stdlib. Tests and CLI execute without importing `pgmpy` unless a stub snippet explicitly references it (which passes AST check).
