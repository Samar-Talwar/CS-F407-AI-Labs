# CS F407 Lab, Week 6 | CHECKLIST.md

**Source notebook:** `week06_bayesian_networks_llm/original/llm_bn.ipynb`

---

## Part 1: Trusted Reference Model Construction

| # | Notebook Task / Output | Requirement | Satisfied By (File / Key / Test) |
|---|------------------------|-------------|-----------------------------------|
| 1.1 | Build reference Sprinkler BN with edges: Cloudy→Rain, Cloudy→Sprinkler, Rain→WetGrass, Sprinkler→WetGrass | `src/bn.py:build_reference_model()` | `tests/test_bn.py::test_reference_model_structure` |
| 1.2 | Define CPD for Cloudy: P(Cloudy=1)=0.5 | `src/bn.py:build_reference_model()` | `tests/test_bn.py::test_cpd_cloudy` |
| 1.3 | Define CPD for Rain: P(Rain=1\|Cloudy=0)=0.2, P(Rain=1\|Cloudy=1)=0.8 | `src/bn.py:build_reference_model()` | `tests/test_bn.py::test_cpd_rain` |
| 1.4 | Define CPD for Sprinkler: P(Sprinkler=1\|Cloudy=0)=0.5, P(Sprinkler=1\|Cloudy=1)=0.1 | `src/bn.py:build_reference_model()` | `tests/test_bn.py::test_cpd_sprinkler` |
| 1.5 | Define CPD for WetGrass: 4 parent configurations with given probs | `src/bn.py:build_reference_model()` | `tests/test_bn.py::test_cpd_wetgrass` |
| 1.6 | `model.check_model()` returns True | `src/bn.py:build_reference_model()` | `tests/test_bn.py::test_check_model_valid` |
| 1.7 | Print model edges and CPDs | `results/reference_cpds.json` | `tests/test_bn.py::test_cpd_values_match_notebook` |

---

## Part 2: Exact Inference with Variable Elimination

| # | Notebook Task / Output | Requirement | Satisfied By (File / Key / Test) |
|---|------------------------|-------------|-----------------------------------|
| 2.1 | Query P(Rain=1 \| WetGrass=1) using VariableElimination | `src/inference.py:query_posterior()` | `tests/test_inference.py::test_posterior_rain_given_wetgrass` |
| 2.2 | Reference result: P(Rain=1 \| WetGrass=1) = 0.7047692307692307 | `results/inference_posteriors.json` key `rain_given_wetgrass` | `tests/test_inference.py::test_posterior_rain_given_wetgrass_exact` |
| 2.3 | Independent enumeration verification (16 assignments) | `src/inference.py:enumerate_joint()` | `tests/test_inference.py::test_enumeration_matches_variable_elimination` |
| 2.4 | Difference between pgmpy and enumeration ≈ 2.22e-16 | `results/inference_posteriors.json` key `enumeration_diff` | `tests/test_inference.py::test_enumeration_diff_tolerance` |

---

## Part 3: LLM Code Generation for Inference

| # | Notebook Task / Output | Requirement | Satisfied By (File / Key / Test) |
|---|------------------------|-------------|-----------------------------------|
| 3.1 | LLM prompt for building BN and querying P(Rain\|WetGrass=1) | `src/llm_interface.py:LLMInterface.generate_inference_code()` | `tests/test_llm_interface.py::test_inference_prompt_contains_requirements` |
| 3.2 | Extract Python code from LLM response (markdown fences) | `src/llm_interface.py:extract_python_code()` | `tests/test_llm_interface.py::test_extract_python_code` |
| 3.3 | Basic AST safety check (allowed imports, forbidden calls) | `src/llm_interface.py:basic_generated_code_check()` | `tests/test_llm_interface.py::test_basic_code_check` |
| 3.4 | Validate LLM-generated model structure, check_model, posterior | `src/llm_interface.py:validate_generated_inference()` | `tests/test_llm_interface.py::test_validate_generated_inference` |
| 3.5 | LLM-generated code execution result (stubbed) | `results/llm_generated_inference.json` with `backend: "stub"` | `tests/test_llm_interface.py::test_stub_backend_deterministic` |

---

## Part 4: Parameter Estimation (MLE)

| # | Notebook Task / Output | Requirement | Satisfied By (File / Key / Test) |
|---|------------------------|-------------|-----------------------------------|
| 4.1 | Generate synthetic data (N=1000, seed=7) from reference model | `src/estimation.py:generate_synthetic_data()` | `tests/test_estimation.py::test_generate_synthetic_data_shape` |
| 4.2 | Fit MLE model on data_1000 using DiscreteMLE | `src/estimation.py:fit_mle()` | `tests/test_estimation.py::test_fit_mle_valid` |
| 4.3 | Print fitted CPDs for all 4 variables | `results/mle_fitted_cpds_N1000.json` | `tests/test_estimation.py::test_mle_cpds_match_notebook` |
| 4.4 | Manual MLE verification: P(Rain=1\|Cloudy=1) = 0.7725409836065574 | `results/mle_manual_verification.json` | `tests/test_estimation.py::test_manual_mle_matches_pgmpy` |
| 4.5 | True parameter P(Rain=1\|Cloudy=1) = 0.8 | — | — |
| 4.6 | Sample size sweep: N ∈ [20, 50, 100, 500, 1000, 5000] | `src/estimation.py:sample_size_sweep()` | `tests/test_estimation.py::test_sample_size_sweep` |
| 4.7 | Estimates and absolute errors for each N | `results/mle_sample_size_sweep.json` | `tests/test_estimation.py::test_sample_size_sweep_values` |
| 4.8 | Plot: parameter estimate vs sample size (log scale) | `results/mle_sweep_plot.png` | `tests/test_estimation.py::test_plot_generated` |

---

## Part 5: LLM Code Generation for MLE Estimation

| # | Notebook Task / Output | Requirement | Satisfied By (File / Key / Test) |
|---|------------------------|-------------|-----------------------------------|
| 5.1 | LLM prompt for MLE estimation with given DataFrame | `src/llm_interface.py:LLMInterface.generate_estimation_code()` | `tests/test_llm_interface.py::test_estimation_prompt_contains_requirements` |
| 5.2 | Extract code, basic check | `src/llm_interface.py:extract_python_code()`, `basic_generated_code_check()` | `tests/test_llm_interface.py::test_extract_python_code_estimation` |
| 5.3 | Compare LLM-generated MLE model with trusted fit (max abs diff) | `src/llm_interface.py:compare_models()` | `tests/test_llm_interface.py::test_compare_mle_models` |
| 5.4 | LLM-generated estimation result (stubbed) | `results/llm_generated_estimation.json` with `backend: "stub"` | `tests/test_llm_interface.py::test_stub_estimation_deterministic` |

---

## Part 6: Bayesian Parameter Estimation (BDeu)

| # | Notebook Task / Output | Requirement | Satisfied By (File / Key / Test) |
|---|------------------------|-------------|-----------------------------------|
| 6.1 | Generate small dataset (N=30, seed=11) | `src/estimation.py:generate_synthetic_data()` | `tests/test_estimation.py::test_small_data_generation` |
| 6.2 | Fit MLE on small data | `src/estimation.py:fit_mle()` | `tests/test_estimation.py::test_mle_small_data` |
| 6.3 | Fit Bayesian estimator (BDeu, ess=10) on small data | `src/estimation.py:fit_bayesian()` | `tests/test_estimation.py::test_fit_bayesian_valid` |
| 6.4 | Compare P(Rain=1\|Cloudy=1): True=0.8, MLE=0.909091, BDeu=0.78125 | `results/bayesian_small_sample_comparison.json` | `tests/test_estimation.py::test_bayesian_vs_mle_small_sample` |

---

## Part 7: LLM Code Generation for Bayesian Estimation

| # | Notebook Task / Output | Requirement | Satisfied By (File / Key / Test) |
|---|------------------------|-------------|-----------------------------------|
| 7.1 | LLM prompt for BayesianEstimator with BDeu, ess=10 | `src/llm_interface.py:LLMInterface.generate_bayesian_code()` | `tests/test_llm_interface.py::test_bayesian_prompt_contains_requirements` |
| 7.2 | Extract code, basic check | `src/llm_interface.py:extract_python_code()`, `basic_generated_code_check()` | `tests/test_llm_interface.py::test_extract_python_code_bayesian` |
| 7.3 | LLM-generated Bayesian estimation result (stubbed) | `results/llm_generated_bayesian.json` with `backend: "stub"` | `tests/test_llm_interface.py::test_stub_bayesian_deterministic` |

---

## Part 8: Validation and Error Handling

| # | Notebook Task / Output | Requirement | Satisfied By (File / Key / Test) |
|---|------------------------|-------------|-----------------------------------|
| 8.1 | Broken CPT: columns don't sum to 1 → ValueError caught | `src/bn.py:validate_cpt()` + `tests/test_bn.py::test_invalid_cpt_rejected` | `tests/test_bn.py::test_invalid_cpt_rejected` |
| 8.2 | Subtle semantic error: permuted WetGrass CPD columns → check_model passes but posterior wrong | `results/semantic_error_comparison.json` | `tests/test_bn.py::test_semantic_error_detected_by_posterior` |
| 8.3 | Trusted test suite for reference model | `tests/test_bn.py::test_reference_model` | `tests/test_bn.py::test_reference_model` |

---

## Part 9: HW Exercises (Student Tasks)

| # | Notebook Task / Output | Requirement | Satisfied By (File / Key / Test) |
|---|------------------------|-------------|-----------------------------------|
| 9.1 | Query P(Sprinkler=1 \| WetGrass=1) = 0.4278461538461539 | `results/exercise_posteriors.json` key `sprinkler_given_wetgrass` | `tests/test_inference.py::test_exercise_sprinkler_given_wetgrass` |
| 9.2 | Query P(Cloudy=1 \| WetGrass=1) = 0.5746153846153845 | `results/exercise_posteriors.json` key `cloudy_given_wetgrass` | `tests/test_inference.py::test_exercise_cloudy_given_wetgrass` |
| 9.3 | Query P(Rain=1 \| WetGrass=1, Sprinkler=0) = 0.9922022048937886 | `results/exercise_posteriors.json` key `rain_given_wetgrass_sprinkler0` | `tests/test_inference.py::test_exercise_rain_given_wetgrass_sprinkler0` |
| 9.4 | LLM explanation of WetGrass CPT columns | — | `REPORT.md` (DRAFT stub) |
| 9.5 | Multiple datasets (seeds 1-5, N=100), record estimates | `results/mle_multi_seed_estimates.json` | `tests/test_estimation.py::test_multi_seed_estimates` |
| 9.6 | LLM explanation of sampling variability | — | `REPORT.md` (DRAFT stub) |

---

## Part 10: Deliverables

| Deliverable | File |
|-------------|------|
| Source modules | `src/bn.py`, `src/inference.py`, `src/estimation.py`, `src/llm_interface.py`, `src/cli.py` |
| Tests | `tests/test_bn.py`, `tests/test_inference.py`, `tests/test_estimation.py`, `tests/test_llm_interface.py` |
| Results | `results/*.json`, `results/*.png` |
| Documentation | `README.md`, `REPORT.md`, `CHECKLIST.md` |
| Prompt log | `docs/prompt_log.md` |

---

## External Dependencies

| Dependency | Used For | Status |
|------------|----------|--------|
| `pgmpy` | Bayesian network, inference, estimation | Required (in venv) |
| `pandas` | DataFrames for estimation | Required (in venv) |
| `numpy` | Numerical computation | Required (in venv) |
| `matplotlib` | Plotting sweep results | Required (in venv) |
| `transformers` + `accelerate` | Real LLM backend (Qwen2.5-Coder) | **Optional** — import-guarded, only used when `USE_REAL_LLM=1` and model available |
| `torch` | Required by transformers | Optional (with transformers) |

**LLM Interface Design:**
- Protocol `LLMBackend` with methods: `generate(prompt) -> str`
- `StubLLMBackend` (deterministic, used by default and all tests)
- `RealLLMBackend` (optional, loads Qwen2.5-Coder via transformers pipeline)
- CLI flag `--use-real-llm` to enable real backend (skipped if unavailable)
- All results files include `"backend": "stub"` or `"backend": "real"`

---

## Quality Gates (from PROJECT_RULES.md)

- [ ] `ruff check .` passes
- [ ] `pytest -q -rs` passes (whole repo)
- [ ] Fresh-clone gate: `git clone . $env:TEMP\fresh_w6`, then `ruff check .`, `pytest -q -rs`, CLI run with existing venv python
- [ ] Skeptical-grader audit against notebook
- [ ] Conventional Commits, git push
- [ ] `git status` clean, HEAD = origin/main