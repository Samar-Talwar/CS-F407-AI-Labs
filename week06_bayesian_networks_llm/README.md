# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

# Week 06 — Bayesian Networks and LLM Integration

**Author:** Samar Talwar  
**Course:** CS F407 (Artificial Intelligence), BITS Pilani Goa Campus  

---

## 1. Overview & Theoretical Framework

This module provides a complete, self-contained, reproducible implementation of discrete Bayesian Networks, exact probabilistic inference via Variable Elimination, parameter learning (MLE and Bayesian BDeu estimation), and static AST-checked LLM code generation and automated mathematical validation.

All core algorithms are implemented using pure Python and NumPy with zero non-standard runtime dependencies (no `pgmpy` or `pandas` needed at runtime).

### Canonical Sprinkler Bayesian Network
The Sprinkler network models four binary random variables ($X \in \{0, 1\}$):
- $C$: Cloudy
- $R$: Rain (child of $C$)
- $S$: Sprinkler (child of $C$)
- $W$: WetGrass (child of $R, S$)

The joint probability factorizes according to the chain rule for Bayesian Networks:
$$P(C, R, S, W) = P(C) \cdot P(R \mid C) \cdot P(S \mid C) \cdot P(W \mid R, S)$$

### Exact Inference: Variable Elimination
Given query variable $Q$ and evidence $\mathbf{E} = \mathbf{e}$, hidden variables $\mathbf{H} = \mathbf{X} \setminus (\{Q\} \cup \mathbf{E})$ are eliminated iteratively:
$$P(Q \mid \mathbf{E} = \mathbf{e}) \propto \sum_{\mathbf{H}} \prod_{\phi \in \Phi} \phi_{\mathbf{E} = \mathbf{e}}$$
where tensor factors $\phi$ undergo evidence reduction, tensor-product expansion along missing dimensions, sum-out marginalization, and final normalization.

### Independent Oracle: 16-State Joint Enumeration
An independent brute-force oracle computes the complete $2^4 = 16$ joint state assignments:
$$P(\text{Rain}=1 \mid \text{WetGrass}=1) = \frac{\sum_{c, s} P(C=c, \text{Rain}=1, S=s, \text{WetGrass}=1)}{\sum_{c, r, s} P(C=c, R=r, S=s, \text{WetGrass}=1)}$$

### Parameter Estimation
1. **Maximum Likelihood Estimation (MLE):**
   $$\hat{\theta}_{x \mid \mathbf{pa}} = \frac{N(x, \mathbf{pa})}{N(\mathbf{pa})}$$
2. **Bayesian Estimation with BDeu Prior:**
   Conjugate Dirichlet prior smoothing with equivalent sample size $\alpha = 10$:
   $$\hat{\theta}_{ijk} = \frac{N_{ijk} + \frac{\alpha}{r_i \cdot q_i}}{N_{ij} + \frac{\alpha}{q_i}}$$
   where $r_i$ is the cardinality of variable $X_i$ ($r_i = 2$) and $q_i$ is the number of parent configurations.

---

## 2. Directory Structure

```text
week06_bayesian_networks_llm/
├── src/                          # Importable modules + CLI entry point
│   ├── __init__.py
│   ├── bn.py                     # TabularCPD, DiscreteBayesianNetwork, SimpleDataFrame, CPT validator
│   ├── inference.py              # DiscreteFactor tensor algebra, VariableElimination, enumerate_joint oracle
│   ├── estimation.py             # Synthetic data generator, fit_mle, fit_bayesian (BDeu), sample size sweep
│   ├── llm_interface.py          # AST safety analyzer, StubLLMBackend, execution sandbox, validator
│   └── cli.py                    # Complete execution pipeline reproducing all JSON and PNG artifacts
├── tests/
│   ├── __init__.py
│   ├── test_bn.py                # Reference model structure, CPT validation, sampling invariants
│   ├── test_inference.py         # VE vs 16-state joint enumeration oracle, factor tensor algebra
│   ├── test_estimation.py        # MLE vs Bayesian BDeu, sample size convergence, manual verification
│   └── test_llm_interface.py     # AST safety checker (forbidden calls/imports), sandbox validation
├── results/                      # Unrounded machine-generated outputs
│   ├── reference_model.json      # Sprinkler DAG nodes, edges, and cardinalities
│   ├── reference_cpds.json       # Exact CPT distributions for Cloudy, Rain, Sprinkler, WetGrass
│   ├── inference_posteriors.json # VE vs Enumeration oracle comparisons and exercises
│   ├── estimation_results.json   # MLE fits, manual verification, BDeu comparison, sample-size sweep
│   ├── llm_generated_inference_stub.json # Stub LLM inference code execution and validation report
│   ├── llm_generated_mle_stub.json       # Stub LLM MLE code execution report
│   ├── llm_generated_bayesian_stub.json  # Stub LLM BDeu Bayesian code execution report
│   └── mle_sweep_plot.png        # Sample size vs parameter error visualization
├── original/                     # Unmodified source notebook
│   └── llm_bn.ipynb
├── CHECKLIST.md                  # Comprehensive mapping of requirements to code, tests, and results
├── REPORT.md                     # Technical report with verified results and student reflection stubs
└── README.md                     # This file
```

---

## 3. How to Run

```bash
# 1. Run full test suite for Week 6 (99 tests)
pytest week06_bayesian_networks_llm/tests/ -q

# 2. Regenerate all result artifacts (results/*.json and results/*.png)
python -m week06_bayesian_networks_llm.src.cli --all

# 3. Run specific sub-pipelines via CLI
python -m week06_bayesian_networks_llm.src.cli --reference
python -m week06_bayesian_networks_llm.src.cli --inference
python -m week06_bayesian_networks_llm.src.cli --estimation
python -m week06_bayesian_networks_llm.src.cli --llm-stub

# 4. Quick verification one-liners
python -c "from week06_bayesian_networks_llm.src.bn import build_reference_model; m = build_reference_model(); print('Model valid:', m.check_model())"
python -c "from week06_bayesian_networks_llm.src.inference import query_posterior, build_reference_model; m = build_reference_model(); print('P(Rain=1|WetGrass=1):', query_posterior(m, ['Rain'], {'WetGrass': 1})[1])"
```

---

## 4. Key Experimental Results (from `results/` at full precision)

- **Target Exact Posterior:**  
  $$P(\text{Rain}=1 \mid \text{WetGrass}=1) = 0.7047692307692307$$
- **Enumeration Oracle Discrepancy:**  
  $$\text{diff} = 2.220446049250313 \times 10^{-16}$$ (machine precision bound)
- **Additional Exercise Posteriors:**
  - $P(\text{Sprinkler}=1 \mid \text{WetGrass}=1) = 0.42784615384615376$
  - $P(\text{Cloudy}=1 \mid \text{WetGrass}=1) = 0.5746153846153845$
  - $P(\text{Rain}=1 \mid \text{WetGrass}=1, \text{Sprinkler}=0) = 0.9922022048937886$
- **MLE Parameter Estimation ($N=1000$, seed=7):**
  - $\hat{P}(\text{Rain}=1 \mid \text{Cloudy}=1) = 0.8293172690763052$ (True = 0.8)
  - Manual counting verification matches native MLE exactly: `0.8293172690763052`
- **Small-Sample Regularization ($N=30$, seed=11):**
  - MLE estimate: $0.625$
  - BDeu ($\alpha=10$) estimate: $0.5952380952380952$
- **Sample-Size Convergence Sweep:**
  - $N=20$: error = $0.13333333333333341$
  - $N=50$: error = $0.09629629629629632$
  - $N=100$: error = $0.040000000000000036$
  - $N=500$: error = $0.03396226415094339$
  - $N=1000$: error = $0.029317269076305164$
  - $N=5000$: error = $0.0054850593532542735$
