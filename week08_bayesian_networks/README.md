# CS F407: AI Lab - Week 8
## Bayesian Networks & Autoregressive Markov Language Models

**Author:** Samar Talwar  
**Course:** CS F407 (Artificial Intelligence), BITS Pilani Goa Campus  

---

## 1. Overview & Theoretical Framework

This laboratory investigates the formulation of autoregressive language models as directed acyclic probabilistic graphical models (**Bayesian Networks**).

### Joint Distribution & Chain-Rule Factorisation
For a sequence of discrete random variables $X_1, X_2, \dots, X_T$ representing word tokens:
$$P(X_1, X_2, \dots, X_T) = P(X_1) \prod_{t=2}^T P(X_t \mid X_1, \dots, X_{t-1})$$

### First-Order Markov Model
Under the first-order Markov assumption, each token $X_t$ is conditionally independent of all preceding history given its immediate predecessor $X_{t-1}$:
$$P(X_t \mid X_1, \dots, X_{t-1}) \approx P(X_t \mid X_{t-1})$$
Bayesian Network structure: $X_1 \to X_2 \to X_3 \to \dots \to X_T$.

### Second-Order Markov Model
Under the second-order Markov assumption, each token $X_t$ is conditioned on the previous two tokens $(X_{t-2}, X_{t-1})$:
$$P(X_t \mid X_1, \dots, X_{t-1}) \approx P(X_t \mid X_{t-2}, X_{t-1})$$
Bayesian Network structure: Each node $X_t$ receives directed edges from $X_{t-2}$ and $X_{t-1}$.

---

## 2. Directory Structure

```text
week08_bayesian_networks/
├── src/
│   ├── __init__.py           # Package exports
│   ├── dataset.py            # Canonical corpus definition & padding tokenizers
│   ├── markov_model.py       # First- and second-order models, CPT estimation, greedy & sampling decoders
│   ├── metrics.py            # Model statistics, perplexity, and comparison evaluators
│   └── cli.py                # Pipeline execution & result generator
├── tests/
│   ├── __init__.py
│   └── test_markov_models.py # Exact Fraction oracles, normalization, sampling, and mutation tests
├── results/
│   ├── counts_first_order.json
│   ├── cpt_first_order.json
│   ├── counts_second_order.json
│   ├── cpt_second_order.json
│   ├── normalisation_checks.json
│   ├── chain_rule_probabilities.json
│   ├── generation.json
│   └── comparison.json
├── CHECKLIST.md              # Requirements trace to code & tests
├── README.md                 # Architecture and usage guide
└── REPORT.md                 # Complete report answering Questions 1–14
```

---

## 3. Quickstart & Execution

### Run the Pipeline & Generate Results
```powershell
.venv\Scripts\python -m week08_bayesian_networks.src.cli
```

### Run the Test Suite
```powershell
.venv\Scripts\python -m pytest week08_bayesian_networks\tests\test_markov_models.py -v
```

---

## 4. Key Results Summary

| Metric | First-Order Model | Second-Order Model |
| :--- | :--- | :--- |
| **Vocabulary Size ($|V|$)** | 12 | 12 |
| **Observed Contexts** | 11 | 15 |
| **Non-Zero Parameters (CPT entries)** | 17 | 19 |
| **Average Branching Factor** | 1.545 | 1.267 |
| **Training Set Total Log-Likelihood** | -22.8874 | -10.7506 |
| **Greedy Decoding Behavior** | Infinite periodic cycle (`the -> cat -> sat -> on -> the`) | Terminates with `the cat sat on the mat` |
| **Verbatim Training Reproduction (100 samples)** | 16.0% | 100.0% |
| **Distinct Sentences in 100 Samples** | 37 | 6 |
