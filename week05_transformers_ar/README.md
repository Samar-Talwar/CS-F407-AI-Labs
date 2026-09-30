# Week 05 — Transformers and Autoregressive Language Models

**Author:** Samar Talwar | CS F407 | Not licensed for reuse or submission by others.

## Overview

This directory contains a complete, verified, from-scratch implementation of an autoregressive decoder-only Transformer Language Model (**TinyGPT**), scaled dot-product self-attention mechanisms, causal attention masking, multi-head projections, Pre-LayerNorm residual blocks, token/position embeddings, training loops, and temperature-controlled text generation.

All implementations strictly adhere to course specifications, using only Python standard library, NumPy, and pure PyTorch (CPU only, deterministic seeds).

---

## How to Run

```bash
# 1. Run full test suite for Week 5 (from repo root, imports via importlib)
pytest week05_transformers_ar/ -q

# 2. Regenerate all result artifacts (results/*.json)
python -m week05_transformers_ar.src.cli

# 3. Quick verification one-liners
python -c "from week05_transformers_ar.src.model import TinyGPT; m = TinyGPT(); print('TinyGPT instantiated. Trainable params:', m.count_parameters())"
python -c "from week05_transformers_ar.src.attention import scaled_dot_product_attention_numpy; import numpy as np; print('NumPy attention oracle OK')"
```

---

## Project Layout

```
week05_transformers_ar/
├── src/                          # Importable modules + CLI entry point
│   ├── __init__.py
│   ├── dataset.py                # CharTokenizer, BigramModel, corpus generator, get_batch
│   ├── attention.py              # Scaled dot-product attention (NumPy & PyTorch), Head, MHA, Custom MHA
│   ├── model.py                  # TinyOneTokenLM, FeedForward (4x ReLU), TransformerBlock, TinyGPT
│   ├── train.py                  # AdamW optimization loop, cross-entropy loss tracking, eval routines
│   ├── generate.py               # Autoregressive decoding loop, temperature scaling softmax(z / tau)
│   ├── metrics.py                # Analytical parameter breakdown, variance scaling, bigram comparisons
│   └── cli.py                    # Complete execution reproducing all 5 JSON result artifacts
├── tests/
│   └── test_transformers.py     # 15 automated test cases: oracles, invariants, mutations, training
├── results/                      # Unrounded machine-generated outputs
│   ├── model_architecture.json   # Parameter tables, analytical breakdown (60,313), custom MHA demo
│   ├── attention_analysis.json   # Dot product variance vs sqrt(d), scaled vs unscaled std, causal demo
│   ├── training_loss.json        # 180-step loss trajectory, evaluation points, hyperparameter metadata
│   ├── generation_samples.json   # Autoregressive generations across temperatures [0.3, 0.7, 0.8, 1.0, 2.0]
│   └── comparison.json           # Quantitative comparison between Bigram baseline and TinyGPT
├── original/                     # Original unmodified notebook
│   └── transformers.ipynb
├── CHECKLIST.md                  # Comprehensive mapping of requirements to code, tests, and results
├── REPORT.md                     # Technical report with detailed derivations, analyses, and student stubs
└── README.md                     # This file
```

---

## Key Experimental Results (from `results/` at full precision)

### 1. Model Architecture & Analytical Parameter Count
- **Vocabulary Size ($V$):** 25 characters
- **Embedding Dimension ($d_{\text{model}}$):** 48
- **Block Context Length ($T$):** 32
- **Attention Heads ($H$):** 4 ($d_{\text{head}} = 12$)
- **Transformer Layers ($L$):** 2
- **Total Trainable Parameters:** **60,313** (Empirical `model.parameters()` exactly matches analytical formula)

| Component | Dimensions / Formula | Parameters |
| :--- | :--- | :--- |
| **Token Embeddings** | $V \times d_{\text{model}} = 25 \times 48$ | 1,200 |
| **Position Embeddings** | $T \times d_{\text{model}} = 32 \times 48$ | 1,536 |
| **Transformer Block 0** | $QKV (6,912) + W_O (2,352) + \text{LN}_1 (96) + \text{FFN}_1 (9,408) + \text{FFN}_2 (9,264) + \text{LN}_2 (96)$ | 28,128 |
| **Transformer Block 1** | Identical structure to Block 0 | 28,128 |
| **Final LayerNorm** | $2 \times d_{\text{model}} = 2 \times 48$ | 96 |
| **Language Model Head** | $d_{\text{model}} \times V + V = 48 \times 25 + 25$ | 1,225 |
| **Total Analytical Count** | $1,200 + 1,536 + 2 \times 28,128 + 96 + 1,225$ | **60,313** |

### 2. Attention Variance Scaling ($\sigma \approx \sqrt{d}$)
Empirical standard deviation of dot products $q^\top k$ for $q, k \sim \mathcal{N}(0, I_d)$ over 3,000 samples:

| Dimension ($d$) | Empirical Std ($\sigma$) | Theoretical $\sqrt{d}$ | Empirical / Theoretical Ratio |
| :--- | :--- | :--- | :--- |
| **2** | 1.408254 | 1.414214 | 0.995786 |
| **8** | 2.845659 | 2.828427 | 1.006093 |
| **32** | 5.529509 | 5.656854 | 0.977488 |
| **128** | 11.418496 | 11.313708 | 1.009262 |
| **512** | 22.655445 | 22.627417 | 1.001239 |

### 3. Training Convergence (180 Steps, $\text{lr}=3\times 10^{-3}$)
- **Initial Loss:** 3.3848 (Step 0)
- **Final Loss:** 0.1598 (Step 180)
- **First-10 Average Loss:** 2.8162
- **Last-10 Average Loss:** 0.1586
- **Total Loss Reduction:** >95.2% cross-entropy drop.

### 4. Bigram Baseline vs. TinyGPT

| Metric | Bigram Model ($N=2$) | TinyGPT (Decoder Transformer) |
| :--- | :--- | :--- |
| **Model Type** | Counting $P(x_t \mid x_{t-1})$ CPT | Multi-layer Self-Attention LM |
| **Context Window ($T$)** | 1 character | 32 characters |
| **Parameters** | 625 ($25^2$) | 60,313 |
| **Corpus Loss** | 1.6013 | 0.1598 |
| **Coherent Sentences** | False (degenerate loops / gibberish) | True (syntactically valid corpus sentences) |

---

## Test Suite Summary

The 15 automated test cases in `week05_transformers_ar/tests/test_transformers.py` enforce:
1. **NumPy vs. PyTorch Oracle:** Strict output and attention weight equivalence between NumPy oracle and PyTorch `Head`.
2. **Analytical Parameter Count Oracle:** Exact equality between mathematical count and PyTorch parameter inspection (60,313).
3. **Causal Invariance:** Future token modifications leave prior token representations strictly unchanged ($< 10^{-6}$ diff).
4. **Attention Score Variance Invariant:** Dot-product variance matches theoretical $\sqrt{d}$ within 15%.
5. **Temperature & Softmax Invariants:** $\tau \to 0$ produces greedy argmax; higher $\tau$ increases entropy.
6. **Autoregressive Generation Invariants:** Proper context truncation and deterministic seed reproducibility.
7. **Convergence Assertion:** 180 steps of AdamW drops loss from $>3.0$ to $<0.50$.
8. **Three Real Monkeypatch Mutation Tests:**
   - *Mutation 1 (Unmasked Causal Attention):* Caught by future-leakage test.
   - *Mutation 2 (Missing $1/\sqrt{d_k}$ Scaling):* Caught by variance explosion test.
   - *Mutation 3 (Broken Residual Connection):* Caught by residual bypass test.
