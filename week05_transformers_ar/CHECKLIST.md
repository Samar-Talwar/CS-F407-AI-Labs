# Week 5 Transformers & Autoregressive Models: Checklist

This checklist tracks every concept, mathematical derivation, numerical experiment, component, test, measurement, comparison, and deliverable from `week05_transformers_ar/original/transformers.ipynb` and `docs/PROJECT_RULES.md`.

## 1. Core Architecture & Mathematical Constraints

- [x] Standard library, NumPy, and PyTorch (CPU only, deterministic seeds) in `week05_transformers_ar/src/`.
- [x] Author header on all source files (`# CS F407 Lab, Week 5 | Author: Samar Talwar | Not licensed for reuse or submission by others.`).
- [x] Canonical repeated 6-sentence corpus (17,680 characters, 25 unique character vocabulary).
- [x] Character-level tokenizer (`CharTokenizer`) with integer IDs, `stoi`/`itos`, encode and decode routines.
- [x] Shifted input-target context pairs $(x_b, y_b)$ of shape $(B=32, T=32)$ for next-token prediction.
- [x] Bigram counting baseline ($P(x_t \mid x_{t-1})$ lookup table, smoothing, and AR generation).
- [x] Scaled dot-product attention: $\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$.
- [x] Attention score variance scaling: dividing by $\sqrt{d_k}$ stabilizes dot-product variance across dimensions.
- [x] Strict lower-triangular causal masking ($M_{ij} = 0$ for $j \le i$, $-\infty$ for $j > i$) preventing future token leakage.
- [x] Multi-Head Attention (MHA): parallel learned projections from full $d_{\text{model}}$ space to $d_{\text{head}}$, concatenation, and linear mixing $W_O$.
- [x] Distinction between attention scaling factor $1/\sqrt{d_k}$ vs head division $d_{\text{head}} = d_{\text{model}} / H$.
- [x] Non-standard MHA demonstration (e.g. $d_{\text{model}}=8, H=6, d_{\text{head}}=2, W_O \in \mathbb{R}^{12 \times 8}$).
- [x] Learned positional embeddings ($x_t = \text{tok}(x_t) + \text{pos}(t)$).
- [x] Pre-LayerNorm residual connections ($x \leftarrow x + \operatorname{MHA}(\operatorname{LN}(x))$, $x \leftarrow x + \operatorname{FFN}(\operatorname{LN}(x))$).
- [x] Position-wise Feed-Forward Network: $\operatorname{FFN}(x) = W_2 \operatorname{ReLU}(W_1 x + b_1) + b_2$ with hidden dimension $4 \times d_{\text{model}}$.
- [x] Full TinyGPT decoder-only architecture ($d_{\text{model}}=48, n_{\text{heads}}=4, n_{\text{layers}}=2, \text{block\_size}=32, V=25$).
- [x] Analytical and empirical parameter count verification (exact 60,313 trainable parameters).
- [x] Training pipeline: AdamW optimizer ($\text{lr}=3\times 10^{-3}$), cross-entropy loss tracking across 180 steps.
- [x] Autoregressive generation: next-token sampling from position $-1$, temperature scaling $\operatorname{softmax}(z / \tau)$, prompt conditioning.
- [x] CLI entry point regenerating all result files and printing comprehensive summaries (`python -m week05_transformers_ar.src.cli`).

---

## 2. Notebook Concept & Analysis Sections

- [x] **Part I: Fundamentals of Neural Networks** (Linear layers $y = xW + b$, logits vs probabilities, stable softmax, cross-entropy loss $L = -\log p$, gradient descent updates).
  - *Satisfied by:* `src/attention.py:softmax_numpy()`, `cross_entropy_numpy()`, `results/attention_analysis.json`.
- [x] **Part II: Language Modelling** (Next-token prediction $P(x_t \mid x_{<t})$, character tokenization, bigram model $P(x_t \mid x_{t-1})$, bigram generation with smoothing, context window failure on long-distance dependencies e.g., "France... fluent French").
  - *Satisfied by:* `src/dataset.py:CharTokenizer`, `BigramModel`, `results/comparison.json`.
- [x] **Part III: Embeddings** (Learned lookup table $V \times d_{\text{model}}$, token ID index vs learned vector, single-token neural LM `TinyOneTokenLM`).
  - *Satisfied by:* `src/model.py:TinyOneTokenLM`, `TinyGPT.token_embedding`.
- [x] **Part IV: Self-Attention Mechanics** (Query, Key, Value projections from $X$; dot-product relevance $q_i^\top k_j$; scaled dot-product attention; empirical verification of dot-product variance growth $\sigma \approx \sqrt{d}$).
  - *Satisfied by:* `src/attention.py:scaled_dot_product_attention_numpy()`, `results/attention_analysis.json`.
- [x] **Part V: Causal Masking** (Lower-triangular mask, $-\infty$ substitution, row softmax with $-\infty$, matrix form of causal self-attention).
  - *Satisfied by:* `src/attention.py:causal_self_attention_numpy()`, `Head.forward()`.
- [x] **Part VI: Multi-Head Attention (MHA)** (Projections from full $d_{\text{model}}$ space vs naive coordinate slicing, shape transformations $(B, T, d_{\text{model}}) \to (B, H, T, d_{\text{head}})$, head concatenation $[H_1 \Vert \dots \Vert H_H]$, output projection $W_O$, standard $d_{\text{head}}=d_{\text{model}}/H$ rule vs mathematical feasibility of $H=6$ with $d_{\text{model}}=8$).
  - *Satisfied by:* `src/attention.py:MultiHeadAttention`, `CustomMultiHeadAttention`, `results/model_architecture.json`.
- [x] **Part VII: Transformer Components** (Learned positional embeddings, pre-LN residual connections, LayerNorm mean/variance normalization, position-wise FFN with $4\times$ expansion, composite decoder block).
  - *Satisfied by:* `src/model.py:FeedForward`, `TransformerBlock`.
- [x] **Part VIII: Building & Training TinyGPT** (Corpus repetition 80x, shifted batch generation $(x_b, y_b)$, complete TinyGPT module, parameter counting breakdown, 180-step AdamW training loop, loss convergence from $\approx 3.37$ to $\approx 0.15$).
  - *Satisfied by:* `src/model.py:TinyGPT`, `src/train.py:train_tiny_gpt()`, `results/training_loss.json`.
- [x] **Part IX: Autoregressive Generation & Temperature Scaling** (Next-token generation loop, context truncation to `block_size`, temperature scaling $\tau \in \{0.3, 0.7, 1.0, 2.0\}$, entropy/sharpness analysis, prompt continuation).
  - *Satisfied by:* `src/generate.py:generate_tokens()`, `results/generation_samples.json`.
- [x] **Common Misconceptions Addressed** (Fixed slicing fallacy, attention weights vs values, Q/K/V sources, causal masking during training, non-parallel AR generation).
  - *Satisfied by:* `week05_transformers_ar/REPORT.md`.

---

## 3. Results Files & Keys (`week05_transformers_ar/results/`)

- [x] `model_architecture.json`: Layer-by-layer parameter counts, tensor shapes, submodule breakdown, and comparison between standard 4-head and custom 6-head architectures.
- [x] `training_loss.json`: Complete 180-step training loss curve, initial loss, final loss, first-10 average loss, and last-10 average loss.
- [x] `attention_analysis.json`: NumPy numerical attention weights, empirical dot-product std vs $\sqrt{d}$ across dimensions [2, 8, 32, 128, 512], and causal mask verification.
- [x] `generation_samples.json`: Autoregressive generation outputs across temperatures ($\tau \in [0.3, 0.7, 0.8, 1.0, 2.0]$) for prompt `"baana "` and empty prompt, plus bigram baseline samples.
- [x] `comparison.json`: Comprehensive comparative analysis between Bigram Baseline and TinyGPT (parameter count, context length, training loss, distinct generated sequences, and verbatim training corpus reproduction rate).

---

## 4. Test Suite Requirements (`week05_transformers_ar/tests/`)

- [x] **Independent Oracle Tests**:
  - Compare NumPy `causal_self_attention_numpy` vs PyTorch `Head` on identical fixed weights and inputs.
  - Compare analytical parameter count formula against PyTorch `model.parameters()` count (60,313 parameters).
- [x] **Shape & Interface Contract Tests**:
  - Input batch $(B, T)$ produces logits $(B, T, V)$ and scalar loss.
  - Exceeding `block_size` triggers `ValueError`.
  - MultiHeadAttention preserves input dimension $(B, T, d_{\text{model}})$.
- [x] **Causal Masking Invariant Tests**:
  - Modifying future tokens in the input does NOT affect past/current token representations or logits.
  - Attention weights at upper-triangular positions are strictly $0.0$.
- [x] **LayerNorm & Normalization Tests**:
  - Output features have zero mean and unit variance.
- [x] **Temperature & Generation Invariant Tests**:
  - Low temperature ($\tau \to 0$) yields deterministic greedy generation.
  - Higher temperature increases token entropy.
  - Generation seed reproducibility (identical seeds yield identical token streams; different seeds differ).
- [x] **Training Convergence Tests**:
  - 180-step training run achieves significant cross-entropy loss reduction ($\text{loss}_{\text{final}} < 0.5$, $\text{loss}_{\text{initial}} > 3.0$).
- [x] **Real Mutation Tests**:
  - Mutation 1: Corrupted causal mask (unmasked / bidirectional attention) caught by causality leakage test.
  - Mutation 2: Missing $\sqrt{d_k}$ attention scaling caught by variance/scale invariant test.
  - Mutation 3: Broken residual connection ($x = \text{attn}(x)$ instead of $x + \text{attn}(x)$) caught by gradient flow / identity pass-through test.

---

## 5. Submission & Gate Compliance

- [x] `README.md` with complete documentation, CLI usage, and module architecture.
- [x] `REPORT.md` answering all questions, presenting experimental metrics, and including LLM reflection stub.
- [x] `docs/prompt_log.md` updated for Week 5.
- [x] Root `README.md` status table updated for Week 5.
- [x] All repo tests pass (`ruff check .`, `pytest -q -rs`).
- [x] Fresh-clone gate passes in isolated directory `$env:TEMP\fresh_w5`.
