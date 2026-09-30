# CS F407: Artificial Intelligence Lab — Week 5 Report
## Transformers, Self-Attention Mechanics, and Autoregressive Language Modeling

**Author:** Samar Talwar  
**Course:** CS F407 (Artificial Intelligence), BITS Pilani Goa Campus  
**Date:** Semester 1, AY 2026-27  

---

## 1. Executive Summary & Mathematical Framework

This report documents the architectural design, mathematical derivations, empirical validation, and training results for **TinyGPT**, an autoregressive decoder-only Transformer Language Model built from scratch in PyTorch and NumPy without high-level library abstractions.

### 1.1 Autoregressive Factorisation
Language modeling factorises the joint probability of a sequence of discrete tokens $(x_1, x_2, \dots, x_T)$ from a finite vocabulary $\mathcal{V}$ via the exact chain rule of probability:
$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

Unlike fixed-order $N$-gram models that truncate context $P(x_t \mid x_{<t}) \approx P(x_t \mid x_{t-N+1:t-1})$, decoder-only Transformers maintain an adaptive full-context representation over a context window of length $T$ using scaled dot-product attention and Pre-LayerNorm residual blocks.

---

## 2. Theoretical Foundations & Architectural Derivations

### 2.1 Scaled Dot-Product Attention
Given an input sequence $X \in \mathbb{R}^{T \times d_{\text{model}}}$, self-attention computes query, key, and value representations via learned linear projections:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
where $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

The attention weights and context-aggregated values are computed as:
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

#### Mathematical Derivation of Variance Scaling ($\sqrt{d_k}$)
Let $q, k \in \mathbb{R}^{d_k}$ be independent query and key vectors whose components are independent random variables with zero mean and unit variance: $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$, $\operatorname{Var}(q_i) = \operatorname{Var}(k_i) = 1$.

The dot product is $S = q^\top k = \sum_{i=1}^{d_k} q_i k_i$.
1. **Expected Value:**
   $$\mathbb{E}[S] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$
2. **Variance:**
   $$\operatorname{Var}(S) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k} \left( \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 \right)$$
   Since $q_i, k_i$ are independent:
   $$\mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = (1)(1) = 1 \implies \operatorname{Var}(S) = \sum_{i=1}^{d_k} 1 = d_k$$
   Therefore, standard deviation scales as $\sigma_S = \sqrt{d_k}$.

3. **Scaled Dot Product:**
   $$\operatorname{Var}\left(\frac{S}{\sqrt{d_k}}\right) = \frac{1}{d_k} \operatorname{Var}(S) = \frac{d_k}{d_k} = 1.0$$

**Why this matters:** Without the $1/\sqrt{d_k}$ scaling factor, for large head dimensions (e.g. $d_k=64$ or $512$), the magnitude of $Q K^\top$ grows large. Large logits feed into the softmax function push it into regions of near-zero gradients (saturation), severely impeding backpropagation.

---

### 2.2 Causal Attention Masking
To preserve the autoregressive property during parallel training, token $t$ must only attend to tokens $\le t$. Future information is suppressed via an additive causal mask $M \in \mathbb{R}^{T \times T}$:
$$M_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

When computing softmax:
$$\operatorname{softmax}(A + M)_{ij} = \frac{\exp(A_{ij} + M_{ij})}{\sum_{k=1}^T \exp(A_{ik} + M_{ik})}$$
For $j > i$, $A_{ij} + M_{ij} = -\infty$, yielding $\exp(-\infty) = 0$. Consequently, future tokens receive strictly zero attention weight.

---

### 2.3 Multi-Head Attention (MHA)
Rather than performing a single attention function in $d_{\text{model}}$ space, Multi-Head Attention projects queries, keys, and values into $H$ distinct subspaces of dimension $d_{\text{head}}$:
$$\operatorname{head}_h = \operatorname{Attention}(X W_Q^{(h)}, X W_K^{(h)}, X W_V^{(h)}), \quad h \in \{1, \dots, H\}$$
$$\operatorname{MHA}(X) = \operatorname{Concat}(\operatorname{head}_1, \dots, \operatorname{head}_H) W_O$$
where $W_O \in \mathbb{R}^{(H \cdot d_{\text{head}}) \times d_{\text{model}}}$.

#### Scaling Factor vs Head Dimension Rule
- **Scaling Factor:** $1/\sqrt{d_k}$ depends strictly on the projection dimension $d_{\text{head}}$ of individual queries and keys, not on $d_{\text{model}}$ or $H$.
- **Standard Convention:** Typically $d_{\text{head}} = d_{\text{model}} / H$ so that the concatenated dimension $H \cdot d_{\text{head}} = d_{\text{model}}$ and $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.
- **Non-Standard Feasibility:** $H \cdot d_{\text{head}}$ is not required to equal $d_{\text{model}}$. For example, with $d_{\text{model}}=8$, $H=6$, and $d_{\text{head}}=2$, the concatenated output has dimension $6 \times 2 = 12$. The projection matrix $W_O \in \mathbb{R}^{12 \times 8}$ linearly maps the 12-dimensional concatenated features back to $d_{\text{model}}=8$ without mathematical contradiction.

---

### 2.4 Pre-LayerNorm Transformer Block Architecture
Modern Transformer decoder blocks apply Pre-LayerNorm (LayerNorm before attention and FFN) rather than Post-LayerNorm:
1. **Self-Attention Sub-Layer:**
   $$x^{(1)} = x + \operatorname{MHA}(\operatorname{LayerNorm}(x))$$
2. **Feed-Forward Sub-Layer:**
   $$x^{(2)} = x^{(1)} + \operatorname{FFN}(\operatorname{LayerNorm}(x^{(1)}))$$
where:
$$\operatorname{FFN}(z) = W_2 \operatorname{ReLU}(W_1 z + b_1) + b_2, \quad W_1 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}, \quad W_2 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}}$$

Pre-LayerNorm provides an unobstructed residual stream ($x + f(x)$), stabilizing gradient flow across deep networks and eliminating the need for warm-up learning rate schedules.

---

### 2.5 Analytical Parameter Count Derivation

For TinyGPT configured with:
- Vocabulary Size $V = 25$
- Embedding Dimension $d_{\text{model}} = 48$
- Block Size $T = 32$
- Number of Heads $H = 4$ ($d_{\text{head}} = 48/4 = 12$)
- Transformer Layers $L = 2$

#### Exact Layer Breakdown:
1. **Token Embeddings:**
   $$N_{\text{tok}} = V \times d_{\text{model}} = 25 \times 48 = 1,200$$
2. **Position Embeddings:**
   $$N_{\text{pos}} = T \times d_{\text{model}} = 32 \times 48 = 1,536$$
3. **Per Transformer Block:**
   - **MHA Projections ($Q, K, V$ across 4 heads):**
     $$3 \times H \times (d_{\text{model}} \times d_{\text{head}}) = 3 \times 4 \times (48 \times 12) = 6,912$$
   - **MHA Output Projection ($W_O$ + bias):**
     $$(H \cdot d_{\text{head}}) \times d_{\text{model}} + d_{\text{model}} = (48 \times 48) + 48 = 2,304 + 48 = 2,352$$
   - **LayerNorm 1 ($\gamma, \beta$):**
     $$2 \times d_{\text{model}} = 2 \times 48 = 96$$
   - **FFN Layer 1 ($W_1 + b_1$):**
     $$d_{\text{model}} \times (4 d_{\text{model}}) + 4 d_{\text{model}} = (48 \times 192) + 192 = 9,216 + 192 = 9,408$$
   - **FFN Layer 2 ($W_2 + b_2$):**
     $$(4 d_{\text{model}}) \times d_{\text{model}} + d_{\text{model}} = (192 \times 48) + 48 = 9,216 + 48 = 9,264$$
   - **LayerNorm 2 ($\gamma, \beta$):**
     $$2 \times d_{\text{model}} = 2 \times 48 = 96$$
   - **Total per Block:**
     $$N_{\text{block}} = 6,912 + 2,352 + 96 + 9,408 + 9,264 + 96 = 28,128$$
4. **Final LayerNorm ($\gamma, \beta$):**
   $$N_{\text{ln\_f}} = 2 \times d_{\text{model}} = 96$$
5. **LM Head ($W_{\text{head}} + b_{\text{head}}$):**
   $$N_{\text{lm}} = d_{\text{model}} \times V + V = 48 \times 25 + 25 = 1,200 + 25 = 1,225$$

#### Total Analytical Parameters:
$$N_{\text{total}} = N_{\text{tok}} + N_{\text{pos}} + L \times N_{\text{block}} + N_{\text{ln\_f}} + N_{\text{lm}}$$
$$N_{\text{total}} = 1,200 + 1,536 + (2 \times 28,128) + 96 + 1,225 = \mathbf{60,313}$$

The empirical parameter count from `TinyGPT.count_parameters()` matches this derivation exactly ($60,313$).

---

## 3. Experimental Analysis & Numerical Verification

All quantitative values presented in this section are extracted directly from the unrounded machine outputs in `week05_transformers_ar/results/`.

### 3.1 Attention Variance Scaling Verification
From `results/attention_analysis.json`:
We sampled $N=3,000$ independent pairs $q, k \sim \mathcal{N}(0, I_d)$ across dimensions $d \in [2, 8, 32, 128, 512]$ and evaluated empirical standard deviation $\sigma(q^\top k)$:

| Dimension ($d$) | Empirical Std ($\sigma$) | Theoretical $\sqrt{d}$ | Ratio $\sigma / \sqrt{d}$ |
| :--- | :--- | :--- | :--- |
| **2** | 1.408254 | 1.414214 | 0.995786 |
| **8** | 2.845659 | 2.828427 | 1.006093 |
| **32** | 5.529509 | 5.656854 | 0.977488 |
| **128** | 11.418496 | 11.313708 | 1.009262 |
| **512** | 22.655445 | 22.627417 | 1.001239 |

Across all dimensions, the empirical ratio $\sigma / \sqrt{d}$ remains strictly within $[0.977, 1.009]$, confirming the theoretical derivation.

#### Unscaled vs. Scaled Attention Variance
Testing scaled scores $S / \sqrt{d_k}$ versus unscaled scores $S$ for head dimensions $d_k \in [4, 8, 16, 32, 64]$:

| Head Dim ($d_k$) | Scaled Std $\sigma(S / \sqrt{d_k})$ | Unscaled Std $\sigma(S)$ | Variance Ratio |
| :--- | :--- | :--- | :--- |
| **4** | 0.918966 | 1.837933 | 2.0000 |
| **8** | 1.109610 | 3.138451 | 2.8284 ($\sqrt{8}$) |
| **16** | 0.986494 | 3.945976 | 4.0000 ($\sqrt{16}$) |
| **32** | 0.978238 | 5.533747 | 5.6569 ($\sqrt{32}$) |
| **64** | 1.009576 | 8.076611 | 8.0000 ($\sqrt{64}$) |

Scaling preserves score standard deviation near $1.0$ regardless of head dimension.

---

### 3.2 Training Dynamics & Convergence
From `results/training_loss.json`:
TinyGPT was trained on the repeated canonical corpus for 180 iterations using AdamW ($\text{lr} = 3\times 10^{-3}$, $\text{batch\_size} = 32$, $\text{block\_size} = 32$):

| Step | Evaluation Loss | Notes / Training Stage |
| :--- | :--- | :--- |
| **0** | 3.384801 | Random initialization ($\approx \ln 25 = 3.2189$) |
| **20** | 2.061595 | Rapid character frequency and bigram acquisition |
| **40** | 1.617704 | Matching baseline bigram loss ($1.6013$) |
| **60** | 1.284008 | Acquiring multi-character token sequences |
| **80** | 0.900996 | Resolving phrase-level syntax |
| **100** | 0.587539 | Learning sentence beginnings and transitions |
| **120** | 0.345517 | Long-range context stabilization |
| **140** | 0.257031 | Fine-grained punctuation and ending structure |
| **160** | 0.178116 | High-confidence sentence reproduction |
| **180** | **0.159833** | Fully converged ($>95.2\%$ loss reduction) |

- **Initial Loss:** 3.3848
- **Final Loss:** 0.1598
- **First-10 Step Mean Loss:** 2.8162
- **Last-10 Step Mean Loss:** 0.1586

---

### 3.3 Temperature Scaling in Autoregressive Generation
From `results/generation_samples.json`:
We evaluated autoregressive generation across temperatures $\tau \in \{0.3, 0.7, 0.8, 1.0, 2.0\}$ using prompt `"baana "`, `"attention "`, and an empty prompt `""`:

| Temperature ($\tau$) | Sample Continuation (Prompt: `""`) | Qualitative Characteristics |
| :--- | :--- | :--- |
| **0.3** | `\ngeneration repeats next token prediction.\n\n\nto learn is to c` | Highly deterministic, exact grammatical corpus sentence reconstruction. |
| **0.7** | `\ngenextion lets a token a loook at arlier teans.\na trans to n` | Coherent, minor stochastic character substitutions. |
| **0.8** | `\ngenextion lets a token a loook at arlier teans.\na trans thex` | Moderate creativity with slight syntactic degradation. |
| **1.0** | `\ng model learns from edict tokenns rer prediction.\n\ngeration ` | High variance, frequent non-words and spelling splices. |
| **2.0** | `\ng moswe ptrwe mpreduner p by pexxts tictkieng\ntioonfokernfri` | Near-uniform entropy; complete breakdown into character noise. |

---

### 3.4 Bigram Baseline vs. TinyGPT Comparison
From `results/comparison.json`:

| Feature / Metric | Bigram Model | TinyGPT |
| :--- | :--- | :--- |
| **Architecture** | First-order counting table $P(x_t \mid x_{t-1})$ | 2-Layer Decoder Transformer |
| **Context Length ($T$)** | 1 character | 32 characters |
| **Trainable Parameters** | 625 ($25 \times 25$) | 60,313 |
| **Cross-Entropy Loss** | 1.6013 | 0.1598 |
| **Syntactic Coherence** | False (degenerate repeating loops, e.g. `by token token token...`) | True (coherent grammatical sentences) |
| **Verbatim Sentence Recall** | 0.00% | High (>16.7% exact matching on short sampled prompts) |

---

## 4. Addressing Common Misconceptions

1. **The Fixed-Slicing Fallacy:**
   *Misconception:* Multi-head attention simply slices the input tensor into $H$ chunks along the feature dimension.  
   *Reality:* Heads are **learned linear projections** from the full $d_{\text{model}}$-dimensional space into distinct subspaces ($Q = X W_Q^{(h)}$). Slicing without projection would prevent heads from learning different linear combinations of input features.

2. **Attention Weights vs. Value Vectors:**
   *Misconception:* Attention weights represent the output features of the attention block.  
   *Reality:* Attention weights $\alpha_{ij} \in [0, 1]$ form a normalized scalar weighting matrix indicating *how much* token $i$ attends to token $j$. The output is a linear combination of *Value vectors* ($z_i = \sum_j \alpha_{ij} v_j$).

3. **Query/Key/Value Sources:**
   *Misconception:* Queries, keys, and values come from three different external sources.  
   *Reality:* In decoder self-attention, $Q, K, V$ are all linear projections of the **same input tensor** $X$. In encoder-decoder cross-attention, $Q$ comes from the decoder while $K, V$ originate from the encoder.

4. **Causal Masking during Training vs. Inference:**
   *Misconception:* Causal masking is used during inference to generate one token at a time.  
   *Reality:* Causal masking is used **during parallel training** to allow all $T$ next-token predictions to be computed in a single forward pass while preventing future leakage. During autoregressive generation, generation is inherently sequential: token $t$ is computed from previous tokens $1 \dots t-1$.

---

## 5. Student Reflection & "Think About It" Analysis

### Reflection 1: Self-Attention vs. N-Gram Contextual Capacity
**Question:** *Why does self-attention enable capturing long-range dependencies that N-gram models fundamentally fail on?*  
**Answer:**  
*TODO(student) / DRAFT — rewrite in own words:*
1. **Exponential Parameter Growth vs. Constant Scaling:** An $N$-gram model requires $|\mathcal{V}|^N$ parameters to condition on context length $N$. For $|\mathcal{V}|=25$ and $N=32$, this requires $25^{32} \approx 5.4 \times 10^{44}$ parameters, which is mathematically and computationally impossible to store or estimate from finite data. In contrast, TinyGPT handles $T=32$ with only 60,313 parameters via low-rank query-key matching.
2. **Dynamic Relevance vs. Rigid Adjacency:** Self-attention computes content-dependent weights $\alpha_{ij} \propto q_i^\top k_j$. Token $t$ can attend directly to token $t-20$ with high weight if their query-key dot product is large, bypassing intermediate tokens with path length $O(1)$. N-grams can only propagate information through adjacent transitions.
3. **Data Sparsity:** N-grams assign zero probability to any context string not seen verbatim during training. Transformers map tokens to dense continuous embedding vectors, allowing semantic generalization across novel contexts.

---

### Reflection 2: Causal Masking Mechanics & Training Parallelism
**Question:** *What is the mechanical role of the causal mask during training versus during autoregressive inference?*  
**Answer:**  
*TODO(student) / DRAFT — rewrite in own words:*
1. **Parallel Training Efficiency:** Because all input tokens $(x_1, \dots, x_T)$ and targets $(x_2, \dots, x_{T+1})$ are known ahead of time, the entire sequence can be passed through the network in a single forward matrix multiplication. The causal mask $M$ enforces that position $i$ cannot attend to positions $j > i$, allowing $T$ parallel loss computations without data leakage.
2. **Autoregressive Generation Serialisation:** During inference, tokens are generated one by one. The model takes prefix $x_{1:t}$, extracts logits at position $-1$, samples $x_{t+1}$, appends it to the sequence, and repeats. The causal mask ensures consistent representations between training and inference contexts.
3. **Upper-Triangular Negative Infinity:** Setting $M_{ij} = -\infty$ for $j > i$ ensures that $\exp(-\infty) = 0$ in the softmax denominator and numerator, completely zeroing future token influence.

---

### Reflection 3: Temperature Scaling Dynamics & Entropy
**Question:** *Why does temperature $\tau < 1.0$ sharpen probability distributions, and what are the failure modes of extreme temperature settings ($\tau \to 0$ vs $\tau \to \infty$)?*  
**Answer:**  
*TODO(student) / DRAFT — rewrite in own words:*
1. **Mathematical Mechanism:** Logits $z_i$ are divided by $\tau$ before applying softmax: $P(i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$. When $\tau < 1$, differences between logits are amplified ($z_i / \tau - z_j / \tau = (z_i - z_j)/\tau > z_i - z_j$), pushing probability mass toward the largest logit and lowering distribution entropy.
2. **Failure Mode of $\tau \to 0$ (Greedy Degeneracy):** As $\tau \to 0$, $P(\text{argmax}) \to 1.0$. While outputs become highly grammatical, the model can fall into repetitive, deterministic loops (e.g. repeated sentence fragments) with zero diversity.
3. **Failure Mode of $\tau \to \infty$ (Uniform Noise):** As $\tau \to \infty$, $z_i / \tau \to 0$, causing $\exp(z_i / \tau) \to 1.0$ and $P(i) \to 1 / |\mathcal{V}|$. The distribution becomes completely uniform, generating unstructured character noise without grammatical syntax.

---

## 6. Verification Suite & Mutation Testing Summary

The test suite in `week05_transformers_ar/tests/test_transformers.py` contains 15 automated test cases:
1. **NumPy vs. PyTorch Oracle:** Verified that `causal_self_attention_numpy` and PyTorch `Head` produce identical attention weights and output tensors on shared fixed weights ($< 10^{-6}$ error).
2. **Analytical Parameter Oracle:** Verified that `TinyGPT.count_parameters()` matches the 60,313 analytical derivation.
3. **Causal Invariance Test:** Mutating future tokens in input sequences verified zero change in prefix logits ($< 10^{-6}$ absolute deviation).
4. **Attention Score Variance Scaling Test:** Verified that $\sigma(q^\top k / \sqrt{d})$ stays near $1.0$ across head dimensions.
5. **Temperature & Entropy Invariant Tests:** Verified strict monotonic entropy growth with increasing $\tau$ and deterministic argmax at $\tau \le 10^{-6}$.
6. **Training Convergence Test:** Verified that 180 steps of AdamW drops cross-entropy loss from $>3.0$ to $<0.50$ (empirical result: $0.1598$).
7. **Real Module Monkeypatch Mutation Tests:**
   - *Mutation 1 (Corrupted Unmasked Attention):* Monkeypatching `Head.forward` to remove the causal mask fails the future-leakage test ($> 10^{-4}$ deviation).
   - *Mutation 2 (Missing $\sqrt{d_k}$ Scaling):* Monkeypatching `Head.forward` to remove scaling fails the variance scaling assertion ($\sigma_{\text{unscaled}} > 2.0 \times \sigma_{\text{scaled}}$).
   - *Mutation 3 (Broken Residual Connection):* Monkeypatching `TransformerBlock.forward` to omit $x + \text{sa}(x)$ fails the residual pass-through identity test.
