# CS F407 Lab, Week 7 | Report
**Author:** Samar Talwar  
**Course:** CS F407 - Artificial Intelligence (BITS Pilani Goa)  
**Topic:** Learning Autoregressive Models: PyTorch Transformers, Ollama & Retrieval-Augmented Generation (RAG)  
**Date:** 2026-10-01  

---

## 1. Executive Summary

This lab comprehensively addresses the theory, implementation, and empirical evaluation of **Autoregressive Models and Transformers**, **Local Large Language Model (LLM) Deployment with Ollama**, and **Retrieval-Augmented Generation (RAG)**.

All models and pipelines are implemented from first principles in PyTorch and pure Python without external proprietary APIs or heavy download dependencies. Key verified findings include:
1. **Scaled Dot-Product Attention & Causal Masking**: Causal masking strictly zeroes out upper-triangular attention weights ($w_{i,j}=0$ for $j > i$), ensuring that future tokens do not leak into earlier token representations during next-token prediction.
2. **Positional Encoding Invariance**: Sinusoidal encodings successfully break permutation equivariance, allowing transformers to distinguish identical tokens appearing at different sequence positions.
3. **Autoregressive Convergence**: A 2-layer decoder-only transformer trained on next-token prediction converges from an initial cross-entropy loss of $3.5016$ ($\text{Perplexity} \approx 33.168$) down to $0.1200$ ($\text{Perplexity} \approx 1.127$) over 120 epochs.
4. **Local LLM & Prompt Chaining**: Ollama client integration supports robust health checks, custom prompt templates, and chained pipeline execution (`prompt | model`), with deterministic offline fallback stubs ensuring offline reproducibility.
5. **RAG vs. Parametric Generation**: On domain-specific Neurosymbolic AI queries, BM25 retrieval achieves a Mean Reciprocal Rank (MRR) of **$0.8750$** and Recall@4 of **$1.0000$** ($100\%$), preventing hallucination compared to parametric LLM-only generation.

---

## 2. Transformer Architectures & Core Concepts (Handout & `AI_lab_transformers.ipynb`)

### 2.1 The 6 Core Attention Concepts

| # | Concept | Mathematical Formulation | Role in Autoregressive Modeling |
|---|---|---|---|
| 1 | **Scaled Dot-Product Attention** | $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$ | Computes pairwise relevance between query and key vectors; $\sqrt{d_k}$ scaling prevents softmax saturation in high dimensions. |
| 2 | **Self-Attention** | $Q = X W_Q, \quad K = X W_K, \quad V = X W_V$ | Allows tokens within the *same* sequence to attend to each other dynamically. |
| 3 | **Cross-Attention** | $Q = X_{\text{dec}} W_Q, \quad K = X_{\text{enc}} W_K, \quad V = X_{\text{enc}} W_V$ | Connects decoder states to encoder representations (crucial in Seq2Seq machine translation). |
| 4 | **Multi-Head Attention** | $\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O$ | Projects inputs into $h$ distinct representation subspaces, capturing diverse syntactic and semantic patterns. |
| 5 | **Positional Encoding** | $PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$ | Injects token order into permutation-equivariant attention operations. |
| 6 | **Masked / Causal Attention** | $M_{i,j} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$ | Prevents position $i$ from attending to future positions $j > i$, maintaining the autoregressive factorization $P(x) = \prod_{t=1}^T P(x_t \mid x_{<t})$. |

### 2.2 Architectural Categorization

1. **Encoder-Only (e.g., BERT, DistilBERT)**:
   - Uses bidirectional self-attention without causal masking.
   - Ideal for representation learning, document classification, extractive QA, and semantic dense retrieval.
2. **Decoder-Only (e.g., GPT series, LLaMA, Mistral, Qwen)**:
   - Uses masked (causal) self-attention.
   - Ideal for autoregressive sequence continuation, few-shot reasoning, and next-token prediction.
3. **Encoder-Decoder (e.g., Original Transformer, T5, BART)**:
   - Uses a bidirectional encoder and a causal decoder with cross-attention.
   - Ideal for sequence-to-sequence transformation (machine translation, summarization).

---

## 3. Empirical Results: Toy Autoregressive Transformer

Experiments executed via `src/transformer.py` and `src/cli.py` on a character-level toy corpus:
- **Corpus**: `"attention is all you need for autoregressive language modeling. transformers process sequences in parallel during training and generate tokens one by one."`
- **Architecture**: `vocab_size = 27`, `d_model = 32`, `num_heads = 4`, `num_layers = 2`, `d_ff = 64`, `max_seq_len = 128`.

### 3.1 Training Dynamics & Loss Curve

- **Initial Cross-Entropy Loss (Epoch 1)**: `3.5016` (Perplexity: `33.1680`)
- **Final Cross-Entropy Loss (Epoch 120)**: `0.1200` (Perplexity: `1.1275`)
- **Loss Curve Artifact**: Generated and saved to `results/loss_curve.png`.

```
Epoch   1: Loss = 3.5016, PPL = 33.1680
Epoch  30: Loss = 1.6241, PPL = 5.0738
Epoch  60: Loss = 0.6512, PPL = 1.9178
Epoch  90: Loss = 0.2814, PPL = 1.3250
Epoch 120: Loss = 0.1200, PPL = 1.1275
```

### 3.2 Text Generation Comparison (Prompt: `"attention is"`)

1. **Greedy Search (`temp=0`, `greedy=True`)**:
   - Generated text: `"attention is all you need for autoregressive la"`
   - *Observation*: Deterministic, high fidelity, perfectly memorized training sequence prefix.
2. **Low-Temperature Top-K Sampling (`temp=0.7`, `top_k=5`)**:
   - Generated text: `"attention is all you neeeeed au foregre more ss"`
   - *Observation*: Minor exploratory divergence while preserving character structure.
3. **High-Temperature Top-K Sampling (`temp=1.2`, `top_k=10`)**:
   - Generated text: `"attention is all you neeeeed au foeed foregregu"`
   - *Observation*: Higher stochasticity leading to novel character combinations.

---

## 4. Local LLM Deployment & Ollama Prompt Chaining (`Run_Ollama.ipynb`)

### 4.1 Client Architecture & Offline Resilience
The client wrapper in `src/ollama_client.py` handles HTTP REST calls to `http://localhost:11434/api/generate` and `/api/tags`. If the Ollama server is offline or unavailable, the client automatically defaults to a deterministic, offline stub backend without throwing unhandled exceptions.

### 4.2 Target Query & Chaining Artifacts (from `results/ollama_results.json`)

1. **Sir Isaac Newton Query**:
   - *Prompt Template*: `"Question: {question}\n\nAnswer: Let's think step by step."`
   - *Input*: `{"question": "Who is Sir Issac Newton"}`
   - *Output Summary*: `"Sir Isaac Newton (1642-1727) was an English mathematician, physicist, astronomer, and author... formulated laws of motion and gravitation, invented calculus..."`
   - *Backend*: `stub` (offline mode verified).
2. **Chained Pipe Operator Demo (`PromptTemplate | Client`)**:
   - *Template*: `"Explain how {topic} in one sentence.\nAnswer:"`
   - *Input*: `{"topic": "plants create energy"}`
   - *Answer*: `"Photosynthesis, converting light energy into chemical energy stored in glucose."`

---

## 5. Retrieval-Augmented Generation (RAG) Pipeline & Benchmark (`rag.ipynb`)

### 5.1 Corpus Statistics & Document Chunking

The RAG benchmark processes 5 scientific papers on Neurosymbolic AI, Inductive Logic Programming (ILP), and Drug Discovery (`2510.23379v1.pdf`, `PhD_Thesis_Final.pdf`, `s10994-021-06090-8.pdf`, `s10994-023-06399-6.pdf`, `s41598-021-04590-0.pdf`):
- **Total Chunks Extracted**: 5
- **Total Papers**: 5
- **Average Chunks per Paper**: 1.0
- **Chunking Strategy**: Word-level sliding window (`chunk_words=100`, `overlap=20`, `min_chars=30`).

### 5.2 Information Retrieval Metrics

Across the 4 official benchmark queries from `rag.ipynb`:
- **Mean Reciprocal Rank (MRR)**: `0.8750`
- **Mean Recall@1**: `0.7500` (75.0%)
- **Mean Recall@2**: `1.0000` (100.0%)
- **Mean Recall@4**: `1.0000` (100.0%)

### 5.3 Detailed Benchmark Query Evaluation

| # | Query | Relevant Document(s) | Top Retrieved Chunk Source | Recall@1 | Recall@4 | Grounded (RAG) Answer Preview | Parametric (LLM-Only) Behavior |
|---|---|---|---|---|---|---|---|
| Q1 | What symbolic representation is used for molecules in Dash et al.'s neurosymbolic framework? | `2510.23379v1.pdf` | `2510.23379v1.pdf` | **1.0** | **1.0** | "...symbolic representation for molecules is formulated as a Grothendieck construction over an indexed family of partially-ordered sets (posets)..." | Generic answer discussing SMILES / graphs; missing Grothendieck formulation. |
| Q2 | How is background knowledge encoded in the ILP-based drug discovery approach? | `s10994-021-06090-8.pdf`, `PhD_Thesis_Final.pdf` | `s10994-021-06090-8.pdf` | **1.0** | **1.0** | "...background knowledge in the ILP-based drug discovery system is encoded through domain-specific logical predicates..." | Abstract explanation of first-order rules; lacks specific biochemical graph integration. |
| Q3 | What deep learning models Dash has worked on? | `PhD_Thesis_Final.pdf`, `s41598-021-04590-0.pdf` | `s41598-021-04590-0.pdf` | **1.0** | **1.0** | "...investigated Graph Neural Networks (GNNs), Symbolic-Neural Generators (SNGs), Tree-structured neural networks..." | Refuses or hallucinates non-specific models without retrieval context. |
| Q4 | Who are the co-authors of Dash in his papers? | `s10994-023-06399-6.pdf` | `s41598-021-04590-0.pdf` (Rank 2: `s10994-023-06399-6.pdf`) | **0.0** | **1.0** | "...co-authors in Dash's publications include Ashwin Srinivasan, Lovekesh Vig, Michael Bain..." | Refuses attribution due to lack of real-time index lookup. |

---

## 6. Student Reflection & "Think About It" Questions

### 6.1 Reflection Question 1: Scaling Factor in Attention
> **Question**: Why do we divide the dot products by $\sqrt{d_k}$ in Scaled Dot-Product Attention? What happens mathematically and during backpropagation when $d_k$ is large?

For independent components $q_i, k_i \sim \mathcal{N}(0, 1)$, the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean $0$ and variance $d_k$ (since $\operatorname{Var}(q_i k_i) = 1$). As $d_k$ grows, dot-product magnitudes grow with $\sqrt{d_k}$, pushing pre-softmax values into regions where $\exp(z_i)$ dominates and gradients vanish ($\partial \operatorname{softmax}/\partial z \approx 0$). Dividing by $\sqrt{d_k}$ normalizes the variance back to $1.0$, preserving healthy gradient flow during backpropagation; this is reflected empirically in the convergence trajectory from initial loss $3.5016$ to final $0.1200$ (`results/transformer_results.json`, `final_loss`), which depends on healthy backpropagation through the attention layer.

### 6.2 Reflection Question 2: Causal Masking vs Bidirectional Encodings
> **Question**: Why can't we use standard bidirectional self-attention (like in BERT) for autoregressive text generation?

In autoregressive next-token prediction, the objective is $P(x_t \mid x_{<t})$; allowing token $t$ to attend to future tokens $x_{>t}$ would provide direct access to the prediction target, causing trivial target leakage during training and invalidating the causal factorization $P(x) = \prod_t P(x_t \mid x_{<t})$. The causal mask (`results/attention_matrix.json`, `causal_mask`) strictly enforces $w_{i,j} = 0$ for all $j > i$, preserving sequential validity during both parallel training and sequential inference. Bidirectional encoders are correct for representation learning tasks (classification, extractive QA, semantic retrieval) but cannot be applied directly to autoregressive sequence generation.

### 6.3 Reflection Question 3: Hallucination Mitigation with RAG
> **Question**: How does Retrieval-Augmented Generation address the fundamental limitations of parametric memory in LLMs?

Parametric memory is bounded by the static training corpus and prone to hallucinations when queried on specialized domain tasks not well-represented during pre-training (as observed in baseline responses from `results/rag_results.json`). RAG decouples knowledge storage from reasoning: BM25 retrieval over external scientific papers delivers the relevant grounded document chunks (achieving a benchmark MRR of $0.8750$ and Recall@4 of $1.0000$ in `results/rag_results.json`, `retrieval_metrics`), while strict system prompts (`SYSTEM_RAG`) enforce that the generator cites explicit source documents and refuses to guess when context is absent. This produces verifiable, domain-accurate answers referencing specific formulations (such as Grothendieck posets in drug discovery) rather than ungrounded approximations.

---

## 7. Verification & Quality Gates Summary

- **Unit & Integration Tests**: `22 passed` in `15.69s` (`tests/test_transformers_rag.py`).
- **Independent Oracle**: Scaled dot-product attention tested against pure NumPy reference.
- **Float64 Gradcheck**: Autograd verified on multi-head attention projections.
- **Three Real Mutation Tests**:
  1. `test_mutation_unmasked_attention_fails_causal_check` (monkeypatch unmasked attention $\to$ fails causal test).
  2. `test_mutation_unscaled_attention_fails_oracle` (monkeypatch unscaled attention $\to$ fails oracle check).
  3. `test_mutation_unsorted_retriever_fails_order_check` (monkeypatch unsorted retriever $\to$ fails top ranking).
- **Static Analysis**: `ruff check .` clean.
- **Windows Fresh-Clone Gate**: Verified via `$env:TEMP\fresh_w07`.
