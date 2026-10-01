# CS F407 Lab, Week 7 | Checklist & Traceability Matrix
**Author:** Samar Talwar  
**Course:** CS F407 - Artificial Intelligence (BITS Pilani Goa)  
**Topic:** Learning Autoregressive Models: Transformers, Ollama & Retrieval-Augmented Generation (RAG)

---

## 1. Source Documents & Provenance

| Source File | Provenance & Description |
|---|---|
| `original/AI_lab_transformers.ipynb` | Official AY 2026-27 S1 Lab notebook covering Hugging Face Transformers: Encoder-Decoder (Seq2Seq), Decoder-Only (GPT-2), Encoder-Only (BERT/DistilBERT Sentiment Analysis). |
| `original/Run_Ollama.ipynb` | Official AY 2026-27 S1 Lab notebook covering local LLM deployment with Ollama, Colab xterm, LangChain Ollama wrapper (`OllamaLLM`), and prompt chaining. |
| `original/Transformer.pdf` | Official Lab Handout / Slides covering Transformer origins (Machine Translation, Seq2Seq), Encoder-only vs Decoder-only vs Encoder-Decoder, 6 core attention concepts (Attention, Self-Attention, Cross-Attention, Multi-Head Attention, Positional Encoding, Masked Attention), and Local LLM execution. |
| `original/rag.ipynb` | Course archive notebook (AY 2025-26 S2 Lab 3, Dr. Tirtharaj Dash) demonstrating full RAG pipeline over Neurosymbolic AI & Drug Discovery papers: PDF parsing, word chunking with overlap, embedding, FAISS indexing, similarity retrieval, system prompts, grounded generation vs parametric LLM-only. |

---

## 2. Traceability Matrix: Concepts, Tasks & Exercises

### 2.1 Transformer Architectures & Core Concepts (from `Transformer.pdf` & `AI_lab_transformers.ipynb`)

| # | Concept / Requirement | Source Reference | Satisfied By (Source Code) | Test Coverage | Results Artifact |
|---|---|---|---|---|---|
| T1 | **Initial Use Case: Machine Translation** (Seq2Seq: e.g. English $\to$ German / French) | `Transformer.pdf` p. 1; `AI_lab_transformers.ipynb` Cells 1-4 | `src/transformer.py` (`CrossAttention`, `EncoderDecoderStub`) | `tests/test_transformers_rag.py::test_cross_attention_shape` | `results/original_notebook_outputs.json` |
| T2 | **Encoder-Only Models** (BERT / DistilBERT: semantic search, document retrieval, classification) | `Transformer.pdf` p. 1; `AI_lab_transformers.ipynb` Cells 7-8 | `src/transformer.py` (bidirectional attention block), `src/rag.py` | `tests/test_transformers_rag.py::test_bidirectional_attention` | `results/original_notebook_outputs.json` |
| T3 | **Decoder-Only Models** (GPT-style: causal text generation, next token prediction) | `Transformer.pdf` p. 2; `AI_lab_transformers.ipynb` Cells 5-6 | `src/transformer.py` (`DecoderOnlyTransformer`, `generate_text`) | `tests/test_transformers_rag.py::test_autoregressive_generation` | `results/transformer_results.json` |
| T4 | **Scaled Dot-Product Attention** ($Attention(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$) | `Transformer.pdf` p. 3 (Concept 1) | `src/transformer.py` (`scaled_dot_product_attention`) | `tests/test_transformers_rag.py::test_scaled_dot_product_attention_oracle` | `results/attention_matrix.json` |
| T5 | **Self-Attention** (Queries, Keys, Values from same sequence) | `Transformer.pdf` p. 3 (Concept 2) | `src/transformer.py` (`MultiHeadAttention` self-attention mode) | `tests/test_transformers_rag.py::test_self_attention_probabilities_sum_to_one` | `results/attention_matrix.json` |
| T6 | **Cross-Attention** (Queries from decoder, Keys/Values from encoder) | `Transformer.pdf` p. 3 (Concept 3) | `src/transformer.py` (`MultiHeadAttention` cross-attention mode) | `tests/test_transformers_rag.py::test_cross_attention_different_seq_lengths` | `results/transformer_results.json` |
| T7 | **Multi-Head Attention** ($h$ parallel projections, concat, $W_O$) | `Transformer.pdf` p. 3 (Concept 4) | `src/transformer.py` (`MultiHeadAttention`) | `tests/test_transformers_rag.py::test_multihead_attention_shapes_and_grad` | `results/transformer_results.json` |
| T8 | **Positional Encoding** (Sinusoidal positional encoding, breaking permutation equivariance) | `Transformer.pdf` p. 3 (Concept 5) | `src/transformer.py` (`PositionalEncoding`) | `tests/test_transformers_rag.py::test_positional_encoding_breaks_permutation_equivariance` | `results/transformer_results.json` |
| T9 | **Masked Attention / Causal Mask** (Upper triangular mask preventing future token leakage) | `Transformer.pdf` p. 3 (Concept 6) | `src/transformer.py` (`create_causal_mask`, `scaled_dot_product_attention`) | `tests/test_transformers_rag.py::test_causal_mask_future_independence` | `results/transformer_results.json` |
| T10 | **Autoregressive Model Training & Loss Curve** (CE loss decreasing on toy corpus) | `AI_lab_transformers.ipynb` Cells 5-6 | `src/transformer.py` (`train_autoregressive_model`) | `tests/test_transformers_rag.py::test_training_loss_decreases_deterministic` | `results/loss_curve.png`, `results/transformer_results.json` |

---

### 2.2 Local LLM Execution & Ollama Wrapper (from `Run_Ollama.ipynb` & `Transformer.pdf` p. 4)

| # | Concept / Requirement | Source Reference | Satisfied By (Source Code) | Test Coverage | Results Artifact |
|---|---|---|---|---|---|
| O1 | **Ollama Client Interface & Health Check** | `Run_Ollama.ipynb` Cells 1-3 | `src/ollama_client.py` (`OllamaClient.health_check`, `list_models`) | `tests/test_transformers_rag.py::test_ollama_health_check_stub_and_real` | `results/ollama_results.json` |
| O2 | **Offline Deterministic Stub & Graceful Fallback** | `Run_Ollama.ipynb` Cell 2 (offline / missing command) | `src/ollama_client.py` (`OllamaClient`, `StubOllamaTransport`) | `tests/test_transformers_rag.py::test_ollama_graceful_failure_when_absent` | `results/ollama_results.json` |
| O3 | **ChatPromptTemplate & Prompt Chaining** ("Question: {q} \n\n Answer: Let's think step by step.") | `Run_Ollama.ipynb` Cell 4 | `src/ollama_client.py` (`PromptTemplate`, `OllamaChain`) | `tests/test_transformers_rag.py::test_ollama_chain_prompt_formatting` | `results/ollama_results.json` |
| O4 | **Target Query Execution** ("Who is Sir Issac Newton") | `Run_Ollama.ipynb` Cell 4 | `src/ollama_client.py` (`run_newton_query`) | `tests/test_transformers_rag.py::test_newton_query_execution` | `results/ollama_results.json` |
| O5 | **Transport Request/Response Builder & Parser** | `Run_Ollama.ipynb` Cells 3-4 | `src/ollama_client.py` (`build_generate_payload`, `parse_response`) | `tests/test_transformers_rag.py::test_ollama_request_parsing_monkeypatch` | `results/ollama_results.json` |

---

### 2.3 Retrieval-Augmented Generation (RAG) Pipeline (from `rag.ipynb`)

| # | Concept / Requirement | Source Reference | Satisfied By (Source Code) | Test Coverage | Results Artifact |
|---|---|---|---|---|---|
| R1 | **PDF / Text Loading & Overlapping Chunking** (`chunk_words=200`, `overlap=40`, min length=60) | `rag.ipynb` Cells 1-3 | `src/rag.py` (`extract_text_chunks`, `DocumentChunk`) | `tests/test_transformers_rag.py::test_chunking_preserves_text_and_overlap` | `results/rag_results.json` |
| R2 | **Corpus Statistics & Paper Distribution** (Count chunks, averages, source metadata) | `rag.ipynb` Cell 12 | `src/rag.py` (`CorpusStats`, `compute_corpus_stats`) | `tests/test_transformers_rag.py::test_corpus_statistics_calculation` | `results/rag_results.json` |
| R3 | **Retriever: Lexical TF-IDF / BM25 & Dense Cosine Interface** | `rag.ipynb` Cells 3-5 | `src/rag.py` (`BM25Retriever`, `TFIDFRetriever`, `BaseRetriever`) | `tests/test_transformers_rag.py::test_retriever_returns_known_chunk_sorted` | `results/rag_results.json` |
| R4 | **Strict System Prompt Assembly** (`SYSTEM_PLAIN` vs `SYSTEM_RAG`) | `rag.ipynb` Cell 8 | `src/rag.py` (`build_prompt`, `SYSTEM_PLAIN`, `SYSTEM_RAG`) | `tests/test_transformers_rag.py::test_prompt_assembly_structure` | `results/rag_results.json` |
| R5 | **Parametric LLM-Only vs RAG Grounded Generation** | `rag.ipynb` Cells 8-10 | `src/rag.py` (`llm_only`, `rag_answer`, `RAGPipeline`) | `tests/test_transformers_rag.py::test_llm_only_vs_rag_grounded_answer` | `results/rag_results.json` |
| R6 | **Evaluation on Benchmark Queries (Recall@K, MRR)** | `rag.ipynb` Cells 10-11 | `src/rag.py` (`evaluate_rag`, `recall_at_k`, `mean_reciprocal_rank`) | `tests/test_transformers_rag.py::test_evaluation_metrics_exact_values` | `results/rag_results.json` |

---

### 2.4 Quality Gates & Testing Standards

| # | Requirement | Verification Method | Status |
|---|---|---|---|
| Q1 | **Independent Oracle**: Scaled dot-product attention checked against pure NumPy oracle | `tests/test_transformers_rag.py::test_scaled_dot_product_attention_oracle` | Verified / Passed |
| Q2 | **Mathematical Invariants**: Attention row softmax sums to 1.0; causal mask eliminates future attention | `tests/test_transformers_rag.py::test_self_attention_probabilities_sum_to_one`, `test_causal_mask_future_independence` | Verified / Passed |
| Q3 | **Permutation Equivariance Check**: Invariance without PE, broken with PE | `tests/test_transformers_rag.py::test_positional_encoding_breaks_permutation_equivariance` | Verified / Passed |
| Q4 | **PyTorch Gradcheck**: Float64 autograd verification on attention layer | `tests/test_transformers_rag.py::test_float64_attention_gradcheck` | Verified / Passed |
| Q5 | **Three Real Mutation Tests** (monkeypatching actual `src` functions):<br>1. Remove causal mask<br>2. Remove $1/\sqrt{d_k}$ scaling<br>3. Unsorted retriever | `tests/test_transformers_rag.py::test_mutation_unmasked_attention_fails_causal_check`<br>`tests/test_transformers_rag.py::test_mutation_unscaled_attention_fails_oracle`<br>`tests/test_transformers_rag.py::test_mutation_unsorted_retriever_fails_order_check` | Verified / Passed |
| Q6 | **CLI Reproducibility**: `python -m week07_ar_models_handson.src.cli` generates all results | CLI execution writes to `results/` | Verified / Passed |
| Q7 | **Fresh-Clone Gate**: `git clone . $env:TEMP\fresh_w07` passes `ruff` and `pytest` | Automated gate in Windows PowerShell | Verified / Passed |
