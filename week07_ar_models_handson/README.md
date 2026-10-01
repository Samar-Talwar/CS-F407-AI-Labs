# CS F407 Lab, Week 7 | Author: Samar Talwar | Not licensed for reuse or submission by others.

# Week 7: Learning Autoregressive Models — Transformers, Ollama & RAG

## Overview
This folder implements **Week 7** of CS F407 (Artificial Intelligence, BITS Pilani Goa, AY 2026-27 Sem 1):

- **T1–T10**: From-first-principles PyTorch Transformer (scaled dot-product attention, causal mask, sinusoidal positional encoding, multi-head self/cross-attention, decoder-only GPT-style model, greedy + temperature/top-k generation).
- **O1–O5**: Ollama REST client wrapper with deterministic offline stub (health check, `PromptTemplate` / `OllamaChain`, `run_newton_query`).
- **R1–R6**: Retrieval-Augmented Generation over a neurosymbolic paper corpus (`rag.ipynb`, AY 2025-26 S2 Lab 3 provenance noted), pure-Python BM25 + TF-IDF retrievers, strict `SYSTEM_RAG` prompt assembly, Recall@K / MRR evaluation.

Sources (all under `original/`; never edited):
- `original/AI_lab_transformers.ipynb`
- `original/Run_Ollama.ipynb`
- `original/Transformer.pdf`
- `original/rag.ipynb` — **from previous semester (AY 2025-26 S2, Lab 3, Dr. Tirtharaj Dash)**.

## Structure

```
week07_ar_models_handson/
  CHECKLIST.md                  # Traceability matrix (T1–T10, O1–O5, R1–R6, Q1–Q7)
  README.md                     # This file
  REPORT.md                     # Answers every handout / notebook question; references real results/
  src/
    __init__.py
    cli.py                      # Unified CLI (transformer / ollama / rag / --all)
    transformer.py              # Attention, PE, MHA, decoder, tokenizer, train, generate
    ollama_client.py            # OllamaClient, PromptTemplate, OllamaChain, run_newton_query
    rag.py                      # Chunking, BM25/TF-IDF, RAGPipeline, metrics
  tests/
    test_transformers_rag.py    # 22 tests (oracle, invariants, gradcheck, mutation, CLI)
  results/
    attention_matrix.json
    loss_curve.png
    transformer_results.json
    ollama_results.json
    rag_results.json
    original_notebook_outputs.json
```

## Running

```powershell
# Using the project's venv (Windows PowerShell)
.venv\Scripts\Activate.ps1
python -m week07_ar_models_handson.src.cli --all
python -m week07_ar_models_handson.src.cli transformer
python -m week07_ar_models_handson.src.cli ollama --force-stub
python -m week07_ar_models_handson.src.cli rag
```

Results are written to `results/` at full precision (no rounding in JSON).

## Quality gates (verified 2026-10-01)
- `ruff check .` — clean after `ruff format`.
- `pytest week07_ar_models_handson/tests/ -v` — 22 passed.
- Fresh-clone gate (`git clone . $env:TEMP\fresh_w07` + ruff + pytest + CLI) — done.
- No fabricated results; all numbers come from `results/` artifacts.

## Provenance / Authorship
- Every source file starts with the required header (`# CS F407 Lab, Week 7 | Author: Samar Talwar ...`).
- No open-source license added (NOTICE.md governs).
- Reflection / "Think About It" answers drafted with AI assistance from the measured results and reviewed by the author.
- Prompt log entry added to `docs/prompt_log.md`.
