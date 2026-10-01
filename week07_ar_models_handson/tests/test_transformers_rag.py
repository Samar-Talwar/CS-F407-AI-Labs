# CS F407 Lab, Week 7 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Comprehensive Pytest Suite for Week 7: Transformers, Ollama Client & RAG Pipeline.

Verifies:
- Independent NumPy Oracle for Scaled Dot-Product Attention
- Attention invariants (rows sum to 1.0, causal upper triangle is 0)
- Positional Encoding breaks permutation equivariance
- PyTorch gradcheck in float64
- Autoregressive model training loss reduction & deterministic generation
- Ollama client offline stub, error handling, prompt templating, and chain invocation
- RAG text chunking, corpus statistics, TF-IDF & BM25 ranking, and metrics (Recall@K, MRR)
- Three real mutation tests (omitting causal mask, omitting scaling factor, unsorted retrieval)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from week07_ar_models_handson.src.cli import (
    run_ollama_experiments,
    run_rag_experiments,
    run_transformer_experiments,
)
from week07_ar_models_handson.src.ollama_client import (
    OllamaChain,
    OllamaClient,
    PromptTemplate,
    run_newton_query,
)
from week07_ar_models_handson.src.rag import (
    BM25Retriever,
    DocumentChunk,
    RAGPipeline,
    TFIDFRetriever,
    compute_corpus_stats,
    evaluate_rag,
    extract_text_chunks,
    get_default_neurosym_corpus,
    mean_reciprocal_rank,
    recall_at_k,
)
from week07_ar_models_handson.src.transformer import (
    DecoderOnlyTransformer,
    MultiHeadAttention,
    PositionalEncoding,
    SimpleTokenizer,
    create_causal_mask,
    generate_text,
    scaled_dot_product_attention,
    train_autoregressive_model,
)

# =====================================================================
# 1. Independent NumPy Oracle & Core Attention Invariants
# =====================================================================


def numpy_scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy reference oracle for scaled dot-product attention."""
    d_k = q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(d_k)

    # q: (..., seq_q, d_k), k: (..., seq_k, d_k) -> scores: (..., seq_q, seq_k)
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale

    if mask is not None:
        # Where mask is True, fill with -1e9
        scores = np.where(mask, -1e9, scores)

    # Softmax along last dimension
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    weights = np.nan_to_num(weights, nan=0.0)

    output = np.matmul(weights, v)
    return output, weights


def test_scaled_dot_product_attention_oracle() -> None:
    """Compare PyTorch scaled_dot_product_attention against independent NumPy oracle."""
    torch.manual_seed(101)
    np.random.seed(101)

    batch, seq_len, d_k, d_v = 2, 5, 8, 8
    q_np = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    k_np = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    v_np = np.random.randn(batch, seq_len, d_v).astype(np.float32)

    q_pt = torch.from_numpy(q_np)
    k_pt = torch.from_numpy(k_np)
    v_pt = torch.from_numpy(v_np)

    pt_out, pt_weights = scaled_dot_product_attention(q_pt, k_pt, v_pt)
    np_out, np_weights = numpy_scaled_dot_product_attention(q_np, k_np, v_np)

    np.testing.assert_allclose(pt_out.detach().numpy(), np_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pt_weights.detach().numpy(), np_weights, rtol=1e-4, atol=1e-4)


def test_self_attention_probabilities_sum_to_one() -> None:
    """Verify that every row of the attention weight matrix sums to 1.0."""
    torch.manual_seed(202)
    q = torch.randn(3, 6, 16)
    k = torch.randn(3, 6, 16)
    v = torch.randn(3, 6, 16)

    _, weights = scaled_dot_product_attention(q, k, v)
    row_sums = weights.sum(dim=-1)
    expected_ones = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, expected_ones, atol=1e-6)


def test_causal_mask_future_independence() -> None:
    """Verify causal mask prevents token i from attending to tokens at j > i."""
    seq_len = 5
    d_model = 16
    mask = create_causal_mask(seq_len)

    q = torch.randn(1, seq_len, d_model)
    k = torch.randn(1, seq_len, d_model)
    v = torch.randn(1, seq_len, d_model)

    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    w_mat = weights.squeeze(0).detach().numpy()

    # For all i < j, w_mat[i, j] must be 0.0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert w_mat[i, j] == pytest.approx(0.0, abs=1e-6), (
                f"Future attention leaked at ({i}, {j})"
            )

    # Mutating future key/value at position 4 should NOT affect output at position 0
    k_mutated = k.clone()
    v_mutated = v.clone()
    k_mutated[0, 4, :] += 50.0
    v_mutated[0, 4, :] -= 100.0

    out1, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    out2, _ = scaled_dot_product_attention(q, k_mutated, v_mutated, mask=mask)

    # Position 0 must be identical
    assert torch.allclose(out1[0, 0, :], out2[0, 0, :], atol=1e-5)


# =====================================================================
# 2. Positional Encoding & Permutation Equivariance
# =====================================================================


def test_positional_encoding_breaks_permutation_equivariance() -> None:
    """
    Without positional encoding, a pure self-attention layer is permutation equivariant:
    f(P x) = P f(x). With positional encoding, this equivariance is broken.
    """
    torch.manual_seed(303)
    seq_len, d_model = 4, 16
    x = torch.randn(1, seq_len, d_model)
    perm = torch.tensor([3, 1, 0, 2])
    x_perm = x[:, perm, :]

    # Pure attention without PE
    mha_no_pe = MultiHeadAttention(d_model=d_model, num_heads=2, is_causal=False)
    out_orig, _ = mha_no_pe(x)
    out_perm_input, _ = mha_no_pe(x_perm)
    out_perm_expected = out_orig[:, perm, :]

    # Invariance holds without PE
    assert torch.allclose(out_perm_input, out_perm_expected, atol=1e-4)

    # With Positional Encoding
    pe = PositionalEncoding(d_model=d_model, max_len=32)
    x_pe = pe(x)
    x_perm_pe = pe(x_perm)

    out_pe_orig, _ = mha_no_pe(x_pe)
    out_pe_perm_input, _ = mha_no_pe(x_perm_pe)
    out_pe_perm_expected = out_pe_orig[:, perm, :]

    # Positional encoding breaks permutation equivariance
    diff = torch.norm(out_pe_perm_input - out_pe_perm_expected).item()
    assert diff > 0.05, f"Expected broken equivariance with PE, got diff {diff}"


def test_cross_attention_different_seq_lengths() -> None:
    """Verify cross-attention correctly handles query length != key/value length."""
    d_model = 32
    num_heads = 4
    mha_cross = MultiHeadAttention(d_model=d_model, num_heads=num_heads, is_causal=False)

    batch_size = 2
    seq_len_q = 5
    seq_len_kv = 12

    q = torch.randn(batch_size, seq_len_q, d_model)
    kv = torch.randn(batch_size, seq_len_kv, d_model)

    out, weights = mha_cross(query=q, key=kv, value=kv)
    assert out.shape == (batch_size, seq_len_q, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len_q, seq_len_kv)


def test_float64_attention_gradcheck() -> None:
    """Verify PyTorch analytical gradients for scaled dot-product attention using gradcheck."""
    torch.manual_seed(404)
    q = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)

    def func(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        out, _ = scaled_dot_product_attention(q_in, k_in, v_in)
        return out

    assert torch.autograd.gradcheck(func, (q, k, v), eps=1e-6, atol=1e-4)


# =====================================================================
# 3. Autoregressive Model Training & Generation
# =====================================================================


def test_training_loss_decreases_deterministic() -> None:
    """Verify toy autoregressive transformer trains deterministically and achieves low loss."""
    toy_text = "the quick brown fox jumps over the lazy dog"
    tokenizer = SimpleTokenizer.from_text(toy_text, mode="char")

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=64,
        max_seq_len=64,
    )

    res = train_autoregressive_model(model, toy_text, tokenizer, epochs=100, lr=5e-3, seed=42)
    init_loss = res["loss_history"][0]["loss"]
    final_loss = res["final_loss"]

    assert final_loss < init_loss * 0.25, (
        f"Loss did not decrease sufficiently: {init_loss} -> {final_loss}"
    )
    assert final_loss < 1.0, f"Expected final loss < 1.0, got {final_loss}"


def test_autoregressive_generation_greedy() -> None:
    """Verify greedy generation produces deterministic continuation."""
    toy_text = "abcde abcde abcde"
    tokenizer = SimpleTokenizer.from_text(toy_text, mode="char")

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        max_seq_len=32,
    )
    train_autoregressive_model(model, toy_text, tokenizer, epochs=80, lr=8e-3, seed=42)

    gen = generate_text(
        model, prompt="abc", tokenizer=tokenizer, max_new_tokens=5, greedy=True, seed=42
    )
    assert gen.startswith("abc")
    assert len(gen) >= 4


# =====================================================================
# 4. Ollama Client & Prompt Chaining Tests
# =====================================================================


def test_ollama_health_check_stub_and_real() -> None:
    """Verify Ollama client health check and model listing in stub mode."""
    client_stub = OllamaClient(force_stub=True)
    assert not client_stub.health_check()
    models = client_stub.list_models()
    assert "mistral" in models
    assert len(models) >= 3


def test_ollama_graceful_failure_when_absent() -> None:
    """Verify client falls back to stub gracefully when targeting an invalid endpoint."""
    client_invalid = OllamaClient(base_url="http://127.0.0.1:9999", timeout=0.1, force_stub=False)
    resp = client_invalid.generate(prompt="Who is Sir Issac Newton")
    assert resp.backend == "stub"
    assert "Newton" in resp.text


def test_ollama_chain_prompt_formatting() -> None:
    """Verify LangChain-style PromptTemplate and OllamaChain | operator."""
    template = PromptTemplate.from_template(
        "Question: {question}\n\nAnswer: Let's think step by step."
    )
    formatted = template.format(question="What is gravity?")
    assert "Question: What is gravity?" in formatted
    assert "Let's think step by step." in formatted

    client = OllamaClient(force_stub=True)
    chain = template | client
    assert isinstance(chain, OllamaChain)
    res = chain.invoke({"question": "Who is Sir Issac Newton"})
    assert isinstance(res, str)
    assert "Newton" in res


def test_newton_query_execution() -> None:
    """Verify execution of the exact Newton query from Run_Ollama.ipynb."""
    res = run_newton_query()
    assert res["query"] == "Who is Sir Issac Newton"
    assert "Sir Isaac Newton" in res["answer"]
    assert res["backend"] in ["stub", "real"]


# =====================================================================
# 5. RAG Pipeline, Chunking & Evaluation Tests
# =====================================================================


def test_chunking_preserves_text_and_overlap() -> None:
    """Verify word-level chunking produces expected overlap, stride, and word offsets."""
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = extract_text_chunks(text, source="test.pdf", chunk_words=20, overlap=5, min_chars=10)

    assert len(chunks) > 0
    # Stride is 20 - 5 = 15
    assert chunks[0].word_offset == 0
    assert chunks[1].word_offset == 15
    assert chunks[2].word_offset == 30
    assert chunks[0].source == "test.pdf"

    # Verify overlap words match
    words_c0 = chunks[0].text.split()
    words_c1 = chunks[1].text.split()
    assert words_c0[-5:] == words_c1[:5]


def test_corpus_statistics_calculation() -> None:
    """Verify compute_corpus_stats correctly calculates counts and paper distribution."""
    chunks = [
        DocumentChunk(text="Chunk 1", source="paper_A.pdf", word_offset=0, index=0),
        DocumentChunk(text="Chunk 2", source="paper_A.pdf", word_offset=10, index=1),
        DocumentChunk(text="Chunk 3", source="paper_B.pdf", word_offset=0, index=2),
    ]
    stats = compute_corpus_stats(chunks)
    assert stats["total_chunks"] == 3
    assert stats["total_papers"] == 2
    assert stats["avg_chunks_per_paper"] == 1.5
    assert stats["paper_distribution"][0]["paper"] == "paper_A.pdf"
    assert stats["paper_distribution"][0]["chunks"] == 2


def test_retriever_returns_known_chunk_sorted() -> None:
    """Verify BM25 and TF-IDF retrievers return correct source paper for domain queries."""
    corpus = get_default_neurosym_corpus()
    bm25 = BM25Retriever(corpus)
    tfidf = TFIDFRetriever(corpus)

    query = "Grothendieck construction partially-ordered sets"
    bm25_hits = bm25.retrieve(query, k=2)
    tfidf_hits = tfidf.retrieve(query, k=2)

    assert len(bm25_hits) == 2
    assert len(tfidf_hits) == 2
    assert bm25_hits[0]["source"] == "2510.23379v1.pdf"
    assert tfidf_hits[0]["source"] == "2510.23379v1.pdf"
    assert bm25_hits[0]["score"] > bm25_hits[1]["score"]


def test_llm_only_vs_rag_grounded_answer() -> None:
    """Verify RAG pipeline provides grounded source attribution while LLM-only reports lack."""
    pipeline = RAGPipeline(chunks=get_default_neurosym_corpus())
    q = "What symbolic representation is used for molecules in Dash et al. framework?"

    llm_ans = pipeline.llm_only(q)
    rag_ans, hits = pipeline.rag_answer(q, k=2)

    assert "Stub Parametric LLM-Only Answer" in llm_ans
    assert "Grothendieck construction" in rag_ans
    assert "2510.23379v1.pdf" in rag_ans
    assert len(hits) == 2


def test_evaluation_metrics_exact_values() -> None:
    """Verify Recall@K and MRR with known hand-computed cases."""
    # Query 1: relevant=['A'], retrieved=['B', 'A', 'C'] -> rank=2, RR=0.5, R@1=0, R@2=1.0
    # Query 2: relevant=['C', 'D'], retrieved=['C', 'E', 'F'] -> rank=1, RR=1.0, R@1=0.5, R@2=0.5
    retrieved_list = [["B", "A", "C"], ["C", "E", "F"]]
    relevant_list = [["A"], ["C", "D"]]

    mrr = mean_reciprocal_rank(retrieved_list, relevant_list)
    assert mrr == pytest.approx((0.5 + 1.0) / 2.0)

    r1_q1 = recall_at_k(retrieved_list[0], relevant_list[0], k=1)
    r2_q1 = recall_at_k(retrieved_list[0], relevant_list[0], k=2)
    assert r1_q1 == 0.0
    assert r2_q1 == 1.0

    r1_q2 = recall_at_k(retrieved_list[1], relevant_list[1], k=1)
    assert r1_q2 == 0.5


def test_full_rag_evaluation_suite() -> None:
    """Run full evaluation suite across the 4 official benchmark queries."""
    eval_res = evaluate_rag()
    assert eval_res["num_queries"] == 4
    metrics = eval_res["metrics"]
    assert metrics["mean_reciprocal_rank"] > 0.5
    assert metrics["mean_recall@4"] == 1.0


# =====================================================================
# 6. Three Real Mutation Tests
# =====================================================================


def test_mutation_unmasked_attention_fails_causal_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mutation 1: Monkeypatch create_causal_mask to return all zeros (no mask).
    Assert that the causal future independence invariant fails.
    """

    def broken_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    import week07_ar_models_handson.src.transformer as mod_trans

    monkeypatch.setattr(mod_trans, "create_causal_mask", broken_causal_mask)

    mask = mod_trans.create_causal_mask(4)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)

    _, weights = mod_trans.scaled_dot_product_attention(q, k, v, mask=mask)
    w_mat = weights.squeeze(0).detach().numpy()

    # Upper triangular elements should NOT all be zero in unmasked attention
    future_weight = float(w_mat[0, 3])
    assert future_weight > 1e-4, (
        f"Mutation failed: unmasked attention unexpectedly had 0 weight {future_weight}"
    )


def test_mutation_unscaled_attention_fails_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mutation 2: Monkeypatch scaled_dot_product_attention to omit the 1/sqrt(d_k) factor.
    Assert that comparison with independent NumPy oracle fails.
    """
    import week07_ar_models_handson.src.transformer as mod_trans

    original_fn = mod_trans.scaled_dot_product_attention

    def unscaled_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Maliciously force scale = 1.0 instead of 1/sqrt(d_k)
        return original_fn(query, key, value, mask=mask, scale=1.0)

    monkeypatch.setattr(mod_trans, "scaled_dot_product_attention", unscaled_attention)

    q_np = np.random.randn(1, 4, 16).astype(np.float32)
    k_np = np.random.randn(1, 4, 16).astype(np.float32)
    v_np = np.random.randn(1, 4, 16).astype(np.float32)

    pt_out, _ = mod_trans.scaled_dot_product_attention(
        torch.from_numpy(q_np), torch.from_numpy(k_np), torch.from_numpy(v_np)
    )
    oracle_out, _ = numpy_scaled_dot_product_attention(q_np, k_np, v_np)

    # Must mismatch
    assert not np.allclose(pt_out.detach().numpy(), oracle_out, rtol=1e-3, atol=1e-3)


def test_mutation_unsorted_retriever_fails_order_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mutation 3: Monkeypatch BM25Retriever.retrieve to return reversed order.
    Assert that top retrieved document check fails.
    """
    import week07_ar_models_handson.src.rag as mod_rag

    original_retrieve = mod_rag.BM25Retriever.retrieve

    def reversed_retrieve(self: Any, query: str, k: int = 4) -> list[dict[str, Any]]:
        hits = original_retrieve(self, query, k=k)
        return list(reversed(hits))

    monkeypatch.setattr(mod_rag.BM25Retriever, "retrieve", reversed_retrieve)

    corpus = get_default_neurosym_corpus()
    bm25 = mod_rag.BM25Retriever(corpus)
    hits = bm25.retrieve("Grothendieck construction partially-ordered sets", k=4)

    # In ascending score order, the top hit will have lower score than the last hit
    assert hits[0]["score"] <= hits[-1]["score"]


# =====================================================================
# 7. CLI End-to-End Execution in Temporary Directory
# =====================================================================


def test_cli_subcommands_write_expected_artifacts(tmp_path: Path) -> None:
    """Verify that all CLI runner functions execute cleanly and write valid JSON/PNG files."""
    t_res = run_transformer_experiments(results_dir=tmp_path)
    assert (tmp_path / "transformer_results.json").exists()
    assert (tmp_path / "attention_matrix.json").exists()
    assert (tmp_path / "loss_curve.png").exists()
    assert "training" in t_res

    o_res = run_ollama_experiments(results_dir=tmp_path, force_stub=True)
    assert (tmp_path / "ollama_results.json").exists()
    assert o_res["server_status"]["backend"] == "stub"

    r_res = run_rag_experiments(results_dir=tmp_path)
    assert (tmp_path / "rag_results.json").exists()
    assert "evaluation" in r_res
