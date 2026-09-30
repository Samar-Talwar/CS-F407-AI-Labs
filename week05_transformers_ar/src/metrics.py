# CS F407 Lab, Week 5 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Metrics, architectural analysis, and comparison functions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from week05_transformers_ar.src.dataset import CANONICAL_CORPUS, BigramModel, CharTokenizer


def analytical_parameter_breakdown(
    vocab_size: int = 25,
    n_embd: int = 48,
    block_size: int = 32,
    n_heads: int = 4,
    n_layers: int = 2,
) -> dict[str, Any]:
    """Compute exact parameter counts by architectural component."""
    head_size = n_embd // n_heads

    # Attention projections per block
    qkv_proj = 3 * n_heads * (n_embd * head_size)
    wo_proj = (n_heads * head_size) * n_embd + n_embd  # weight + bias
    ln1 = 2 * n_embd
    ffn1 = n_embd * (4 * n_embd) + (4 * n_embd)
    ffn2 = (4 * n_embd) * n_embd + n_embd
    ln2 = 2 * n_embd
    per_block_exact = qkv_proj + wo_proj + ln1 + ffn1 + ffn2 + ln2

    # Embeddings
    tok_emb = vocab_size * n_embd
    pos_emb = block_size * n_embd

    # Final LayerNorm
    final_ln = 2 * n_embd

    # LM Head
    lm_head = n_embd * vocab_size + vocab_size

    total = tok_emb + pos_emb + n_layers * per_block_exact + final_ln + lm_head

    return {
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "block_size": block_size,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "head_size": head_size,
        "token_embedding": tok_emb,
        "position_embedding": pos_emb,
        "per_block": per_block_exact,
        "attention_qkv": qkv_proj,
        "attention_wo": wo_proj,
        "layernorm_1": ln1,
        "ffn_1": ffn1,
        "ffn_2": ffn2,
        "layernorm_2": ln2,
        "final_layernorm": final_ln,
        "lm_head": lm_head,
        "total_analytical": total,
    }


def attention_variance_check(
    dims: list[int] | None = None,
    n_samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Empirically verify dot-product variance scales approximately as d (before 1/sqrt(d))."""
    if dims is None:
        dims = [2, 8, 32, 128, 512]
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for d in dims:
        q_samples = rng.standard_normal((n_samples, d))
        k_samples = rng.standard_normal((n_samples, d))
        dot_products = np.sum(q_samples * k_samples, axis=1)
        std = float(np.std(dot_products))
        theoretical = math.sqrt(d)
        results.append({
            "d": d,
            "empirical_std": std,
            "theoretical_sqrt_d": theoretical,
            "ratio_std_sqrt_d": std / theoretical if theoretical > 0 else None,
        })
    return {
        "dimensions": dims,
        "n_samples": n_samples,
        "results": results,
    }


def compare_bigram_vs_transformer(
    text: str = CANONICAL_CORPUS,
    block_size: int = 32,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Compare bigram baseline and TinyGPT on the canonical corpus."""
    bigram = BigramModel(text, smoothing=0.1)
    bigram_loss = bigram.evaluate_loss(text)

    tokenizer = CharTokenizer(text)

    return {
        "bigram_loss": bigram_loss,
        "bigram_vocab_size": bigram.V,
        "transformer_vocab_size": tokenizer.vocab_size,
        "canonical_corpus_length": len(text),
        "canonical_unique_chars": len(tokenizer.chars),
    }


def compare_attention_scaling(
    d_k_values: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare scaled vs unscaled attention variance across head dimensions."""
    if d_k_values is None:
        d_k_values = [4, 8, 16, 32, 64]
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for d_k in d_k_values:
        q = rng.standard_normal((100, d_k))
        k = rng.standard_normal((100, d_k))
        scores = np.sum(q * k, axis=1) / np.sqrt(d_k)
        std_scaled = float(np.std(scores))
        scores_unscaled = np.sum(q * k, axis=1)
        std_unscaled = float(np.std(scores_unscaled))
        results.append({
            "d_k": d_k,
            "std_scaled": std_scaled,
            "std_unscaled": std_unscaled,
            "ratio": std_unscaled / std_scaled if std_scaled > 0 else None,
        })
    return {"d_k_values": d_k_values, "results": results}
