# CS F407 Lab, Week 7 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Unified CLI for Week 7: Autoregressive Models, Transformers, Ollama & RAG.

Usage:
    python -m week07_ar_models_handson.src.cli --all
    python -m week07_ar_models_handson.src.cli transformer
    python -m week07_ar_models_handson.src.cli ollama
    python -m week07_ar_models_handson.src.cli rag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .ollama_client import OllamaClient, PromptTemplate, run_newton_query
from .rag import (
    BM25Retriever,
    TFIDFRetriever,
    evaluate_rag,
    get_default_neurosym_corpus,
)
from .transformer import (
    DecoderOnlyTransformer,
    MultiHeadAttention,
    PositionalEncoding,
    SimpleTokenizer,
    create_causal_mask,
    generate_text,
    scaled_dot_product_attention,
    train_autoregressive_model,
)


def get_results_dir() -> Path:
    """Resolve results directory relative to this source file."""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_transformer_experiments(results_dir: Path | None = None) -> dict[str, Any]:
    """
    Run PyTorch Transformer experiments, train toy AR model, generate text, and save plots/JSONs.
    """
    if results_dir is None:
        results_dir = get_results_dir()

    print("[*] Running Transformer Experiments...")

    # 1. Scaled Dot-Product Attention & Attention Matrix
    torch.manual_seed(42)
    q = torch.randn(1, 4, 16)
    k = torch.randn(1, 4, 16)
    v = torch.randn(1, 4, 16)
    mask = create_causal_mask(4)
    out_causal, weights_causal = scaled_dot_product_attention(q, k, v, mask=mask)
    out_unmasked, weights_unmasked = scaled_dot_product_attention(q, k, v, mask=None)

    attention_artifact = {
        "description": "Scaled dot-product attention weights (unmasked vs causal masked)",
        "seq_len": 4,
        "d_k": 16,
        "unmasked_weights": weights_unmasked.squeeze(0).tolist(),
        "causal_weights": weights_causal.squeeze(0).tolist(),
        "causal_upper_triangle_is_zero": bool((weights_causal.squeeze(0) == 0.0)[0, 1].item()),
    }
    with open(results_dir / "attention_matrix.json", "w", encoding="utf-8") as f:
        json.dump(attention_artifact, f, indent=2)

    # 2. Positional Encoding Verification
    pe = PositionalEncoding(d_model=32, max_len=64)
    x_zeros = torch.zeros(1, 10, 32)
    pe_out = pe(x_zeros)
    pe_sample = pe_out[0, :5, :8].tolist()

    # 3. Multi-Head Attention (Self vs Cross)
    mha_self = MultiHeadAttention(d_model=32, num_heads=4, is_causal=True)
    x_seq = torch.randn(2, 6, 32)
    self_out, self_attn = mha_self(x_seq)

    mha_cross = MultiHeadAttention(d_model=32, num_heads=4, is_causal=False)
    enc_kv = torch.randn(2, 10, 32)
    cross_out, cross_attn = mha_cross(query=x_seq, key=enc_kv, value=enc_kv)

    # 4. Train Toy Autoregressive Transformer
    toy_corpus = (
        "attention is all you need for autoregressive language modeling. "
        "transformers process sequences in parallel during training and generate tokens one by one."
    )
    tokenizer = SimpleTokenizer.from_text(toy_corpus, mode="char")
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        max_seq_len=256,
    )

    train_res = train_autoregressive_model(
        model, toy_corpus, tokenizer, epochs=120, lr=5e-3, seed=42
    )

    # Generate completions (greedy, temp=0.7, temp=1.2)
    prompt = "attention is"
    gen_greedy = generate_text(model, prompt, tokenizer, max_new_tokens=35, greedy=True, seed=42)
    gen_temp_low = generate_text(
        model, prompt, tokenizer, max_new_tokens=35, temperature=0.7, top_k=5, greedy=False, seed=42
    )
    gen_temp_high = generate_text(
        model,
        prompt,
        tokenizer,
        max_new_tokens=35,
        temperature=1.2,
        top_k=10,
        greedy=False,
        seed=42,
    )

    # Plot loss curve
    loss_history = train_res["loss_history"]
    epochs_list = [h["epoch"] for h in loss_history]
    losses_list = [h["loss"] for h in loss_history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs_list, losses_list, label="Cross-Entropy Loss", color="#1f77b4", linewidth=2.0)
    plt.title("Autoregressive Transformer Training Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curve.png", dpi=200)
    plt.close()

    transformer_results = {
        "architecture": {
            "vocab_size": tokenizer.vocab_size,
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 2,
            "d_ff": 64,
            "max_seq_len": 128,
        },
        "self_attention_output_shape": list(self_out.shape),
        "cross_attention_output_shape": list(cross_out.shape),
        "positional_encoding_sample_5x8": pe_sample,
        "training": {
            "epochs": train_res["epochs"],
            "seed": train_res["seed"],
            "initial_loss": loss_history[0]["loss"],
            "final_loss": train_res["final_loss"],
            "final_perplexity": train_res["final_perplexity"],
        },
        "generations": {
            "prompt": prompt,
            "greedy": gen_greedy,
            "temp_0_7_topk_5": gen_temp_low,
            "temp_1_2_topk_10": gen_temp_high,
        },
    }

    with open(results_dir / "transformer_results.json", "w", encoding="utf-8") as f:
        json.dump(transformer_results, f, indent=2)

    init_loss = loss_history[0]["loss"]
    final_loss = train_res["final_loss"]
    print(f"  [+] Initial Loss: {init_loss:.4f} -> Final Loss: {final_loss:.4f}")
    print(f"  [+] Greedy generation: '{gen_greedy}'")
    return transformer_results


def run_ollama_experiments(
    results_dir: Path | None = None, force_stub: bool = False
) -> dict[str, Any]:
    """
    Run Ollama client tests, prompt chaining, and Sir Isaac Newton query execution.
    """
    if results_dir is None:
        results_dir = get_results_dir()

    print("[*] Running Ollama Integration Experiments...")
    client = OllamaClient(force_stub=force_stub)
    is_healthy = client.health_check()
    models = client.list_models()

    newton_res = run_newton_query(client)

    # Chain execution on another query
    template = PromptTemplate.from_template("Explain how {topic} in one sentence.\nAnswer:")
    chain = template | client
    chain_res = chain.invoke({"topic": "plants create energy"})

    ollama_results = {
        "server_status": {
            "base_url": client.base_url,
            "healthy": is_healthy,
            "backend": "real" if is_healthy and not client.force_stub else "stub",
            "available_models": models,
        },
        "newton_query": newton_res,
        "chain_demo": {
            "template": template.template,
            "input": {"topic": "plants create energy"},
            "answer": chain_res,
        },
    }

    with open(results_dir / "ollama_results.json", "w", encoding="utf-8") as f:
        json.dump(ollama_results, f, indent=2)

    print(f"  [+] Backend: {ollama_results['server_status']['backend']}")
    print(f"  [+] Newton query answer preview: {newton_res['answer'][:100]}...")
    return ollama_results


def run_rag_experiments(results_dir: Path | None = None) -> dict[str, Any]:
    """Run RAG chunking, indexing, retrieval, prompt assembly, benchmark."""
    if results_dir is None:
        results_dir = get_results_dir()

    print("[*] Running RAG Pipeline Experiments...")
    eval_res = evaluate_rag()

    # Compare retrievers on a specific test query
    chunks = get_default_neurosym_corpus()
    bm25 = BM25Retriever(chunks)
    tfidf = TFIDFRetriever(chunks)

    test_q = "symbolic representation molecules posets Grothendieck"
    bm25_hits = bm25.retrieve(test_q, k=3)
    tfidf_hits = tfidf.retrieve(test_q, k=3)

    rag_results = {
        "corpus_statistics": eval_res["corpus_stats"],
        "retriever_comparison": {
            "query": test_q,
            "bm25_top_source": bm25_hits[0]["source"] if bm25_hits else None,
            "bm25_top_score": bm25_hits[0]["score"] if bm25_hits else 0.0,
            "tfidf_top_source": tfidf_hits[0]["source"] if tfidf_hits else None,
            "tfidf_top_score": tfidf_hits[0]["score"] if tfidf_hits else 0.0,
        },
        "evaluation": {
            "metrics": eval_res["metrics"],
            "query_evaluations": eval_res["query_evaluations"],
        },
    }

    with open(results_dir / "rag_results.json", "w", encoding="utf-8") as f:
        json.dump(rag_results, f, indent=2)

    total_chunks = eval_res["corpus_stats"]["total_chunks"]
    total_papers = eval_res["corpus_stats"]["total_papers"]
    print(f"  [+] Total Chunks: {total_chunks} across {total_papers} papers")
    print(f"  [+] Mean Reciprocal Rank (MRR): {eval_res['metrics']['mean_reciprocal_rank']}")
    rec1 = eval_res["metrics"]["mean_recall@1"]
    rec4 = eval_res["metrics"]["mean_recall@4"]
    print(f"  [+] Mean Recall@1: {rec1}, Mean Recall@4: {rec4}")
    return rag_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 7 AR Models, Transformers, Ollama & RAG CLI")
    parser.add_argument(
        "--all", action="store_true", help="Run all experiments and generate all result files"
    )
    parser.add_argument(
        "--force-stub", action="store_true", help="Force deterministic offline stub for Ollama"
    )
    parser.add_argument(
        "subcommand", nargs="?", default="all", choices=["all", "transformer", "ollama", "rag"]
    )

    args = parser.parse_args()
    results_dir = get_results_dir()

    if args.all or args.subcommand == "all":
        run_transformer_experiments(results_dir)
        run_ollama_experiments(results_dir, force_stub=args.force_stub)
        run_rag_experiments(results_dir)
        print(f"\n[+] All experiments completed successfully. Artifacts saved in {results_dir}")
    elif args.subcommand == "transformer":
        run_transformer_experiments(results_dir)
    elif args.subcommand == "ollama":
        run_ollama_experiments(results_dir, force_stub=args.force_stub)
    elif args.subcommand == "rag":
        run_rag_experiments(results_dir)


if __name__ == "__main__":
    main()
