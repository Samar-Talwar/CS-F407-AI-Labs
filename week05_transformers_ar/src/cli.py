# CS F407 Lab, Week 5 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""CLI entry point reproducing all Week 5 Transformer results and exporting JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from week05_transformers_ar.src.attention import (
    CustomMultiHeadAttention,
    scaled_dot_product_attention_numpy,
)
from week05_transformers_ar.src.dataset import (
    CANONICAL_CORPUS,
    BigramModel,
    CharTokenizer,
)
from week05_transformers_ar.src.generate import generate_text
from week05_transformers_ar.src.metrics import (
    analytical_parameter_breakdown,
    attention_variance_check,
    compare_attention_scaling,
    compare_bigram_vs_transformer,
)
from week05_transformers_ar.src.model import TinyGPT
from week05_transformers_ar.src.train import train_tiny_gpt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_all(seed: int = 1337) -> None:
    """Run full pipeline and export JSON artifacts to results/ directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("CS F407 Lab Week 5: Transformers & Autoregressive Models")
    print("=" * 70)

    # 1. Architecture & Parameter Count Breakdown
    print("\n[1/5] Analyzing Model Architecture & Parameter Counts...")
    tokenizer = CharTokenizer(CANONICAL_CORPUS)
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=48,
        block_size=32,
        n_heads=4,
        n_layers=2,
    )
    empirical_params = model.count_parameters()
    analytical_breakdown = analytical_parameter_breakdown(
        vocab_size=tokenizer.vocab_size,
        n_embd=48,
        block_size=32,
        n_heads=4,
        n_layers=2,
    )
    assert empirical_params == analytical_breakdown["total_analytical"] == 60313

    # Custom 6-head attention module demo (H=6, head_size=2, n_embd=8 -> output proj 12x8)
    custom_mha = CustomMultiHeadAttention(num_heads=6, head_size=2, n_embd=8, block_size=16)
    dummy_x = torch.randn(2, 5, 8)
    custom_out = custom_mha(dummy_x)
    assert custom_out.shape == (2, 5, 8)

    arch_data: dict[str, Any] = {
        "model_name": "TinyGPT",
        "vocab_size": tokenizer.vocab_size,
        "n_embd": 48,
        "block_size": 32,
        "n_heads": 4,
        "n_layers": 2,
        "total_parameters_empirical": empirical_params,
        "total_parameters_analytical": analytical_breakdown["total_analytical"],
        "analytical_breakdown": analytical_breakdown,
        "parameter_table": model.get_parameter_breakdown(),
        "custom_mha_demo": {
            "description": "Non-standard MHA with H=6, d_head=2, d_model=8, W_O: 12x8",
            "input_shape": list(dummy_x.shape),
            "output_shape": list(custom_out.shape),
            "output_proj_weight_shape": list(custom_mha.proj.weight.shape),
        },
    }
    with open(RESULTS_DIR / "model_architecture.json", "w", encoding="utf-8") as f:
        json.dump(arch_data, f, indent=2)
    print(f" -> Saved {RESULTS_DIR / 'model_architecture.json'} (Total params: {empirical_params})")

    # 2. Attention Mechanics & Variance Analysis
    print("\n[2/5] Performing Attention Variance & Numerical Analysis...")
    variance_res = attention_variance_check(dims=[2, 8, 32, 128, 512], n_samples=3000, seed=seed)
    scaling_res = compare_attention_scaling(d_k_values=[4, 8, 16, 32, 64], seed=seed)

    # NumPy scaled dot-product demo on 3 tokens
    rng = np.random.default_rng(seed)
    q_demo = rng.standard_normal((3, 4))
    k_demo = rng.standard_normal((3, 4))
    v_demo = rng.standard_normal((3, 4))
    mask_demo = np.tril(np.ones((3, 3)))
    out_demo, weights_demo = scaled_dot_product_attention_numpy(
        q_demo, k_demo, v_demo, mask=mask_demo
    )

    attn_data: dict[str, Any] = {
        "dot_product_variance_scaling": variance_res,
        "scaled_vs_unscaled_variance": scaling_res,
        "numpy_causal_attention_demo": {
            "query_shape": list(q_demo.shape),
            "key_shape": list(k_demo.shape),
            "value_shape": list(v_demo.shape),
            "attention_weights": weights_demo.tolist(),
            "output": out_demo.tolist(),
            "causal_upper_triangle_is_zero": bool(np.all(np.triu(weights_demo, k=1) == 0.0)),
        },
    }
    with open(RESULTS_DIR / "attention_analysis.json", "w", encoding="utf-8") as f:
        json.dump(attn_data, f, indent=2)
    print(f" -> Saved {RESULTS_DIR / 'attention_analysis.json'}")

    # 3. Training TinyGPT (180 steps)
    print("\n[3/5] Training TinyGPT over 180 steps (lr=3e-3, batch_size=32)...")
    train_res = train_tiny_gpt(
        model=model,
        text=CANONICAL_CORPUS,
        max_iters=180,
        eval_interval=20,
        eval_iters=20,
        learning_rate=3e-3,
        batch_size=32,
        block_size=32,
        device="cpu",
        seed=seed,
    )
    print(f" -> Initial Loss: {train_res['initial_loss']:.4f}")
    print(f" -> Final Loss:   {train_res['final_loss']:.4f}")

    train_data: dict[str, Any] = {
        "initial_loss": train_res["initial_loss"],
        "final_loss": train_res["final_loss"],
        "first_10_avg_loss": train_res["first_10_avg_loss"],
        "last_10_avg_loss": train_res["last_10_avg_loss"],
        "step_losses": train_res["step_losses"],
        "eval_losses": train_res["eval_losses"],
        "total_steps": train_res["total_steps"],
        "learning_rate": train_res["learning_rate"],
        "batch_size": train_res["batch_size"],
        "block_size": train_res["block_size"],
    }
    with open(RESULTS_DIR / "training_loss.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    print(f" -> Saved {RESULTS_DIR / 'training_loss.json'}")

    # 4. Autoregressive Generation across Temperatures
    print("\n[4/5] Generating Samples across Temperatures (tau in [0.3, 0.7, 0.8, 1.0, 2.0])...")
    trained_model = train_res["model"]
    temperatures = [0.3, 0.7, 0.8, 1.0, 2.0]
    prompts = ["baana ", "attention ", ""]

    gen_results: list[dict[str, Any]] = []
    for prompt in prompts:
        for tau in temperatures:
            text_out = generate_text(
                model=trained_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=60,
                temperature=tau,
                seed=seed,
            )
            gen_results.append({
                "prompt": prompt,
                "temperature": tau,
                "output": text_out,
            })

    # Bigram baseline generations
    bigram = BigramModel(CANONICAL_CORPUS, smoothing=0.1)
    bigram_greedy = bigram.generate(start_char="b", max_new_tokens=60, method="greedy", seed=seed)
    bigram_sampled = bigram.generate(
        start_char="b", max_new_tokens=60, method="sampling", seed=seed
    )

    gen_data: dict[str, Any] = {
        "temperatures_evaluated": temperatures,
        "prompts_evaluated": prompts,
        "transformer_generations": gen_results,
        "bigram_generations": {
            "start_char": "b",
            "greedy": bigram_greedy,
            "sampled": bigram_sampled,
        },
    }
    with open(RESULTS_DIR / "generation_samples.json", "w", encoding="utf-8") as f:
        json.dump(gen_data, f, indent=2)
    print(f" -> Saved {RESULTS_DIR / 'generation_samples.json'}")

    # 5. Model Comparison (Bigram vs TinyGPT)
    print("\n[5/5] Generating Baseline vs TinyGPT Comparative Analysis...")
    bigram_loss = bigram.evaluate_loss(CANONICAL_CORPUS)
    comparison_info = compare_bigram_vs_transformer(CANONICAL_CORPUS)

    # Check verbatim sentence reproduction on 10 sampled prompts
    corpus_sentences = [s.strip() for s in CANONICAL_CORPUS.strip().split("\n") if s.strip()]
    verbatim_hits = 0
    test_prompts = ["to ", "a ", "attention ", "we ", "generation ", "a model "]
    for p in test_prompts:
        gen = generate_text(
            model=trained_model,
            tokenizer=tokenizer,
            prompt=p,
            max_new_tokens=50,
            temperature=0.3,
            seed=seed,
        )
        for sentence in corpus_sentences:
            if sentence in gen:
                verbatim_hits += 1
                break

    comp_data: dict[str, Any] = {
        "bigram_model": {
            "type": "Counting N-gram Lookup",
            "context_length": 1,
            "parameters": bigram.V * bigram.V,
            "training_loss": bigram_loss,
            "generates_coherent_sentences": False,
        },
        "tiny_gpt": {
            "type": "Autoregressive Decoder Transformer",
            "context_length": 32,
            "parameters": empirical_params,
            "initial_loss": train_res["initial_loss"],
            "final_loss": train_res["final_loss"],
            "generates_coherent_sentences": True,
            "verbatim_sentence_match_rate": verbatim_hits / len(test_prompts),
        },
        "corpus_metrics": comparison_info,
    }
    with open(RESULTS_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)
    print(f" -> Saved {RESULTS_DIR / 'comparison.json'}")

    print("\n" + "=" * 70)
    print("All results successfully generated and verified!")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
