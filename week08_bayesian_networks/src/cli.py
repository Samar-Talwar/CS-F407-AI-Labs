# CS F407 Lab, Week 8 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Command-line interface to train models, evaluate statistics, and generate all result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from week08_bayesian_networks.src.dataset import CANONICAL_CORPUS
from week08_bayesian_networks.src.markov_model import (
    FirstOrderMarkovModel,
    SecondOrderMarkovModel,
)
from week08_bayesian_networks.src.metrics import compare_models


def get_results_dir() -> Path:
    """Return the absolute path to the results directory."""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def serialize_second_order_dict(
    d: dict[tuple[str, str], Any],
) -> dict[str, Any]:
    """Convert tuple keys (w1, w2) to formatted string keys 'w1 w2' for JSON."""
    return {f"{k[0]} {k[1]}": v for k, v in d.items()}


def run_pipeline() -> dict[str, Any]:
    """Execute complete training, evaluation, generation, and export pipeline."""
    results_dir = get_results_dir()

    # 1. Train First-Order Model
    m1 = FirstOrderMarkovModel()
    m1.train(CANONICAL_CORPUS)

    # 2. Train Second-Order Model
    m2 = SecondOrderMarkovModel()
    m2.train(CANONICAL_CORPUS)

    # 3. Export Counts
    counts_m1_path = results_dir / "counts_first_order.json"
    counts_m1_data = {k: dict(v) for k, v in m1.counts.items()}
    with open(counts_m1_path, "w", encoding="utf-8") as f:
        json.dump(counts_m1_data, f, indent=2)

    counts_m2_path = results_dir / "counts_second_order.json"
    counts_m2_data = {
        f"{k[0]} {k[1]}": dict(v) for k, v in m2.counts.items()
    }
    with open(counts_m2_path, "w", encoding="utf-8") as f:
        json.dump(counts_m2_data, f, indent=2)

    # 4. Export CPTs
    cpt_m1_path = results_dir / "cpt_first_order.json"
    with open(cpt_m1_path, "w", encoding="utf-8") as f:
        json.dump(m1.cpt, f, indent=2)

    cpt_m2_path = results_dir / "cpt_second_order.json"
    cpt_m2_data = serialize_second_order_dict(m2.cpt)
    with open(cpt_m2_path, "w", encoding="utf-8") as f:
        json.dump(cpt_m2_data, f, indent=2)

    # 5. Normalization Checks
    norm_m1 = m1.check_normalization()
    norm_m2 = m2.check_normalization()
    norm_data = {
        "first_order": norm_m1,
        "second_order": norm_m2,
    }
    norm_path = results_dir / "normalisation_checks.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm_data, f, indent=2)

    # 6. Chain-Rule Sentence Probabilities
    chain_rule_data: list[dict[str, Any]] = []
    for idx, sentence in enumerate(CANONICAL_CORPUS, start=1):
        res1 = m1.sentence_probability(sentence)
        res2 = m2.sentence_probability(sentence)
        chain_rule_data.append({
            "sentence_id": idx,
            "sentence": sentence,
            "first_order": res1.to_dict(),
            "second_order": res2.to_dict(),
        })

    chain_rule_path = results_dir / "chain_rule_probabilities.json"
    with open(chain_rule_path, "w", encoding="utf-8") as f:
        json.dump(chain_rule_data, f, indent=2)

    # 7. Generation Experiments (Greedy + 10 Seeded Samples per model)
    seeds = [0, 1, 2, 3, 4, 42, 100, 2024, 2026, 9999]
    gen_m1_greedy = m1.generate(method="greedy", max_length=30)
    gen_m1_samples = [
        m1.generate(method="sampling", max_length=30, seed=s).to_dict()
        for s in seeds
    ]

    gen_m2_greedy = m2.generate(method="greedy", max_length=30)
    gen_m2_samples = [
        m2.generate(method="sampling", max_length=30, seed=s).to_dict()
        for s in seeds
    ]

    generation_data = {
        "seeds": seeds,
        "first_order": {
            "greedy": gen_m1_greedy.to_dict(),
            "samples": gen_m1_samples,
        },
        "second_order": {
            "greedy": gen_m2_greedy.to_dict(),
            "samples": gen_m2_samples,
        },
    }
    gen_path = results_dir / "generation.json"
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(generation_data, f, indent=2)

    # 8. Model Comparison
    comparison_data = compare_models(m1, m2, num_samples=100)
    comp_path = results_dir / "comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    return {
        "m1": m1,
        "m2": m2,
        "norm_data": norm_data,
        "comparison_data": comparison_data,
        "generation_data": generation_data,
        "chain_rule_data": chain_rule_data,
    }


def main() -> None:
    """CLI entry point: run pipeline and print structured report."""
    print("=" * 70)
    print(" CS F407 Week 8: Bayesian Networks & Autoregressive Models")
    print("=" * 70)

    results = run_pipeline()
    m1: FirstOrderMarkovModel = results["m1"]
    m2: SecondOrderMarkovModel = results["m2"]
    norm = results["norm_data"]
    comp = results["comparison_data"]

    print("\n[1] DATASET & VOCABULARY")
    print(f"  Training sentences count: {len(CANONICAL_CORPUS)}")
    print(f"  Vocabulary size:          {len(m1.get_vocabulary())}")
    print(f"  Vocabulary tokens:        {m1.get_vocabulary()}")

    print("\n[2] FIRST-ORDER MODEL CPT (Selected Contexts)")
    for ctx in ["<START>", "the", "cat", "dog", "sat", "ran", "mat"]:
        cpt = m1.get_cpt(ctx)
        print(f"  P(w_t | w_{{t-1}}='{ctx}') -> {cpt}")

    print("\n[3] SECOND-ORDER MODEL CPT (Selected Contexts)")
    for ctx in [
        ("<START>", "<START>"),
        ("<START>", "the"),
        ("the", "cat"),
        ("sat", "on"),
        ("on", "the"),
        ("to", "the"),
    ]:
        cpt = m2.get_cpt(ctx)
        print(f"  P(w_t | w_{{t-2}}='{ctx[0]}', w_{{t-1}}='{ctx[1]}') -> {cpt}")

    print("\n[4] NORMALISATION CHECKS")
    print(f"  First-Order Max Deviation:  {norm['first_order']['max_deviation']:.2e}")
    print(f"  Second-Order Max Deviation: {norm['second_order']['max_deviation']:.2e}")

    print("\n[5] GENERATION SAMPLES")
    print("  First-Order Greedy:")
    g1 = results["generation_data"]["first_order"]["greedy"]
    print(f"    Text: {g1['text']!r} | Cycle: {g1['cycle_detected']} | Tokens: {g1['tokens']}")
    print("  First-Order Sample (seed=42):")
    s1 = results["generation_data"]["first_order"]["samples"][5]
    print(f"    Text: {s1['text']!r} | Verbatim: {s1['is_in_training_corpus']}")

    print("  Second-Order Greedy:")
    g2 = results["generation_data"]["second_order"]["greedy"]
    print(f"    Text: {g2['text']!r} | Cycle: {g2['cycle_detected']} | Tokens: {g2['tokens']}")
    print("  Second-Order Sample (seed=42):")
    s2 = results["generation_data"]["second_order"]["samples"][5]
    print(f"    Text: {s2['text']!r} | Verbatim: {s2['is_in_training_corpus']}")

    print("\n[6] MODEL COMPARISON SUMMARY")
    sum_data = comp["summary"]
    oc_f = sum_data['observed_contexts']['first_order']
    oc_s = sum_data['observed_contexts']['second_order']
    print(f"  Observed Contexts:    First-Order={oc_f} | Second-Order={oc_s}")

    nz_f = sum_data['total_non_zero_parameters']['first_order']
    nz_s = sum_data['total_non_zero_parameters']['second_order']
    print(f"  Non-Zero Parameters:  First-Order={nz_f} | Second-Order={nz_s}")
    bf_f = sum_data['average_branching_factor']['first_order']
    bf_s = sum_data['average_branching_factor']['second_order']
    print(f"  Avg Branching Factor: First-Order={bf_f:.3f} | Second-Order={bf_s:.3f}")

    ll_f = sum_data['training_total_log_likelihood']['first_order']
    ll_s = sum_data['training_total_log_likelihood']['second_order']
    print(f"  Training Log-Likelihood: First-Order={ll_f:.4f} | Second-Order={ll_s:.4f}")

    dist_f = sum_data['distinct_generated_sentences_in_100_samples']['first_order']
    dist_s = sum_data['distinct_generated_sentences_in_100_samples']['second_order']
    print(f"  Distinct in 100 Runs: First-Order={dist_f} | Second-Order={dist_s}")

    verb_f = sum_data['verbatim_copied_fraction_in_100_samples']['first_order']
    verb_s = sum_data['verbatim_copied_fraction_in_100_samples']['second_order']
    print(f"  Verbatim Fraction:    First-Order={verb_f:.2%} | Second-Order={verb_s:.2%}")
    print("\nAll result files exported to week08_bayesian_networks/results/")


if __name__ == "__main__":
    main()
