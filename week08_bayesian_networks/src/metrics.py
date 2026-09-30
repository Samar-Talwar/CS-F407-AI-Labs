# CS F407 Lab, Week 8 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Metrics and comparison utilities for first- and second-order Markov language models."""

from __future__ import annotations

from typing import Any

from week08_bayesian_networks.src.dataset import CANONICAL_CORPUS
from week08_bayesian_networks.src.markov_model import FirstOrderMarkovModel, SecondOrderMarkovModel


def compute_model_statistics(
    model: FirstOrderMarkovModel | SecondOrderMarkovModel,
    num_samples: int = 100,
    sample_seed_start: int = 42,
) -> dict[str, Any]:
    """Compute comprehensive structural and empirical statistics for a Markov model.

    Args:
        model: Trained FirstOrderMarkovModel or SecondOrderMarkovModel.
        num_samples: Number of random sample generations for empirical diversity evaluation.
        sample_seed_start: Base seed for reproducible sample generations.

    Returns:
        Dictionary containing structural, probabilistic, and generation statistics.
    """
    is_first_order = isinstance(model, FirstOrderMarkovModel)
    order = 1 if is_first_order else 2

    vocab = model.get_vocabulary()
    vocab_size = len(vocab)
    observed_contexts = model.get_observed_contexts()
    num_observed_contexts = len(observed_contexts)

    # Theoretical maximum contexts and transitions
    total_possible_contexts = vocab_size**order
    total_possible_transitions = vocab_size ** (order + 1)

    # Count observed non-zero transitions (model parameters)
    total_non_zero_transitions = 0
    branching_factors: list[int] = []

    for ctx in observed_contexts:
        dist = model.get_cpt(ctx)  # type: ignore[arg-type]
        if dist is not None:
            n_transitions = len(dist)
            total_non_zero_transitions += n_transitions
            branching_factors.append(n_transitions)

    avg_branching_factor = (
        sum(branching_factors) / len(branching_factors)
        if branching_factors
        else 0.0
    )
    unobserved_contexts_count = total_possible_contexts - num_observed_contexts
    zero_probability_transitions_count = (
        total_possible_transitions - total_non_zero_transitions
    )

    # Training corpus log-likelihood
    training_log_likelihood = 0.0
    sentence_probs: list[dict[str, Any]] = []
    for s in CANONICAL_CORPUS:
        res = model.sentence_probability(s)
        sentence_probs.append(res.to_dict())
        training_log_likelihood += res.log_probability

    # Generate a large fixed suite of seeded samples to measure empirical diversity
    generated_samples: list[str] = []
    verbatim_count = 0
    cycle_count = 0

    for i in range(num_samples):
        seed = sample_seed_start + i
        gen_res = model.generate(method="sampling", max_length=50, seed=seed)
        generated_samples.append(gen_res.text)
        if gen_res.is_in_training_corpus:
            verbatim_count += 1
        if gen_res.cycle_detected:
            cycle_count += 1

    distinct_generated_sentences = len(set(generated_samples))
    verbatim_fraction = verbatim_count / num_samples if num_samples > 0 else 0.0

    # Also evaluate greedy generation
    greedy_res = model.generate(method="greedy", max_length=50)

    return {
        "model_order": order,
        "vocabulary_size": vocab_size,
        "vocabulary": vocab,
        "num_observed_contexts": num_observed_contexts,
        "total_possible_contexts": total_possible_contexts,
        "unobserved_contexts_count": unobserved_contexts_count,
        "total_non_zero_transitions": total_non_zero_transitions,
        "total_possible_transitions": total_possible_transitions,
        "zero_probability_transitions_count": zero_probability_transitions_count,
        "average_branching_factor": avg_branching_factor,
        "training_log_likelihood": training_log_likelihood,
        "num_empirical_samples": num_samples,
        "distinct_generated_sentences": distinct_generated_sentences,
        "verbatim_copied_samples_count": verbatim_count,
        "verbatim_copied_fraction": verbatim_fraction,
        "sample_cycle_detected_count": cycle_count,
        "greedy_generation": greedy_res.to_dict(),
    }


def compare_models(
    model1: FirstOrderMarkovModel,
    model2: SecondOrderMarkovModel,
    num_samples: int = 100,
) -> dict[str, Any]:
    """Generate side-by-side comparison between first-order and second-order models."""
    stats1 = compute_model_statistics(model1, num_samples=num_samples)
    stats2 = compute_model_statistics(model2, num_samples=num_samples)

    return {
        "first_order": stats1,
        "second_order": stats2,
        "summary": {
            "vocabulary_size": stats1["vocabulary_size"],
            "observed_contexts": {
                "first_order": stats1["num_observed_contexts"],
                "second_order": stats2["num_observed_contexts"],
            },
            "total_non_zero_parameters": {
                "first_order": stats1["total_non_zero_transitions"],
                "second_order": stats2["total_non_zero_transitions"],
            },
            "average_branching_factor": {
                "first_order": stats1["average_branching_factor"],
                "second_order": stats2["average_branching_factor"],
            },
            "training_total_log_likelihood": {
                "first_order": stats1["training_log_likelihood"],
                "second_order": stats2["training_log_likelihood"],
            },
            "distinct_generated_sentences_in_100_samples": {
                "first_order": stats1["distinct_generated_sentences"],
                "second_order": stats2["distinct_generated_sentences"],
            },
            "verbatim_copied_fraction_in_100_samples": {
                "first_order": stats1["verbatim_copied_fraction"],
                "second_order": stats2["verbatim_copied_fraction"],
            },
            "greedy_cycle_detected": {
                "first_order": stats1["greedy_generation"]["cycle_detected"],
                "second_order": stats2["greedy_generation"]["cycle_detected"],
            },
        },
    }
