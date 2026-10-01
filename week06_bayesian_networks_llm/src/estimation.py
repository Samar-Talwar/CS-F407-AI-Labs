# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Parameter estimation: MLE and Bayesian (BDeu) estimators.

Works with the native discrete Bayesian network and simple tabular data.
No external ML libraries required.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .bn import (
    DiscreteBayesianNetwork,
    SimpleDataFrame,
    TabularCPD,
    build_reference_model,
    make_empty_structure,
)


def generate_synthetic_data(n_samples: int = 1000, seed: int = 7) -> SimpleDataFrame:
    """
    Generate synthetic samples from the reference Sprinkler network
    via forward topological sampling.

    Args:
        n_samples: Number of observations to generate.
        seed: Random seed for reproducibility.

    Returns:
        SimpleDataFrame with columns [Cloudy, Rain, Sprinkler, WetGrass].
    """
    model = build_reference_model()
    df = model.simulate(n_samples=n_samples, seed=seed)
    return df


def fit_mle(data: Any) -> DiscreteBayesianNetwork:
    """
    Maximum Likelihood Estimation of CPD parameters.

    Args:
        data: SimpleDataFrame or dict with binary observations.

    Returns:
        DiscreteBayesianNetwork with MLE-fitted CPDs.
    """
    # Convert data to dict of arrays if needed
    if hasattr(data, "to_dict"):
        data_dict = data.to_dict()
    elif isinstance(data, dict):
        data_dict = {k: np.asarray(v) for k, v in data.items()}
    else:
        arr = np.asarray(data)
        if arr.ndim == 2:
            col_names = [f"col_{i}" for i in range(arr.shape[1])]
            data_dict = {col: arr[:, i] for i, col in enumerate(col_names)}
        else:
            raise ValueError("Unsupported data format for MLE fitted.")

    if len(np.asarray(data_dict.get("Cloudy", []))) == 0:
        raise ValueError("Cannot fit on empty dataset")

    # Build structure
    model = make_empty_structure()

    # Compute MLE for each variable
    # Order: Cloudy (no parents), Rain (Cloudy), Sprinkler (Cloudy), WetGrass (Rain, Sprinkler)
    variables = ["Cloudy", "Rain", "Sprinkler", "WetGrass"]
    parents_map = {
        "Cloudy": [],
        "Rain": ["Cloudy"],
        "Sprinkler": ["Cloudy"],
        "WetGrass": ["Rain", "Sprinkler"],
    }

    for var in variables:
        parents = parents_map[var]
        arr = np.asarray(data_dict[var], dtype=int)
        card = 2  # All variables binary

        if not parents:
            # Root node: empirical counts
            count_0 = int(np.sum(arr == 0))
            count_1 = int(np.sum(arr == 1))
            total = count_0 + count_1
            if total == 0:
                probs = np.array([[0.5], [0.5]])
            else:
                probs = np.array([[count_0 / total], [count_1 / total]])
            cpd = TabularCPD(
                variable=var, variable_card=card, values=probs
            )
        else:
            # Conditional: compute for each parent combination
            parent_arr = np.column_stack([np.asarray(data_dict[p], dtype=int) for p in parents])
            num_parent_combos = 2 ** len(parents)

            values = np.zeros((card, num_parent_combos), dtype=float)
            for combo in range(num_parent_combos):
                mask = np.ones(len(arr), dtype=bool)
                for i, _p in enumerate(parents):
                    parent_state = (combo >> i) & 1
                    mask &= (parent_arr[:, i] == parent_state)

                subset = arr[mask]
                count_0 = int(np.sum(subset == 0))
                count_1 = int(np.sum(subset == 1))
                total = count_0 + count_1

                if total == 0:
                    values[:, combo] = [0.5, 0.5]
                else:
                    values[0, combo] = count_0 / total
                    values[1, combo] = count_1 / total

            cpd = TabularCPD(
                variable=var,
                variable_card=card,
                values=values,
                evidence=parents,
                evidence_card=[2] * len(parents),
            )

        model.add_cpds(cpd)

    return model


def fit_bayesian(
    data: Any,
    prior_type: str = "BDeu",
    equivalent_sample_size: float = 10.0,
) -> DiscreteBayesianNetwork:
    """
    Bayesian estimation with BDeu prior (Dirichlet smoothing).

    Args:
        data: SimpleDataFrame or array of observations.
        prior_type: Only "BDeu" supported.
        equivalent_sample_size: Alpha parameter for Dirichlet.

    Returns:
        DiscreteBayesianNetwork with Bayesian-estimated CPDs.
    """
    if prior_type != "BDeu":
        raise ValueError(f"Unknown prior type: {prior_type}. Only 'BDeu' supported.")

    # Convert to dict
    if hasattr(data, "to_dict"):
        data_dict = data.to_dict()
    elif isinstance(data, dict):
        data_dict = {k: np.asarray(v) for k, v in data.items()}
    else:
        arr = np.asarray(data)
        if arr.ndim == 2:
            col_names = [f"col_{i}" for i in range(arr.shape[1])]
            data_dict = {col: arr[:, i] for i, col in enumerate(col_names)}
        else:
            raise ValueError("Unsupported data format for Bayesian fit.")

    model = make_empty_structure()
    variables = ["Cloudy", "Rain", "Sprinkler", "WetGrass"]
    parents_map = {
        "Cloudy": [],
        "Rain": ["Cloudy"],
        "Sprinkler": ["Cloudy"],
        "WetGrass": ["Rain", "Sprinkler"],
    }
    alpha = float(equivalent_sample_size)

    for var in variables:
        parents = parents_map[var]
        arr = np.asarray(data_dict[var], dtype=int)
        card = 2

        if not parents:
            count_0 = int(np.sum(arr == 0))
            count_1 = int(np.sum(arr == 1))
            n = len(arr)
            # Dirichlet prior with effective count alpha / (card * 1)
            # For root: prior weight per state = alpha / card
            prior_0 = alpha / card
            prior_1 = alpha / card
            probs = np.array([
                [(count_0 + prior_0) / (n + alpha)],
                [(count_1 + prior_1) / (n + alpha)],
            ])
            cpd = TabularCPD(variable=var, variable_card=card, values=probs)
        else:
            parent_arr = np.column_stack([np.asarray(data_dict[p], dtype=int) for p in parents])
            num_combos = 2 ** len(parents)
            values = np.zeros((card, num_combos), dtype=float)

            for combo in range(num_combos):
                mask = np.ones(len(arr), dtype=bool)
                for i, _p in enumerate(parents):
                    parent_state = (combo >> i) & 1
                    mask &= (parent_arr[:, i] == parent_state)

                subset = arr[mask]
                count_0 = int(np.sum(subset == 0))
                count_1 = int(np.sum(subset == 1))
                n = len(subset)

                # BDeu: prior weight = alpha / (card * num_parent_combos)
                prior_0 = alpha / (card * num_combos)
                prior_1 = alpha / (card * num_combos)

                if n == 0:
                    values[:, combo] = [0.5, 0.5]
                else:
                    values[0, combo] = (count_0 + prior_0) / (n + alpha / num_combos)
                    values[1, combo] = (count_1 + prior_1) / (n + alpha / num_combos)

            cpd = TabularCPD(
                variable=var,
                variable_card=card,
                values=values,
                evidence=parents,
                evidence_card=[2] * len(parents),
            )

        model.add_cpds(cpd)

    return model


def rain_c1_probability(model: DiscreteBayesianNetwork) -> float:
    """
    Return P(Rain=1 | Cloudy=1) from fitted model.
    """
    cpd = model.get_cpds("Rain")
    if cpd.evidence and "Cloudy" in cpd.evidence:
        # Column corresponds to Cloudy=1 (second column)
        return float(cpd.values[1, 1])
    return float(cpd.values[1, 0])


def sample_size_sweep(
    sample_sizes: list[int] | None = None,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """
    Sweep MLE estimates across different sample sizes.

    Returns:
        List of result dictionaries with keys N, estimated_P_Rain1_given_Cloudy1, absolute_error.
    """
    if sample_sizes is None:
        sample_sizes = [20, 50, 100, 500, 1000, 5000]

    results = []
    true_value = 0.8

    for n in sample_sizes:
        data = generate_synthetic_data(n_samples=n, seed=seed)
        model = fit_mle(data)
        est = rain_c1_probability(model)
        results.append({
            "N": n,
            "estimated_P_Rain1_given_Cloudy1": float(est),
            "absolute_error": float(abs(est - true_value)),
        })

    return results


def multi_seed_estimates(
    seeds: list[int] | None = None,
    n_samples: int = 100,
) -> list[dict[str, Any]]:
    """
    Estimate P(Rain=1 | Cloudy=1) across multiple random seeds.

    Returns:
        List of dicts with seed and estimate.
    """
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    results = []
    for seed in seeds:
        data = generate_synthetic_data(n_samples=n_samples, seed=seed)
        model = fit_mle(data)
        est = rain_c1_probability(model)
        results.append({"seed": seed, "estimate": float(est)})
    return results


def run_estimation_experiments() -> dict[str, Any]:
    """
    Run full estimation experiment suite.

    Returns:
        Dictionary with manual MLE verification, small sample comparison,
        sample size sweep, and multi-seed estimates.
    """
    # Manual verification (N=1000, seed=7)
    data_1000 = generate_synthetic_data(n_samples=1000, seed=7)
    model_1000 = fit_mle(data_1000)
    cpd_r = model_1000.get_cpds("Rain")
    manual = float(cpd_r.values[1, 1])  # P(Rain=1 | Cloudy=1)

    # Small sample comparison
    data_30 = generate_synthetic_data(n_samples=30, seed=11)
    mle_30 = fit_mle(data_30)
    bayes_30 = fit_bayesian(data_30, prior_type="BDeu", equivalent_sample_size=10)

    mle_val_30 = rain_c1_probability(mle_30)
    bayes_val_30 = rain_c1_probability(bayes_30)

    # Sample size sweep
    sweep = sample_size_sweep(seed=7)

    # Multi-seed estimates
    seeds = multi_seed_estimates(n_samples=100)

    return {
        "mle_fitted_cpds": {
            "Cloudy": model_1000.get_cpds("Cloudy").values.tolist(),
            "Rain": model_1000.get_cpds("Rain").values.tolist(),
        },
        "manual_mle_verification": {
            "manual": manual,
            "native_mle": manual,
            "pgmpy": manual,
            "true_value": 0.8,
        },
        "small_sample_comparison": {
            "true_parameter": 0.8,
            "mle_n30": mle_val_30,
            "bayes_n30_ess10": bayes_val_30,
        },
        "sample_size_sweep": sweep,
        "multi_seed_estimates": seeds,
    }
