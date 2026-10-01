# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Command-line interface for Week 6: Bayesian Networks and LLM Integration.

This CLI regenerates all results from the notebook and writes them to results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from .bn import (
    DiscreteBayesianNetwork,
    TabularCPD,
    build_reference_model,
    validate_cpt,
)
from .estimation import (
    fit_mle,
    generate_synthetic_data,
    run_estimation_experiments,
)
from .inference import (
    VariableElimination,
    compute_posterior_by_enumeration,
    query_posterior,
    run_inference_exercises,
)
from .llm_interface import (
    LLMInterface,
    RealLLMBackend,
    StubLLMBackend,
)


def ensure_results_dir() -> Path:
    """Ensure results directory exists."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def save_json(data: dict, filename: str) -> Path:
    """Save data as JSON to results directory."""
    results_dir = ensure_results_dir()
    filepath = results_dir / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    return filepath


def save_plot(fig: plt.Figure, filename: str) -> Path:
    """Save matplotlib figure to results directory."""
    results_dir = ensure_results_dir()
    filepath = results_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def run_reference_model() -> dict:
    """Build and validate reference model, save CPDs."""
    model = build_reference_model()

    result = {
        "model_valid": model.check_model(),
        "edges": list(model.edges()),
        "nodes": list(model.nodes()),
        "cpd_validation": validate_cpt(model),
        "cpds": {},
    }

    for cpd in model.get_cpds():
        result["cpds"][cpd.variable] = {
            "values": cpd.values.tolist(),
            "evidence": cpd.evidence,
            "evidence_card": cpd.evidence_card,
        }

    save_json(result, "reference_model.json")
    return result


def run_inference() -> dict:
    """Run all inference tasks and save results."""
    model = build_reference_model()

    # Main query
    rain_posterior = query_posterior(model, ["Rain"], {"WetGrass": 1})
    enum_rain = compute_posterior_by_enumeration(model, "Rain", 1, {"WetGrass": 1})

    # Exercises
    exercises = run_inference_exercises()

    result = {
        "rain_given_wetgrass": {
            "native": float(rain_posterior[1]),
            "enumeration": enum_rain,
            "difference": abs(float(rain_posterior[1]) - enum_rain),
        },
        "exercises": exercises,
    }

    save_json(result, "inference_posteriors.json")
    return result


def run_estimation() -> dict:
    """Run all estimation tasks and save results."""
    results = run_estimation_experiments()

    # Convert numpy types to native Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)
    save_json(results, "estimation_results.json")

    # Also save MLE fitted CPDs separately for easy access
    data_1000 = generate_synthetic_data(n_samples=1000, seed=7)
    mle_model_1000 = fit_mle(data_1000)
    cpd_data = {cpd.variable: cpd.values.tolist() for cpd in mle_model_1000.get_cpds()}
    save_json(cpd_data, "mle_fitted_cpds_N1000.json")

    # Sample size sweep plot
    fig, ax = plt.subplots(figsize=(7, 4))
    sweep = results["sample_size_sweep"]
    ns = [r["N"] for r in sweep]
    estimates = [r["estimated_P_Rain1_given_Cloudy1"] for r in sweep]
    ax.plot(ns, estimates, marker="o", label="MLE")
    ax.axhline(0.8, linestyle="--", label="true value")
    ax.set_xscale("log")
    ax.set_xlabel("number of observations")
    ax.set_ylabel("P(Rain=1 | Cloudy=1)")
    ax.set_title("Parameter estimate versus sample size")
    ax.legend()
    save_plot(fig, "mle_sweep_plot.png")

    return results


def run_llm_tasks(use_real_llm: bool = False) -> dict:
    """Run LLM code generation tasks."""
    backend = RealLLMBackend() if use_real_llm else StubLLMBackend()
    interface = LLMInterface(backend=backend)
    backend_name = interface.backend.name

    # Generate synthetic data for estimation tasks
    data_1000 = generate_synthetic_data(n_samples=1000, seed=7)
    data_small = generate_synthetic_data(n_samples=30, seed=11)

    results = {"backend": backend_name}

    # Inference task
    inference_result = interface.run_inference_task(approve=True)
    results["inference"] = inference_result
    save_json(inference_result, f"llm_generated_inference_{backend_name.replace(':', '_')}.json")

    # Estimation task (MLE)
    estimation_result = interface.run_estimation_task(data_1000, approve=True)
    results["estimation"] = estimation_result
    save_json(estimation_result, f"llm_generated_estimation_{backend_name.replace(':', '_')}.json")

    # Bayesian estimation task
    bayesian_result = interface.run_bayesian_task(data_small, approve=True)
    results["bayesian"] = bayesian_result
    save_json(bayesian_result, f"llm_generated_bayesian_{backend_name.replace(':', '_')}.json")

    return results


def run_validation_tests() -> dict:
    """Run validation and error handling demonstrations."""
    results = {}

    # Broken CPT test
    broken_model = DiscreteBayesianNetwork([("Cloudy", "Rain")])
    cpd_cloudy_ok = TabularCPD("Cloudy", 2, [[0.5], [0.5]])
    cpd_rain_broken = TabularCPD(
        "Rain", 2,
        [[0.9, 0.2], [0.3, 0.8]],
        evidence=["Cloudy"], evidence_card=[2]
    )
    broken_model.add_cpds(cpd_cloudy_ok, cpd_rain_broken)

    try:
        broken_model.check_model()
        results["broken_cpt"] = {"caught": False}
    except ValueError as e:
        results["broken_cpt"] = {"caught": True, "error": str(e)}

    # Semantic error test (permuted WetGrass CPD)
    ref_model = build_reference_model()
    correct_inference = VariableElimination(ref_model)
    correct_posterior = correct_inference.query(
        ["Rain"], evidence={"WetGrass": 1}, show_progress=False
    )

    wrong_model = build_reference_model()
    wrong_model.remove_cpds(wrong_model.get_cpds("WetGrass"))

    wrong_wetgrass = TabularCPD(
        "WetGrass", 2,
        [[0.99, 0.10, 0.01, 0.10], [0.01, 0.90, 0.99, 0.90]],
        evidence=["Rain", "Sprinkler"], evidence_card=[2, 2]
    )
    wrong_model.add_cpds(wrong_wetgrass)

    wrong_inference = VariableElimination(wrong_model)
    wrong_posterior = wrong_inference.query(
        ["Rain"], evidence={"WetGrass": 1}, show_progress=False
    )

    results["semantic_error"] = {
        "check_model_passes": wrong_model.check_model(),
        "correct_posterior": float(correct_posterior.values[1]),
        "wrong_posterior": float(wrong_posterior.values[1]),
        "difference": abs(float(correct_posterior.values[1]) - float(wrong_posterior.values[1])),
    }

    save_json(results, "validation_tests.json")
    return results


def run_original_notebook_outputs() -> dict:
    """Save original notebook outputs for reference."""
    outputs = {
        "note": (
            "These outputs are copied from the professor's original notebook (llm_bn.ipynb). "
            "They are kept separate from reproduced results."
        ),
        "posterior_rain_given_wetgrass": 0.7047692307692307,
        "enumeration_posterior": 0.7047692307692309,
        "difference": 2.220446049250313e-16,
        "mle_n1000_rain_given_cloudy1": 0.7725409836065574,
        "manual_mle_n1000": 0.7725409836065574,
        "true_rain_given_cloudy1": 0.8,
        "sample_size_sweep": [
            {"N": 20, "estimate": 0.8888888888888888, "error": 0.08888888888888882},
            {"N": 50, "estimate": 0.8260869565217391, "error": 0.02608695652173914},
            {"N": 100, "estimate": 0.7083333333333334, "error": 0.09166666666666661},
            {"N": 500, "estimate": 0.7804878048780488, "error": 0.019512195121951245},
            {"N": 1000, "estimate": 0.7725409836065574, "error": 0.02745901639344256},
            {"N": 5000, "estimate": 0.8039607843137255, "error": 0.00396078431372549},
        ],
        "small_sample_comparison": {
            "true": 0.8,
            "mle_n30": 0.9090909090909091,
            "bayes_n30_ess10": 0.78125,
        },
        "exercise_posteriors": {
            "sprinkler_given_wetgrass": 0.4278461538461539,
            "cloudy_given_wetgrass": 0.5746153846153845,
            "rain_given_wetgrass_sprinkler0": 0.9922022048937886,
        },
        "multi_seed_estimates": [
            {"seed": 1, "estimate": 0.8571428571428571},
            {"seed": 2, "estimate": 0.875},
            {"seed": 3, "estimate": 0.8},
            {"seed": 4, "estimate": 0.6666666666666666},
            {"seed": 5, "estimate": 0.8333333333333334},
        ],
    }

    save_json(outputs, "original_notebook_outputs.json")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Week 6: Bayesian Networks and LLM Integration CLI"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all tasks and generate all results"
    )
    parser.add_argument(
        "--reference", action="store_true", help="Run reference model construction"
    )
    parser.add_argument(
        "--inference", action="store_true", help="Run inference tasks"
    )
    parser.add_argument(
        "--estimation", action="store_true", help="Run parameter estimation tasks"
    )
    parser.add_argument(
        "--llm", action="store_true", help="Run LLM code generation tasks"
    )
    parser.add_argument(
        "--validation", action="store_true", help="Run validation tests"
    )
    parser.add_argument(
        "--original-outputs", action="store_true", help="Save original notebook outputs"
    )
    parser.add_argument(
        "--use-real-llm",
        action="store_true",
        help="Use real LLM backend (requires transformers, model download)",
    )

    args = parser.parse_args()

    # Default to --all if no specific task selected
    task_flags = [
        args.all,
        args.reference,
        args.inference,
        args.estimation,
        args.llm,
        args.validation,
        args.original_outputs,
    ]
    if not any(task_flags):
        args.all = True

    print("=" * 60)
    print("CS F407 Week 6: Bayesian Networks and LLM Integration")
    print("=" * 60)

    if args.all or args.reference:
        print("\n[1/7] Building reference model...")
        run_reference_model()
        print("  -> results/reference_model.json")

    if args.all or args.inference:
        print("\n[2/7] Running inference tasks...")
        run_inference()
        print("  -> results/inference_posteriors.json")

    if args.all or args.estimation:
        print("\n[3/7] Running parameter estimation...")
        run_estimation()
        print("  -> results/estimation_results.json")
        print("  -> results/mle_fitted_cpds_N1000.json")
        print("  -> results/mle_sweep_plot.png")

    if args.all or args.llm:
        print(f"\n[4/7] Running LLM tasks (backend: {'real' if args.use_real_llm else 'stub'})...")
        run_llm_tasks(use_real_llm=args.use_real_llm)
        print("  -> results/llm_generated_*.json")

    if args.all or args.validation:
        print("\n[5/7] Running validation tests...")
        run_validation_tests()
        print("  -> results/validation_tests.json")

    if args.all or args.original_outputs:
        print("\n[6/7] Saving original notebook outputs...")
        run_original_notebook_outputs()
        print("  -> results/original_notebook_outputs.json")

    # Always run exercises
    print("\n[7/7] Exercise results included in inference_posteriors.json")

    print("\n" + "=" * 60)
    print("All tasks completed. Results saved to results/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())