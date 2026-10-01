# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Week 6: Bayesian Networks and LLM Integration

This package provides:
- Core Bayesian network construction (bn.py)
- Exact inference with variable elimination and enumeration (inference.py)
- Parameter estimation: MLE and Bayesian (estimation.py)
- LLM code generation interface with stub and real backends (llm_interface.py)
- CLI entry point (cli.py)
"""

from .bn import (
    build_reference_model,
    compare_models,
    get_cpd_values_map,
    make_empty_structure,
    validate_cpt,
)
from .estimation import (
    fit_bayesian,
    fit_mle,
    generate_synthetic_data,
    multi_seed_estimates,
    rain_c1_probability,
    run_estimation_experiments,
    sample_size_sweep,
)
from .inference import (
    compute_posterior_by_enumeration,
    enumerate_joint,
    query_posterior,
    run_inference_exercises,
)
from .llm_interface import (
    LLMBackend,
    LLMInterface,
    RealLLMBackend,
    StubLLMBackend,
    ValidationReport,
    basic_generated_code_check,
    extract_python_code,
    validate_generated_inference,
)

__all__ = [
    # bn
    "build_reference_model",
    "validate_cpt",
    "make_empty_structure",
    "get_cpd_values_map",
    "compare_models",
    # inference
    "query_posterior",
    "enumerate_joint",
    "compute_posterior_by_enumeration",
    "run_inference_exercises",
    # estimation
    "generate_synthetic_data",
    "fit_mle",
    "fit_bayesian",
    "rain_c1_probability",
    "sample_size_sweep",
    "multi_seed_estimates",
    "run_estimation_experiments",
    # llm_interface
    "LLMInterface",
    "StubLLMBackend",
    "RealLLMBackend",
    "LLMBackend",
    "extract_python_code",
    "basic_generated_code_check",
    "validate_generated_inference",
    "ValidationReport",
]

__version__ = "1.0.0"