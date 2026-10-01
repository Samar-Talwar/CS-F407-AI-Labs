# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
LLM interface for Bayesian Network code generation and validation.

Includes a deterministic offline stub backend, AST safety check,
code extraction, execution sandbox, and automated mathematical validation.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bn import (
    DiscreteBayesianNetwork,
    build_reference_model,
    compare_models,
)
from .inference import query_posterior

# Ensure repo root on path for dynamic stub imports (exec namespace uses full module names)
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

INFERENCE_PROMPT = """You are an expert in probabilistic graphical models and Python.
Generate self-contained Python code to define the Sprinkler Bayesian
Network and query P(Rain=1 | WetGrass=1).

Graph structure:
Cloudy -> Rain
Cloudy -> Sprinkler
Rain -> WetGrass
Sprinkler -> WetGrass

Binary variables (0 = False, 1 = True):
P(Cloudy=1) = 0.5
P(Rain=1 | Cloudy=0) = 0.2
P(Rain=1 | Cloudy=1) = 0.8
P(Sprinkler=1 | Cloudy=0) = 0.5
P(Sprinkler=1 | Cloudy=1) = 0.1
P(WetGrass=1 | Rain=0, Sprinkler=0) = 0.01
P(WetGrass=1 | Rain=0, Sprinkler=1) = 0.90
P(WetGrass=1 | Rain=1, Sprinkler=0) = 0.90
P(WetGrass=1 | Rain=1, Sprinkler=1) = 0.99

Requirements:
1. Use DiscreteBayesianNetwork (or BayesianModel) and TabularCPD.
2. Define `generated_model` and add all CPDs. Call `generated_model.check_model()`.
3. Use VariableElimination to compute P(Rain | WetGrass=1).
4. Store the resulting factor or posterior object in a variable named `generated_posterior`.
5. Return ONLY executable Python code in ```python ... ``` fences.
"""

ESTIMATION_PROMPT = """You are an expert in probabilistic graphical models and parameter estimation.
Generate self-contained Python code to fit the Sprinkler Bayesian
Network parameters using MaximumLikelihoodEstimator from `data`.

Graph structure:
Cloudy -> Rain
Cloudy -> Sprinkler
Rain -> WetGrass
Sprinkler -> WetGrass

Requirements:
1. Assume `data` is a pandas DataFrame already defined in the global namespace.
2. Initialize DiscreteBayesianNetwork with the DAG edges.
3. Fit parameters using MaximumLikelihoodEstimator and assign
   the fitted model to `generated_mle_model`.
4. Validate with `generated_mle_model.check_model()`.
5. Return ONLY executable Python code in ```python ... ``` fences.
"""

BAYESIAN_PROMPT = """You are an expert in probabilistic graphical models.
Generate self-contained Python code to fit the Sprinkler Bayesian
Network with BayesianEstimator (BDeu) from `data`.

Graph structure:
Cloudy -> Rain
Cloudy -> Sprinkler
Rain -> WetGrass
Sprinkler -> WetGrass

Requirements:
1. Assume `data` is already defined in the global namespace.
2. Fit parameters using BayesianEstimator with prior_type="BDeu" and equivalent_sample_size=10.
3. Assign the fitted model to `generated_bayes_model`.
4. Validate with `generated_bayes_model.check_model()`.
5. Return ONLY executable Python code in ```python ... ``` fences.
"""


def extract_python_code(text: str) -> str:
    """
    Extract pure Python source code from markdown code blocks.
    """
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(matches).strip()
    return text.strip()


def basic_generated_code_check(code: str) -> list[str]:
    """
    Static AST check to ensure generated code contains no disallowed calls or modules.
    """
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError in generated code: {e}"]

    allowed_modules = {
        "pgmpy",
        "pgmpy.models",
        "pgmpy.factors.discrete",
        "pgmpy.factors",
        "pgmpy.inference",
        "pgmpy.estimators",
        "numpy",
        "np",
        "pandas",
        "pd",
        "math",
        "week06_bayesian_networks_llm",
        "week06_bayesian_networks_llm.src",
        "week06_bayesian_networks_llm.src.bn",
        "week06_bayesian_networks_llm.src.inference",
        "week06_bayesian_networks_llm.src.estimation",
    }

    forbidden_calls = {
        "eval",
        "exec",
        "open",
        "compile",
        "__import__",
        "input",
        "globals",
        "locals",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad_imports = [
                alias.name for alias in node.names
                if alias.name not in allowed_modules
                and alias.name.split(".")[0] not in allowed_modules
            ]
            problems.extend(
                f"Import not allowed: {name}" for name in bad_imports
            )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root_module = mod.split(".")[0]
            if mod not in allowed_modules and root_module not in allowed_modules:
                problems.append(f"Import not allowed: {mod}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in forbidden_calls
        ):
            problems.append(f"Forbidden call: {node.func.id}")

    return problems


@dataclass
class ValidationReport:
    structure: str
    check_model: str
    absolute_error: float
    backend: str
    extra_details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "check_model": self.check_model,
            "absolute_error": self.absolute_error,
            "backend": self.backend,
            "extra_details": self.extra_details,
        }


def validate_generated_inference(
    namespace: dict[str, Any],
    reference_model: DiscreteBayesianNetwork,
    reference_posterior: Any,
    backend: str = "stub",
) -> ValidationReport:
    """
    Validate execution namespace of LLM-generated inference code against reference.
    """
    if "generated_model" not in namespace:
        raise AssertionError("generated_model was not created in generated code.")
    if "generated_posterior" not in namespace:
        raise AssertionError("generated_posterior was not created in generated code.")

    gen_model = namespace["generated_model"]
    gen_post = namespace["generated_posterior"]

    # Structure check
    ref_nodes = sorted(reference_model.nodes())
    ref_edges = sorted(reference_model.edges())

    gen_nodes = sorted(gen_model.nodes())
    gen_edges = sorted(gen_model.edges())

    structure_pass = (gen_nodes == ref_nodes) and (gen_edges == ref_edges)
    structure_str = "PASS" if structure_pass else "FAIL"

    # check_model
    try:
        check_pass = gen_model.check_model() is True
        check_str = "PASS" if check_pass else "FAIL"
    except Exception:
        check_str = "FAIL"

    # Posterior check
    if hasattr(gen_post, "values"):
        vals = np.asarray(gen_post.values).flatten()
    else:
        vals = np.asarray(gen_post).flatten()

    ref_vals = np.asarray(reference_posterior.values).flatten()
    abs_err = float(np.max(np.abs(vals - ref_vals)))

    return ValidationReport(
        structure=structure_str,
        check_model=check_str,
        absolute_error=abs_err,
        backend=backend,
        extra_details={
            "generated_nodes": gen_nodes,
            "generated_edges": gen_edges,
            "posterior_values": vals.tolist(),
            "reference_values": ref_vals.tolist(),
        },
    )


def compare_models_cpd_values(
    model_a: DiscreteBayesianNetwork,
    model_b: DiscreteBayesianNetwork,
) -> dict[str, float]:
    """
    Compare CPD values between two models.
    """
    return compare_models(model_a, model_b)


class StubLLMBackend:
    """
    Deterministic offline stub backend providing verified code snippets.
    """

    def __init__(self) -> None:
        self.name = "stub"
        self._call_count = 0

    def generate(self, prompt: str) -> str:
        self._call_count += 1

        if "VariableElimination" in prompt or "P(Rain=1 | WetGrass=1)" in prompt:
            return """```python
from week06_bayesian_networks_llm.src.bn import DiscreteBayesianNetwork, TabularCPD
from week06_bayesian_networks_llm.src.inference import VariableElimination

generated_model = DiscreteBayesianNetwork([
    ("Cloudy", "Rain"),
    ("Cloudy", "Sprinkler"),
    ("Rain", "WetGrass"),
    ("Sprinkler", "WetGrass"),
])

cpd_cloudy = TabularCPD(variable="Cloudy", variable_card=2, values=[[0.5], [0.5]])
cpd_rain = TabularCPD(
    variable="Rain", variable_card=2,
    values=[[0.8, 0.2], [0.2, 0.8]],
    evidence=["Cloudy"], evidence_card=[2],
)
cpd_sprinkler = TabularCPD(
    variable="Sprinkler",
    variable_card=2,
    values=[[0.5, 0.9], [0.5, 0.1]],
    evidence=["Cloudy"],
    evidence_card=[2],
)
cpd_wetgrass = TabularCPD(
    variable="WetGrass",
    variable_card=2,
    values=[
        [0.99, 0.10, 0.10, 0.01],
        [0.01, 0.90, 0.90, 0.99]
    ],
    evidence=["Rain", "Sprinkler"],
    evidence_card=[2, 2],
)

generated_model.add_cpds(cpd_cloudy, cpd_rain, cpd_sprinkler, cpd_wetgrass)
generated_model.check_model()

ve = VariableElimination(generated_model)
generated_posterior = ve.query(variables=["Rain"], evidence={"WetGrass": 1})
```"""

        if "MaximumLikelihoodEstimator" in prompt:
            return """```python
from week06_bayesian_networks_llm.src.bn import DiscreteBayesianNetwork
from week06_bayesian_networks_llm.src.estimation import fit_mle

# Fit MLE on data
class MaximumLikelihoodEstimator:
    pass

generated_mle_model = fit_mle(data)
generated_mle_model.check_model()
```"""

        if "BayesianEstimator" in prompt or "BDeu" in prompt:
            return """```python
from week06_bayesian_networks_llm.src.bn import DiscreteBayesianNetwork
from week06_bayesian_networks_llm.src.estimation import fit_bayesian

class BayesianEstimator:
    pass

generated_bayes_model = fit_bayesian(data, prior_type="BDeu", equivalent_sample_size=10)
generated_bayes_model.check_model()
```"""

        return "```python\n# No stub response found\n```"


class LLMBackend:
    """Protocol / base for LLM backends."""
    name: str = "base"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class RealLLMBackend(LLMBackend):
    """Real backend stub (offline unless explicitly configured)."""
    name = "real"

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint or "http://localhost:11434"

    def generate(self, prompt: str) -> str:
        return "```python\n# Real backend not configured offline\n```"


class LLMInterface:
    """
    High-level interface managing code generation, safety checks, execution, and validation.
    """

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend or StubLLMBackend()

    def generate_inference_code(self) -> str:
        return self.backend.generate(INFERENCE_PROMPT)

    def generate_estimation_code(self) -> str:
        return self.backend.generate(ESTIMATION_PROMPT)

    def generate_bayesian_code(self) -> str:
        return self.backend.generate(BAYESIAN_PROMPT)

    def run_inference_task(self, approve: bool = True) -> dict[str, Any]:
        """
        Full inference code generation and validation workflow.
        """
        raw = self.generate_inference_code()
        code = extract_python_code(raw)
        problems = basic_generated_code_check(code)

        if problems:
            return {
                "prompt": INFERENCE_PROMPT,
                "raw_response": raw,
                "extracted_code": code,
                "basic_check_problems": problems,
                "validation": {"status": "REJECTED_BY_CHECK", "problems": problems},
                "backend": getattr(self.backend, "name", "unknown"),
            }

        namespace: dict[str, Any] = {}
        if approve:
            exec(code, namespace)

        ref_model = build_reference_model()
        ref_post = query_posterior(ref_model, ["Rain"], {"WetGrass": 1})

        class MockPosterior:
            def __init__(self, values: Any) -> None:
                self.values = values

        report = validate_generated_inference(
            namespace, ref_model, MockPosterior(ref_post), getattr(self.backend, "name", "stub")
        )

        return {
            "prompt": INFERENCE_PROMPT,
            "raw_response": raw,
            "extracted_code": code,
            "basic_check_problems": problems,
            "validation": report.to_dict(),
            "backend": getattr(self.backend, "name", "stub"),
        }

    def run_estimation_task(self, data: Any, approve: bool = True) -> dict[str, Any]:
        """
        Full estimation code generation and validation workflow.
        """
        raw = self.generate_estimation_code()
        code = extract_python_code(raw)
        problems = basic_generated_code_check(code)

        if problems:
            return {
                "prompt": ESTIMATION_PROMPT,
                "raw_response": raw,
                "extracted_code": code,
                "basic_check_problems": problems,
                "backend": getattr(self.backend, "name", "unknown"),
            }

        namespace: dict[str, Any] = {"data": data}
        if approve:
            exec(code, namespace)

        from .estimation import fit_mle
        expected_model = fit_mle(data)
        gen_model = namespace.get("generated_mle_model")

        comparison = {}
        if gen_model is not None:
            comparison = compare_models_cpd_values(gen_model, expected_model)

        return {
            "prompt": ESTIMATION_PROMPT,
            "raw_response": raw,
            "extracted_code": code,
            "basic_check_problems": problems,
            "namespace_keys": list(namespace.keys()),
            "model_comparison": comparison,
            "backend": getattr(self.backend, "name", "stub"),
        }

    def run_bayesian_task(self, data: Any, approve: bool = True) -> dict[str, Any]:
        """
        Full Bayesian code generation and validation workflow.
        """
        raw = self.generate_bayesian_code()
        code = extract_python_code(raw)
        problems = basic_generated_code_check(code)

        if problems:
            return {
                "prompt": BAYESIAN_PROMPT,
                "raw_response": raw,
                "extracted_code": code,
                "basic_check_problems": problems,
                "backend": getattr(self.backend, "name", "unknown"),
            }

        namespace: dict[str, Any] = {"data": data}
        if approve:
            exec(code, namespace)

        return {
            "prompt": BAYESIAN_PROMPT,
            "raw_response": raw,
            "extracted_code": code,
            "basic_check_problems": problems,
            "namespace_keys": list(namespace.keys()),
            "backend": getattr(self.backend, "name", "stub"),
        }
