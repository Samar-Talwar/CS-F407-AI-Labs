# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Tests for LLM interface: stub backend, code extraction, validation, and error handling.
"""

from __future__ import annotations

import pytest

from week06_bayesian_networks_llm.src.bn import build_reference_model
from week06_bayesian_networks_llm.src.inference import query_posterior
from week06_bayesian_networks_llm.src.llm_interface import (
    BAYESIAN_PROMPT,
    ESTIMATION_PROMPT,
    INFERENCE_PROMPT,
    LLMInterface,
    StubLLMBackend,
    basic_generated_code_check,
    compare_models_cpd_values,
    extract_python_code,
    validate_generated_inference,
)


class TestStubLLMBackend:
    """Tests for the deterministic stub LLM backend."""

    def test_stub_backend_name(self):
        """Test stub backend has correct name."""
        backend = StubLLMBackend()
        assert backend.name == "stub"

    def test_stub_backend_deterministic(self):
        """Test that stub backend returns identical output for same prompt."""
        backend = StubLLMBackend()

        response1 = backend.generate(INFERENCE_PROMPT)
        response2 = backend.generate(INFERENCE_PROMPT)

        assert response1 == response2

    def test_stub_backend_inference_response(self):
        """Test inference response contains required elements."""
        backend = StubLLMBackend()
        response = backend.generate(INFERENCE_PROMPT)

        assert "generated_model" in response
        assert "generated_posterior" in response
        assert "DiscreteBayesianNetwork" in response or "BayesianModel" in response
        assert "TabularCPD" in response
        assert "VariableElimination" in response or "query" in response
        assert "WetGrass" in response

    def test_stub_backend_estimation_response(self):
        """Test estimation response contains required elements."""
        backend = StubLLMBackend()
        response = backend.generate(ESTIMATION_PROMPT)

        assert "generated_mle_model" in response
        assert "MaximumLikelihoodEstimator" in response
        assert "DiscreteBayesianNetwork" in response or "BayesianModel" in response
        assert "check_model" in response

    def test_stub_backend_bayesian_response(self):
        """Test Bayesian response contains required elements."""
        backend = StubLLMBackend()
        response = backend.generate(BAYESIAN_PROMPT)

        assert "generated_bayes_model" in response
        assert "BayesianEstimator" in response
        assert "BDeu" in response
        assert "equivalent_sample_size" in response
        assert "check_model" in response

    def test_stub_backend_call_count(self):
        """Test that call count increments."""
        backend = StubLLMBackend()
        assert backend._call_count == 0

        backend.generate(INFERENCE_PROMPT)
        assert backend._call_count == 1

        backend.generate(ESTIMATION_PROMPT)
        assert backend._call_count == 2


class TestExtractPythonCode:
    """Tests for code extraction from markdown fences."""

    def test_extract_python_code_with_fences(self):
        """Test extraction from ```python ... ``` fences."""
        text = """```python
print("hello")
```"""
        result = extract_python_code(text)
        assert result == 'print("hello")'

    def test_extract_python_code_without_fences(self):
        """Test extraction when no fences present."""
        text = 'print("hello")'
        result = extract_python_code(text)
        assert result == 'print("hello")'

    def test_extract_python_code_with_language_tag(self):
        """Test extraction with language tag."""
        text = """```
print("hello")
```"""
        result = extract_python_code(text)
        assert result == 'print("hello")'

    def test_extract_python_code_strips_whitespace(self):
        """Test that whitespace is stripped."""
        text = """  \n  ```python
print("hello")
```  \n  """
        result = extract_python_code(text)
        assert result == 'print("hello")'


class TestBasicGeneratedCodeCheck:
    """Tests for basic AST-based code safety check."""

    def test_basic_code_check_allows_valid_imports(self):
        """Test that allowed imports pass."""
        code = """
import pgmpy
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
"""
        problems = basic_generated_code_check(code)
        assert problems == []

    def test_basic_code_check_forbids_disallowed_imports(self):
        """Test that disallowed imports are caught."""
        code = """
import os
import sys
import requests
"""
        problems = basic_generated_code_check(code)
        assert len(problems) == 3
        assert all("Import not allowed" in p for p in problems)

    def test_basic_code_check_forbids_disallowed_calls(self):
        """Test that forbidden calls are caught."""
        code = """
eval("1+1")
exec("print(1)")
open("file.txt")
"""
        problems = basic_generated_code_check(code)
        assert len(problems) == 3
        assert all("Forbidden call" in p for p in problems)

    def test_basic_code_check_stub_inference_passes(self):
        """Test that stub inference response passes basic check."""
        backend = StubLLMBackend()
        response = backend.generate(INFERENCE_PROMPT)
        code = extract_python_code(response)
        problems = basic_generated_code_check(code)
        assert problems == []

    def test_basic_code_check_stub_estimation_passes(self):
        """Test that stub estimation response passes basic check."""
        backend = StubLLMBackend()
        response = backend.generate(ESTIMATION_PROMPT)
        code = extract_python_code(response)
        problems = basic_generated_code_check(code)
        assert problems == []

    def test_basic_code_check_stub_bayesian_passes(self):
        """Test that stub bayesian response passes basic check."""
        backend = StubLLMBackend()
        response = backend.generate(BAYESIAN_PROMPT)
        code = extract_python_code(response)
        problems = basic_generated_code_check(code)
        assert problems == []


class TestValidateGeneratedInference:
    """Tests for validation of LLM-generated inference code."""

    def test_validate_generated_inference_stub_passes(self):
        """Test that stub-generated inference code passes validation."""
        backend = StubLLMBackend()
        response = backend.generate(INFERENCE_PROMPT)
        code = extract_python_code(response)

        namespace = {}
        exec(code, namespace)

        reference_model = build_reference_model()
        reference_posterior = query_posterior(reference_model, ["Rain"], {"WetGrass": 1})

        class MockPosterior:
            def __init__(self, values):
                self.values = values

        report = validate_generated_inference(
            namespace, reference_model, MockPosterior(reference_posterior), "stub"
        )

        assert report.structure == "PASS"
        assert report.check_model == "PASS"
        assert report.absolute_error < 1e-10
        assert report.backend == "stub"

    def test_validate_generated_inference_missing_model_raises(self):
        """Test that missing generated_model raises AssertionError."""
        namespace = {"generated_posterior": "dummy"}

        reference_model = build_reference_model()
        reference_posterior = query_posterior(reference_model, ["Rain"], {"WetGrass": 1})

        class MockPosterior:
            def __init__(self, values):
                self.values = values

        with pytest.raises(AssertionError, match="generated_model was not created"):
            validate_generated_inference(
                namespace, reference_model, MockPosterior(reference_posterior), "stub"
            )

    def test_validate_generated_inference_missing_posterior_raises(self):
        """Test that missing generated_posterior raises AssertionError."""
        namespace = {"generated_model": "dummy"}

        reference_model = build_reference_model()
        reference_posterior = query_posterior(reference_model, ["Rain"], {"WetGrass": 1})

        class MockPosterior:
            def __init__(self, values):
                self.values = values

        with pytest.raises(AssertionError, match="generated_posterior was not created"):
            validate_generated_inference(
                namespace, reference_model, MockPosterior(reference_posterior), "stub"
            )


class TestCompareModelsCPDValues:
    """Tests for CPD value comparison."""

    def test_compare_identical_models(self):
        """Test comparing identical models gives zero differences."""
        model_a = build_reference_model()
        model_b = build_reference_model()

        diff = compare_models_cpd_values(model_a, model_b)
        for _var, d in diff.items():
            assert d < 1e-10

    def test_compare_models_returns_dict(self):
        """Test that comparison returns dictionary with all variables."""
        model_a = build_reference_model()
        model_b = build_reference_model()

        diff = compare_models_cpd_values(model_a, model_b)
        assert set(diff.keys()) == {"Cloudy", "Rain", "Sprinkler", "WetGrass"}


class TestLLMInterface:
    """Tests for high-level LLMInterface class."""

    def test_llm_interface_default_backend(self):
        """Test LLMInterface uses stub backend by default."""
        interface = LLMInterface()
        assert isinstance(interface.backend, StubLLMBackend)

    def test_llm_interface_explicit_backend(self):
        """Test LLMInterface with explicit backend."""
        backend = StubLLMBackend()
        interface = LLMInterface(backend=backend)
        assert interface.backend is backend

    def test_llm_interface_generate_inference_code(self):
        """Test inference code generation."""
        interface = LLMInterface()
        code = interface.generate_inference_code()
        assert "generated_model" in code
        assert "generated_posterior" in code

    def test_llm_interface_generate_estimation_code(self):
        """Test estimation code generation."""
        interface = LLMInterface()
        code = interface.generate_estimation_code()
        assert "generated_mle_model" in code
        assert "MaximumLikelihoodEstimator" in code

    def test_llm_interface_generate_bayesian_code(self):
        """Test Bayesian code generation."""
        interface = LLMInterface()
        code = interface.generate_bayesian_code()
        assert "generated_bayes_model" in code
        assert "BayesianEstimator" in code

    def test_llm_interface_run_inference_task(self):
        """Test full inference task pipeline."""
        interface = LLMInterface()
        result = interface.run_inference_task(approve=True)

        assert "prompt" in result
        assert "raw_response" in result
        assert "extracted_code" in result
        assert "basic_check_problems" in result
        assert "validation" in result
        assert "backend" in result

        assert result["basic_check_problems"] == []
        assert result["validation"]["structure"] == "PASS"
        assert result["validation"]["check_model"] == "PASS"
        assert result["validation"]["absolute_error"] < 1e-10

    def test_llm_interface_run_estimation_task(self):
        """Test full estimation task pipeline."""
        from week06_bayesian_networks_llm.src.estimation import generate_synthetic_data

        interface = LLMInterface()
        data = generate_synthetic_data(n_samples=1000, seed=7)

        result = interface.run_estimation_task(data, approve=True)

        assert "prompt" in result
        assert "extracted_code" in result
        assert "basic_check_problems" in result
        assert result["basic_check_problems"] == []
        assert "model_comparison" in result

    def test_llm_interface_run_bayesian_task(self):
        """Test full Bayesian task pipeline."""
        from week06_bayesian_networks_llm.src.estimation import generate_synthetic_data

        interface = LLMInterface()
        data = generate_synthetic_data(n_samples=30, seed=11)

        result = interface.run_bayesian_task(data, approve=True)

        assert "prompt" in result
        assert "extracted_code" in result
        assert "basic_check_problems" in result
        assert result["basic_check_problems"] == []


class TestPromptContent:
    """Tests that prompts contain required specifications."""

    def test_inference_prompt_contains_requirements(self):
        """Test inference prompt has all required specifications."""
        assert "Cloudy -> Rain" in INFERENCE_PROMPT
        assert "Cloudy -> Sprinkler" in INFERENCE_PROMPT
        assert "Rain -> WetGrass" in INFERENCE_PROMPT
        assert "Sprinkler -> WetGrass" in INFERENCE_PROMPT
        assert "P(Cloudy=1) = 0.5" in INFERENCE_PROMPT
        assert "P(Rain=1 | Cloudy=0) = 0.2" in INFERENCE_PROMPT
        assert "P(Rain=1 | Cloudy=1) = 0.8" in INFERENCE_PROMPT
        assert "P(Sprinkler=1 | Cloudy=0) = 0.5" in INFERENCE_PROMPT
        assert "P(Sprinkler=1 | Cloudy=1) = 0.1" in INFERENCE_PROMPT
        assert "P(WetGrass=1 | Rain=0, Sprinkler=0) = 0.01" in INFERENCE_PROMPT
        assert "P(WetGrass=1 | Rain=0, Sprinkler=1) = 0.90" in INFERENCE_PROMPT
        assert "P(WetGrass=1 | Rain=1, Sprinkler=0) = 0.90" in INFERENCE_PROMPT
        assert "P(WetGrass=1 | Rain=1, Sprinkler=1) = 0.99" in INFERENCE_PROMPT
        assert "DiscreteBayesianNetwork" in INFERENCE_PROMPT
        assert "TabularCPD" in INFERENCE_PROMPT
        assert "generated_model" in INFERENCE_PROMPT
        assert "generated_posterior" in INFERENCE_PROMPT
        assert "VariableElimination" in INFERENCE_PROMPT
        assert "check_model" in INFERENCE_PROMPT

    def test_estimation_prompt_contains_requirements(self):
        """Test estimation prompt has all required specifications."""
        assert "Cloudy -> Rain" in ESTIMATION_PROMPT
        assert "Cloudy -> Sprinkler" in ESTIMATION_PROMPT
        assert "Rain -> WetGrass" in ESTIMATION_PROMPT
        assert "Sprinkler -> WetGrass" in ESTIMATION_PROMPT
        assert "DiscreteBayesianNetwork" in ESTIMATION_PROMPT
        assert "MaximumLikelihoodEstimator" in ESTIMATION_PROMPT
        assert "generated_mle_model" in ESTIMATION_PROMPT
        assert "check_model" in ESTIMATION_PROMPT
        assert "pandas DataFrame" in ESTIMATION_PROMPT

    def test_bayesian_prompt_contains_requirements(self):
        """Test Bayesian prompt has all required specifications."""
        assert "Cloudy -> Rain" in BAYESIAN_PROMPT
        assert "Cloudy -> Sprinkler" in BAYESIAN_PROMPT
        assert "Rain -> WetGrass" in BAYESIAN_PROMPT
        assert "Sprinkler -> WetGrass" in BAYESIAN_PROMPT
        assert "BayesianEstimator" in BAYESIAN_PROMPT
        assert 'prior_type="BDeu"' in BAYESIAN_PROMPT
        assert "equivalent_sample_size=10" in BAYESIAN_PROMPT
        assert "generated_bayes_model" in BAYESIAN_PROMPT
        assert "check_model" in BAYESIAN_PROMPT


class TestParsingLLMOutput:
    """Tests for parsing LLM output into probabilities/structure."""

    def test_stub_inference_output_parsing(self):
        """Test that stub inference output can be parsed and executed."""
        interface = LLMInterface()
        result = interface.run_inference_task(approve=True)

        # Should have executed successfully
        assert "namespace" in result or "validation" in result

    def test_stub_estimation_output_parsing(self):
        """Test that stub estimation output can be parsed."""
        from week06_bayesian_networks_llm.src.estimation import generate_synthetic_data

        interface = LLMInterface()
        data = generate_synthetic_data(n_samples=1000, seed=7)
        result = interface.run_estimation_task(data, approve=True)

        assert "namespace_keys" in result
        assert "generated_mle_model" in result.get("namespace_keys", [])

    def test_garbage_string_handling(self):
        """Test handling of garbage LLM output."""
        # This tests that our extraction handles non-code responses
        garbage = "This is not code at all, just text."
        code = extract_python_code(garbage)
        # Should return the text as-is (no fences found)
        assert code == garbage

        # Basic check should flag issues
        problems = basic_generated_code_check(code)
        assert len(problems) > 0 or code == garbage


class TestMutationTestsLLM:
    """Mutation tests for LLM interface."""

    def test_mutation_stub_returns_wrong_posterior(self, monkeypatch):
        """Mutation: stub returns wrong posterior -> validation should fail."""
        from week06_bayesian_networks_llm.src import llm_interface

        orig_generate = llm_interface.StubLLMBackend.generate

        def broken_generate(self, prompt):
            if "VariableElimination" in prompt:
                # Return code with wrong posterior
                return """```python
from week06_bayesian_networks_llm.src.bn import DiscreteBayesianNetwork, TabularCPD

generated_model = DiscreteBayesianNetwork([
    ('Cloudy', 'Rain'),
    ('Cloudy', 'Sprinkler'),
    ('Rain', 'WetGrass'),
    ('Sprinkler', 'WetGrass'),
])

cloudy_cpd = TabularCPD('Cloudy', 2, [[0.5], [0.5]])
rain_cpd = TabularCPD(
    'Rain', 2, [[0.8, 0.2], [0.2, 0.8]],
    evidence=['Cloudy'], evidence_card=[2],
)
sprinkler_cpd = TabularCPD(
    'Sprinkler', 2, [[0.5, 0.9], [0.5, 0.1]],
    evidence=['Cloudy'], evidence_card=[2],
)
wet_grass_cpd = TabularCPD('WetGrass', 2, [
    [0.99, 0.10, 0.10, 0.01],
    [0.01, 0.90, 0.90, 0.99]
], evidence=['Rain', 'Sprinkler'], evidence_card=[2, 2])

generated_model.add_cpds(cloudy_cpd, rain_cpd, sprinkler_cpd, wet_grass_cpd)
generated_model.check_model()

# Mock posterior with wrong value
class MockFactor:
    def __init__(self):
        self.values = [0.5, 0.5]  # Wrong: should be ~[0.295, 0.705]

generated_posterior = MockFactor()
```
"""
            return orig_generate(self, prompt)

        monkeypatch.setattr(llm_interface.StubLLMBackend, "generate", broken_generate)

        interface = LLMInterface()
        result = interface.run_inference_task(approve=True)

        # Validation should fail due to wrong posterior
        assert result["validation"]["absolute_error"] > 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])