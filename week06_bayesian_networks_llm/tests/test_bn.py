# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Tests for core Bayesian network construction and validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from week06_bayesian_networks_llm.src.bn import (
    build_reference_model,
    compare_models,
    make_empty_structure,
    validate_cpt,
)


class TestReferenceModelStructure:
    """Tests for the reference Sprinkler Bayesian network structure."""

    def test_reference_model_structure(self):
        """Test that reference model has correct nodes and edges."""
        model = build_reference_model()

        expected_nodes = {"Cloudy", "Rain", "Sprinkler", "WetGrass"}
        expected_edges = {
            ("Cloudy", "Rain"),
            ("Cloudy", "Sprinkler"),
            ("Rain", "WetGrass"),
            ("Sprinkler", "WetGrass"),
        }

        assert set(model.nodes()) == expected_nodes
        assert set(model.edges()) == expected_edges

    def test_check_model_valid(self):
        """Test that reference model passes check_model()."""
        model = build_reference_model()
        assert model.check_model() is True

    def test_cpd_cloudy(self):
        """Test Cloudy CPD matches specification: P(Cloudy=1)=0.5."""
        model = build_reference_model()
        cpd = model.get_cpds("Cloudy")

        assert cpd.variable == "Cloudy"
        assert cpd.variable_card == 2
        assert cpd.evidence is None
        assert np.allclose(cpd.values.flatten(), [0.5, 0.5])

    def test_cpd_rain(self):
        """Test Rain CPD matches specification."""
        model = build_reference_model()
        cpd = model.get_cpds("Rain")

        assert cpd.variable == "Rain"
        assert cpd.variable_card == 2
        assert cpd.evidence == ["Cloudy"]
        assert cpd.evidence_card == [2]

        # P(Rain=0|Cloudy=0)=0.8, P(Rain=1|Cloudy=0)=0.2
        # P(Rain=0|Cloudy=1)=0.2, P(Rain=1|Cloudy=1)=0.8
        expected = np.array([[0.8, 0.2], [0.2, 0.8]])
        assert np.allclose(cpd.values, expected)

    def test_cpd_sprinkler(self):
        """Test Sprinkler CPD matches specification."""
        model = build_reference_model()
        cpd = model.get_cpds("Sprinkler")

        assert cpd.variable == "Sprinkler"
        assert cpd.variable_card == 2
        assert cpd.evidence == ["Cloudy"]
        assert cpd.evidence_card == [2]

        # P(Sprinkler=0|Cloudy=0)=0.5, P(Sprinkler=1|Cloudy=0)=0.5
        # P(Sprinkler=0|Cloudy=1)=0.9, P(Sprinkler=1|Cloudy=1)=0.1
        expected = np.array([[0.5, 0.9], [0.5, 0.1]])
        assert np.allclose(cpd.values, expected)

    def test_cpd_wetgrass(self):
        """Test WetGrass CPD matches specification."""
        model = build_reference_model()
        cpd = model.get_cpds("WetGrass")

        assert cpd.variable == "WetGrass"
        assert cpd.variable_card == 2
        assert cpd.evidence == ["Rain", "Sprinkler"]
        assert cpd.evidence_card == [2, 2]

        # Columns: (R=0,S=0), (R=0,S=1), (R=1,S=0), (R=1,S=1)
        # W=0: [0.99, 0.10, 0.10, 0.01]
        # W=1: [0.01, 0.90, 0.90, 0.99]
        expected = np.array([
            [0.99, 0.10, 0.10, 0.01],
            [0.01, 0.90, 0.90, 0.99],
        ])
        assert np.allclose(cpd.values, expected)

    def test_cpd_normalization(self):
        """Test that all CPDs are properly normalized (columns sum to 1)."""
        model = build_reference_model()
        assert validate_cpt(model) is True

    def test_cpd_no_negatives(self):
        """Test that no CPD has negative values."""
        model = build_reference_model()
        for cpd in model.get_cpds():
            assert np.all(cpd.values >= -1e-10)


class TestCPTValuesMatchNotebook:
    """Test that CPD values match the notebook exactly."""

    def test_cpd_values_match_notebook(self):
        """Verify exact values from notebook specification."""
        model = build_reference_model()

        # Cloudy
        cpd_c = model.get_cpds("Cloudy")
        assert cpd_c.values[0, 0] == 0.5  # P(C=0)
        assert cpd_c.values[1, 0] == 0.5  # P(C=1)

        # Rain
        cpd_r = model.get_cpds("Rain")
        assert cpd_r.values[1, 0] == 0.2  # P(R=1|C=0)
        assert cpd_r.values[1, 1] == 0.8  # P(R=1|C=1)

        # Sprinkler
        cpd_s = model.get_cpds("Sprinkler")
        assert cpd_s.values[1, 0] == 0.5  # P(S=1|C=0)
        assert cpd_s.values[1, 1] == 0.1  # P(S=1|C=1)

        # WetGrass
        cpd_w = model.get_cpds("WetGrass")
        # Column order: (R=0,S=0), (R=0,S=1), (R=1,S=0), (R=1,S=1)
        assert cpd_w.values[1, 0] == 0.01  # P(W=1|R=0,S=0)
        assert cpd_w.values[1, 1] == 0.90  # P(W=1|R=0,S=1)
        assert cpd_w.values[1, 2] == 0.90  # P(W=1|R=1,S=0)
        assert cpd_w.values[1, 3] == 0.99  # P(W=1|R=1,S=1)


class TestInvalidCPTRejection:
    """Tests that invalid CPTs are rejected."""

    def test_invalid_cpt_rejected(self):
        """Test that CPT with columns not summing to 1 is rejected."""
        from week06_bayesian_networks_llm.src.bn import DiscreteBayesianNetwork, TabularCPD

        broken_model = DiscreteBayesianNetwork([("Cloudy", "Rain")])
        cpd_cloudy_ok = TabularCPD("Cloudy", 2, [[0.5], [0.5]])
        # Invalid: for Cloudy=0, 0.9 + 0.3 = 1.2
        cpd_rain_broken = TabularCPD(
            "Rain", 2,
            [[0.9, 0.2], [0.3, 0.8]],
            evidence=["Cloudy"], evidence_card=[2]
        )
        broken_model.add_cpds(cpd_cloudy_ok, cpd_rain_broken)

        with pytest.raises(ValueError, match="not equal to 1"):
            broken_model.check_model()

    def test_validate_cpt_catches_broken(self):
        """Test that our validate_cpt function catches broken CPTs."""
        from week06_bayesian_networks_llm.src.bn import DiscreteBayesianNetwork, TabularCPD

        broken_model = DiscreteBayesianNetwork([("Cloudy", "Rain")])
        cpd_cloudy_ok = TabularCPD("Cloudy", 2, [[0.5], [0.5]])
        cpd_rain_broken = TabularCPD(
            "Rain", 2,
            [[0.9, 0.2], [0.3, 0.8]],
            evidence=["Cloudy"], evidence_card=[2]
        )
        broken_model.add_cpds(cpd_cloudy_ok, cpd_rain_broken)

        assert validate_cpt(broken_model) is False


class TestSemanticErrorDetection:
    """Tests for semantic errors (valid CPTs but wrong semantics)."""

    def test_semantic_error_detected_by_posterior(self):
        """Test that permuted WetGrass CPD passes check_model but gives wrong posterior."""
        from week06_bayesian_networks_llm.src.bn import TabularCPD
        from week06_bayesian_networks_llm.src.inference import VariableElimination

        # Build correct model
        correct_model = build_reference_model()
        correct_inference = VariableElimination(correct_model)
        correct_posterior = correct_inference.query(
            ["Rain"], evidence={"WetGrass": 1}, show_progress=False
        )

        # Build model with permuted WetGrass CPD
        wrong_model = build_reference_model()
        wrong_model.remove_cpds(wrong_model.get_cpds("WetGrass"))

        wrong_wetgrass = TabularCPD(
            "WetGrass", 2,
            [[0.99, 0.10, 0.01, 0.10], [0.01, 0.90, 0.99, 0.90]],
            evidence=["Rain", "Sprinkler"], evidence_card=[2, 2]
        )
        wrong_model.add_cpds(wrong_wetgrass)

        # check_model should still pass (columns sum to 1)
        assert wrong_model.check_model() is True

        # But posterior should be different
        wrong_inference = VariableElimination(wrong_model)
        wrong_posterior = wrong_inference.query(
            ["Rain"], evidence={"WetGrass": 1}, show_progress=False
        )

        correct_val = float(correct_posterior.values[1])
        wrong_val = float(wrong_posterior.values[1])

        assert not np.isclose(correct_val, wrong_val, atol=1e-10)
        # Notebook shows correct=0.704769..., wrong=0.717295...
        assert correct_val == pytest.approx(0.7047692307692307, rel=1e-10)
        assert wrong_val == pytest.approx(0.7172952268709487, rel=1e-10)


class TestModelComparison:
    """Tests for model comparison utilities."""

    def test_compare_identical_models(self):
        """Test that comparing a model with itself gives zero differences."""
        model_a = build_reference_model()
        model_b = build_reference_model()

        diff = compare_models(model_a, model_b)
        for _var, d in diff.items():
            assert d < 1e-10

    def test_compare_different_models(self):
        """Test that comparing different models shows differences."""
        model_a = build_reference_model()
        model_b = make_empty_structure()
        # Don't add CPDs to model_b - comparison should show inf
        diff = compare_models(model_a, model_b)
        for _var, d in diff.items():
            assert d == float("inf")


class TestMakeEmptyStructure:
    """Tests for make_empty_structure function."""

    def test_empty_structure_has_correct_edges(self):
        """Test that empty structure has correct edges but no CPDs."""
        model = make_empty_structure()
        expected_edges = {
            ("Cloudy", "Rain"),
            ("Cloudy", "Sprinkler"),
            ("Rain", "WetGrass"),
            ("Sprinkler", "WetGrass"),
        }
        assert set(model.edges()) == expected_edges
        assert len(model.get_cpds()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])