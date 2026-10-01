# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Tests for exact inference: variable elimination and enumeration verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from week06_bayesian_networks_llm.src.bn import build_reference_model
from week06_bayesian_networks_llm.src.inference import (
    compute_posterior_by_enumeration,
    compute_posterior_by_enumeration_dict,
    enumerate_joint,
    query_posterior,
    run_inference_exercises,
)


class TestVariableElimination:
    """Tests for VariableElimination-based inference."""

    def test_posterior_rain_given_wetgrass(self):
        """Test P(Rain=1 | WetGrass=1) using VariableElimination."""
        model = build_reference_model()
        posterior = query_posterior(model, ["Rain"], {"WetGrass": 1})

        # Notebook result: 0.7047692307692307
        assert posterior[1] == pytest.approx(0.7047692307692307, rel=1e-12)

    def test_posterior_rain_given_wetgrass_exact(self):
        """Test exact value from notebook."""
        model = build_reference_model()
        posterior = query_posterior(model, ["Rain"], {"WetGrass": 1})

        expected = 0.7047692307692307
        assert abs(posterior[1] - expected) < 1e-12

    def test_posterior_sums_to_one(self):
        """Test that posterior distribution sums to 1."""
        model = build_reference_model()
        posterior = query_posterior(model, ["Rain"], {"WetGrass": 1})

        assert abs(posterior.sum() - 1.0) < 1e-10

    def test_posterior_no_negatives(self):
        """Test that posterior has no negative values."""
        model = build_reference_model()
        posterior = query_posterior(model, ["Rain"], {"WetGrass": 1})

        assert np.all(posterior >= -1e-10)

    def test_exercise_sprinkler_given_wetgrass(self):
        """Test P(Sprinkler=1 | WetGrass=1) = 0.4278461538461539."""
        model = build_reference_model()
        posterior = query_posterior(model, ["Sprinkler"], {"WetGrass": 1})

        expected = 0.4278461538461539
        assert posterior[1] == pytest.approx(expected, rel=1e-12)

    def test_exercise_cloudy_given_wetgrass(self):
        """Test P(Cloudy=1 | WetGrass=1) = 0.5746153846153845."""
        model = build_reference_model()
        posterior = query_posterior(model, ["Cloudy"], {"WetGrass": 1})

        expected = 0.5746153846153845
        assert posterior[1] == pytest.approx(expected, rel=1e-12)

    def test_exercise_rain_given_wetgrass_sprinkler0(self):
        """Test P(Rain=1 | WetGrass=1, Sprinkler=0) = 0.9922022048937886."""
        model = build_reference_model()
        posterior = query_posterior(model, ["Rain"], {"WetGrass": 1, "Sprinkler": 0})

        expected = 0.9922022048937886
        assert posterior[1] == pytest.approx(expected, rel=1e-12)


class TestEnumerationOracle:
    """Independent oracle: brute-force enumeration of full joint."""

    def test_enumeration_matches_variable_elimination(self):
        """Test that enumeration matches VariableElimination for main query."""
        model = build_reference_model()

        ve_result = query_posterior(model, ["Rain"], {"WetGrass": 1})[1]
        enum_result = compute_posterior_by_enumeration(model, "Rain", 1, {"WetGrass": 1})

        # Notebook shows difference ~2.22e-16
        assert abs(ve_result - enum_result) < 1e-12

    def test_enumeration_diff_tolerance(self):
        """Test that enumeration difference is within tolerance."""
        model = build_reference_model()

        ve_result = query_posterior(model, ["Rain"], {"WetGrass": 1})[1]
        enum_result = compute_posterior_by_enumeration(model, "Rain", 1, {"WetGrass": 1})

        diff = abs(ve_result - enum_result)
        # Notebook: 2.220446049250313e-16
        assert diff < 1e-12

    def test_enumerate_joint_complete(self):
        """Test that enumerate_joint produces 16 assignments for 4 binary vars."""
        model = build_reference_model()
        joint = enumerate_joint(model)

        assert len(joint) == 16  # 2^4

        # Sum of all joint probabilities should be 1
        total = sum(joint.values())
        assert abs(total - 1.0) < 1e-10

    def test_enumerate_joint_all_positive(self):
        """Test that all joint probabilities are non-negative."""
        model = build_reference_model()
        joint = enumerate_joint(model)

        assert all(p >= -1e-10 for p in joint.values())

    def test_compute_posterior_by_enumeration_dict(self):
        """Test full posterior dict from enumeration."""
        model = build_reference_model()
        post = compute_posterior_by_enumeration_dict(model, ["Rain"], {"WetGrass": 1})

        assert 0 in post and 1 in post
        assert abs(post[0] + post[1] - 1.0) < 1e-10
        assert post[1] == pytest.approx(0.7047692307692307, rel=1e-12)

    def test_enumeration_hand_checkable_values(self):
        """Test at least 5 hand-checkable probabilities from the joint."""
        model = build_reference_model()
        joint = enumerate_joint(model)

        # Verify specific joint probabilities (from factorisation)
        # P(C=0,R=0,S=0,W=0) = 0.5 * 0.8 * 0.5 * 0.99 = 0.198
        p_0000 = joint[(
            ("Cloudy", 0), ("Rain", 0), ("Sprinkler", 0), ("WetGrass", 0)
        )]
        assert p_0000 == pytest.approx(0.198, rel=1e-10)

        # P(C=0,R=0,S=0,W=1) = 0.5 * 0.8 * 0.5 * 0.01 = 0.002
        p_0001 = joint[(
            ("Cloudy", 0), ("Rain", 0), ("Sprinkler", 0), ("WetGrass", 1)
        )]
        assert p_0001 == pytest.approx(0.002, rel=1e-10)

        # P(C=0,R=1,S=0,W=1) = 0.5 * 0.2 * 0.5 * 0.90 = 0.045
        p_0101 = joint[(
            ("Cloudy", 0), ("Rain", 1), ("Sprinkler", 0), ("WetGrass", 1)
        )]
        assert p_0101 == pytest.approx(0.045, rel=1e-10)

        # P(C=1,R=1,S=1,W=1) = 0.5 * 0.8 * 0.1 * 0.99 = 0.0396
        p_1111 = joint[(
            ("Cloudy", 1), ("Rain", 1), ("Sprinkler", 1), ("WetGrass", 1)
        )]
        assert p_1111 == pytest.approx(0.0396, rel=1e-10)

        # P(C=1,R=0,S=1,W=0) = 0.5 * 0.2 * 0.1 * 0.10 = 0.001
        p_1010 = joint[(
            ("Cloudy", 1), ("Rain", 0), ("Sprinkler", 1), ("WetGrass", 0)
        )]
        assert p_1010 == pytest.approx(0.001, rel=1e-10)


class TestBayesRuleRoundTrip:
    """Tests for Bayes rule consistency."""

    def test_bayes_rule_round_trip(self):
        """Test P(R|W) * P(W) = P(W|R) * P(R) for Rain/WetGrass."""
        model = build_reference_model()
        joint = enumerate_joint(model)

        # Compute marginals
        p_rain = {0: 0.0, 1: 0.0}
        p_wetgrass = {0: 0.0, 1: 0.0}
        p_rain_wetgrass = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0}

        for assignment, prob in joint.items():
            d = dict(assignment)
            r = d["Rain"]
            w = d["WetGrass"]
            p_rain[r] += prob
            p_wetgrass[w] += prob
            p_rain_wetgrass[(r, w)] += prob

        # Bayes: P(R=1|W=1) = P(W=1|R=1) * P(R=1) / P(W=1)
        p_r1_given_w1 = p_rain_wetgrass[(1, 1)] / p_wetgrass[1]
        p_w1_given_r1 = p_rain_wetgrass[(1, 1)] / p_rain[1]
        p_r1 = p_rain[1]
        p_w1 = p_wetgrass[1]

        lhs = p_r1_given_w1 * p_w1
        rhs = p_w1_given_r1 * p_r1

        assert abs(lhs - rhs) < 1e-12

    def test_chain_rule_joint_equals_product_cpts(self):
        """Test that joint equals product of CPT entries."""
        model = build_reference_model()
        joint = enumerate_joint(model)

        # For each assignment, compute product of CPTs manually
        for assignment, prob in joint.items():
            d = dict(assignment)
            manual = 1.0

            for cpd in model.get_cpds():
                var = cpd.variable
                var_state = d[var]

                if cpd.evidence:
                    evidence_states = tuple(d[ev] for ev in cpd.evidence)
                    col_idx = 0
                    for i, (_ev, state) in enumerate(
                        zip(cpd.evidence, evidence_states, strict=False)
                    ):
                        col_idx = col_idx * cpd.evidence_card[i] + state
                    manual *= cpd.values[var_state, col_idx]
                else:
                    manual *= cpd.values[var_state, 0]

            assert abs(prob - manual) < 1e-12


class TestEvidenceCollapse:
    """Tests that evidence collapses marginal correctly."""

    def test_evidence_collapses_marginal(self):
        """Test that P(WetGrass=1 | WetGrass=1) = 1."""
        model = build_reference_model()
        posterior = query_posterior(model, ["WetGrass"], {"WetGrass": 1})

        assert posterior[1] == pytest.approx(1.0, abs=1e-10)
        assert posterior[0] == pytest.approx(0.0, abs=1e-10)

    def test_evidence_collapses_marginal_enum(self):
        """Test evidence collapse via enumeration."""
        model = build_reference_model()
        post = compute_posterior_by_enumeration_dict(model, ["WetGrass"], {"WetGrass": 1})

        assert post[1] == pytest.approx(1.0, abs=1e-10)
        assert post[0] == pytest.approx(0.0, abs=1e-10)

    def test_zero_matching_evidence(self):
        """Test handling of evidence with no matching joint probability."""
        model = build_reference_model()
        post = compute_posterior_by_enumeration_dict(model, ["Rain"], {"WetGrass": 99})
        assert post == {0: 0.0, 1: 0.0}


class TestRunInferenceExercises:
    """Tests for the run_inference_exercises function."""

    def test_run_inference_exercises_returns_all_keys(self):
        """Test that all exercise results are returned."""
        results = run_inference_exercises()

        expected_keys = {
            "rain_given_wetgrass",
            "sprinkler_given_wetgrass",
            "cloudy_given_wetgrass",
            "rain_given_wetgrass_sprinkler0",
            "enumeration_rain_given_wetgrass",
            "enumeration_diff",
        }
        assert set(results.keys()) == expected_keys

    def test_run_inference_exercises_values_match_notebook(self):
        """Test that exercise values match notebook exactly."""
        results = run_inference_exercises()

        assert results["rain_given_wetgrass"] == pytest.approx(0.7047692307692307, rel=1e-12)
        assert results["sprinkler_given_wetgrass"] == pytest.approx(0.4278461538461539, rel=1e-12)
        assert results["cloudy_given_wetgrass"] == pytest.approx(0.5746153846153845, rel=1e-12)
        p_r_s0 = results["rain_given_wetgrass_sprinkler0"]
        assert p_r_s0 == pytest.approx(0.9922022048937886, rel=1e-12)
        assert results["enumeration_diff"] < 1e-12


class TestErrorHandling:
    """Tests for error handling in inference."""

    def test_unknown_variable_raises(self):
        """Test that unknown variable raises appropriate error."""
        model = build_reference_model()
        with pytest.raises(ValueError, match="Unknown variable"):
            query_posterior(model, ["UnknownVar"], {"WetGrass": 1})

    def test_invalid_evidence_state(self):
        """Test that invalid evidence state (e.g., 2 for binary) is handled."""
        model = build_reference_model()
        with pytest.raises(ValueError, match="Invalid state"):
            query_posterior(model, ["Rain"], {"WetGrass": 2})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])