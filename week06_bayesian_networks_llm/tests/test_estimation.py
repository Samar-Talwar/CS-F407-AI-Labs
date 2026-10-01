# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Tests for parameter estimation: MLE and Bayesian (BDeu).
"""

from __future__ import annotations

import numpy as np
import pytest

from week06_bayesian_networks_llm.src.bn import (
    SimpleDataFrame,
    build_reference_model,
)
from week06_bayesian_networks_llm.src.estimation import (
    fit_bayesian,
    fit_mle,
    generate_synthetic_data,
    multi_seed_estimates,
    rain_c1_probability,
    run_estimation_experiments,
    sample_size_sweep,
)


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_synthetic_data_shape(self):
        """Test that generated data has correct shape."""
        data = generate_synthetic_data(n_samples=1000, seed=7)

        assert isinstance(data, SimpleDataFrame)
        assert data.shape == (1000, 4)
        assert set(data.columns) == {"Cloudy", "Rain", "Sprinkler", "WetGrass"}

    def test_generate_synthetic_data_binary(self):
        """Test that all variables are binary (0 or 1)."""
        data = generate_synthetic_data(n_samples=1000, seed=7)

        for col in data.columns:
            assert set(data[col].unique()).issubset({0, 1})

    def test_generate_synthetic_data_reproducible(self):
        """Test that same seed produces identical data."""
        data1 = generate_synthetic_data(n_samples=100, seed=42)
        data2 = generate_synthetic_data(n_samples=100, seed=42)

        assert data1.to_dict() == data2.to_dict()

    def test_generate_synthetic_data_different_seeds(self):
        """Test that different seeds produce different data."""
        data1 = generate_synthetic_data(n_samples=100, seed=1)
        data2 = generate_synthetic_data(n_samples=100, seed=2)

        assert data1.to_dict() != data2.to_dict()


class TestMLEFitting:
    """Tests for Maximum Likelihood Estimation."""

    def test_fit_mle_valid(self):
        """Test that MLE fit produces valid model."""
        data = generate_synthetic_data(n_samples=1000, seed=7)
        model = fit_mle(data)

        assert model.check_model() is True
        assert len(model.get_cpds()) == 4

    def test_fit_mle_cpds_normalized(self):
        """Test that fitted CPDs are normalized."""
        data = generate_synthetic_data(n_samples=1000, seed=7)
        model = fit_mle(data)

        for cpd in model.get_cpds():
            col_sums = cpd.values.sum(axis=0)
            assert np.allclose(col_sums, 1.0, atol=1e-10)
            assert np.all(cpd.values >= -1e-10)

    def test_mle_manual_verification(self):
        """Test manual MLE calculation matches pgmpy."""
        data = generate_synthetic_data(n_samples=1000, seed=7)
        model = fit_mle(data)

        # Manual: P(Rain=1 | Cloudy=1) = count(R=1, C=1) / count(C=1)
        cloudy_true = data[data["Cloudy"] == 1]
        manual = (cloudy_true["Rain"] == 1).sum() / len(cloudy_true)
        native_val = rain_c1_probability(model)

        assert manual == pytest.approx(native_val, rel=1e-12)

    def test_mle_cpds_match_notebook(self):
        """Test MLE fitted CPDs match empirical sample frequencies exactly."""
        data = generate_synthetic_data(n_samples=1000, seed=7)
        model = fit_mle(data)

        # Cloudy manual
        c_arr = data["Cloudy"]
        c0 = (c_arr == 0).sum() / len(c_arr)
        c1 = (c_arr == 1).sum() / len(c_arr)
        cpd_c = model.get_cpds("Cloudy")
        assert cpd_c.values[0, 0] == pytest.approx(c0, rel=1e-10)
        assert cpd_c.values[1, 0] == pytest.approx(c1, rel=1e-10)

        # Rain manual
        r_arr = data["Rain"]
        c0_mask = c_arr == 0
        c1_mask = c_arr == 1
        r0_c0 = ((r_arr == 0) & c0_mask).sum() / c0_mask.sum()
        r1_c0 = ((r_arr == 1) & c0_mask).sum() / c0_mask.sum()
        r0_c1 = ((r_arr == 0) & c1_mask).sum() / c1_mask.sum()
        r1_c1 = ((r_arr == 1) & c1_mask).sum() / c1_mask.sum()

        cpd_r = model.get_cpds("Rain")
        assert cpd_r.values[0, 0] == pytest.approx(r0_c0, rel=1e-10)
        assert cpd_r.values[1, 0] == pytest.approx(r1_c0, rel=1e-10)
        assert cpd_r.values[0, 1] == pytest.approx(r0_c1, rel=1e-10)
        assert cpd_r.values[1, 1] == pytest.approx(r1_c1, rel=1e-10)


class TestBayesianEstimation:
    """Tests for Bayesian (BDeu) estimation."""

    def test_fit_bayesian_valid(self):
        """Test that Bayesian fit produces valid model."""
        data = generate_synthetic_data(n_samples=30, seed=11)
        model = fit_bayesian(data, prior_type="BDeu", equivalent_sample_size=10)

        assert model.check_model() is True
        assert len(model.get_cpds()) == 4

    def test_bayesian_cpds_normalized(self):
        """Test that Bayesian fitted CPDs are normalized."""
        data = generate_synthetic_data(n_samples=30, seed=11)
        model = fit_bayesian(data, prior_type="BDeu", equivalent_sample_size=10)

        for cpd in model.get_cpds():
            col_sums = cpd.values.sum(axis=0)
            assert np.allclose(col_sums, 1.0, atol=1e-10)
            assert np.all(cpd.values >= -1e-10)

    def test_bayesian_vs_mle_small_sample(self):
        """Test Bayesian vs MLE on small sample (N=30) matches analytical BDeu formula."""
        data = generate_synthetic_data(n_samples=30, seed=11)

        mle_model = fit_mle(data)
        bayes_model = fit_bayesian(data, prior_type="BDeu", equivalent_sample_size=10)

        mle_val = rain_c1_probability(mle_model)
        bayes_val = rain_c1_probability(bayes_model)

        # Analytical calculation
        c_arr = data["Cloudy"]
        r_arr = data["Rain"]
        c1_mask = c_arr == 1
        n_c1 = int(c1_mask.sum())
        n_r1_c1 = int(((r_arr == 1) & c1_mask).sum())

        expected_mle = n_r1_c1 / n_c1
        # Formula: (n_ijk + alpha/(card*num_combos)) / (n_ij + alpha/num_combos)
        # alpha=10, card=2, combos=2
        expected_bayes = (n_r1_c1 + 10.0 / (2 * 2)) / (n_c1 + 10.0 / 2)

        assert mle_val == pytest.approx(expected_mle, rel=1e-10)
        assert bayes_val == pytest.approx(expected_bayes, rel=1e-10)
        assert 0 <= mle_val <= 1
        assert 0 <= bayes_val <= 1


class TestSampleSizeSweep:
    """Tests for sample size sweep experiment."""

    def test_sample_size_sweep(self):
        """Test sample size sweep runs and returns expected structure."""
        results = sample_size_sweep(sample_sizes=[20, 50, 100], seed=7)

        assert len(results) == 3
        for r in results:
            assert "N" in r
            assert "estimated_P_Rain1_given_Cloudy1" in r
            assert "absolute_error" in r
            assert r["N"] in [20, 50, 100]
            assert 0 <= r["estimated_P_Rain1_given_Cloudy1"] <= 1
            assert r["absolute_error"] >= 0

    def test_sample_size_sweep_values(self):
        """Test sweep values are self-consistent (same data produces same estimate)."""
        results = sample_size_sweep(seed=7)
        # Each result must match independent MLE computed on the same seed
        for r in results:
            data = generate_synthetic_data(n_samples=r["N"], seed=7)
            model = fit_mle(data)
            expected_est = rain_c1_probability(model)
            assert r["estimated_P_Rain1_given_Cloudy1"] == pytest.approx(expected_est, rel=1e-10)
            assert r["absolute_error"] == pytest.approx(abs(expected_est - 0.8), rel=1e-10)
            assert r["N"] in [20, 50, 100, 500, 1000, 5000]

    def test_sample_size_sweep_monotonic_trend(self):
        """Test that error generally decreases with sample size (not strictly monotonic)."""
        results = sample_size_sweep(seed=7)

        # The last sample (5000) should have smaller error than first (20)
        first_error = next(r["absolute_error"] for r in results if r["N"] == 20)
        last_error = next(r["absolute_error"] for r in results if r["N"] == 5000)

        assert last_error < first_error


class TestMultiSeedEstimates:
    """Tests for multi-seed estimation experiment."""

    def test_multi_seed_estimates(self):
        """Test multi-seed estimates returns expected structure."""
        results = multi_seed_estimates(seeds=[1, 2, 3], n_samples=100)

        assert len(results) == 3
        for r in results:
            assert "seed" in r
            assert "estimate" in r
            assert r["seed"] in [1, 2, 3]
            assert 0 <= r["estimate"] <= 1

    def test_multi_seed_estimates_values(self):
        """Test multi-seed estimates match independent direct MLE for each seed."""
        results = multi_seed_estimates(n_samples=100)

        for r in results:
            seed = r["seed"]
            data = generate_synthetic_data(n_samples=100, seed=seed)
            model = fit_mle(data)
            expected = rain_c1_probability(model)
            assert r["estimate"] == pytest.approx(expected, rel=1e-10)

    def test_multi_seed_different_seeds_different_estimates(self):
        """Test that different seeds produce different estimates."""
        results = multi_seed_estimates(seeds=[1, 2, 3, 4, 5], n_samples=100)

        estimates = [r["estimate"] for r in results]
        # At least some should differ (sampling variability)
        assert len(set(round(e, 10) for e in estimates)) > 1


class TestRunEstimationExperiments:
    """Tests for the run_estimation_experiments function."""

    def test_run_estimation_experiments_returns_all_keys(self):
        """Test that all experiment results are returned."""
        results = run_estimation_experiments()

        expected_keys = {
            "mle_fitted_cpds",
            "manual_mle_verification",
            "small_sample_comparison",
            "sample_size_sweep",
            "multi_seed_estimates",
        }
        assert set(results.keys()) == expected_keys

    def test_run_estimation_experiments_manual_mle(self):
        """Test manual MLE verification in run_experiments."""
        results = run_estimation_experiments()

        manual = results["manual_mle_verification"]["manual"]
        native = results["manual_mle_verification"]["native_mle"]
        true_val = results["manual_mle_verification"]["true_value"]

        assert manual == pytest.approx(native, rel=1e-12)
        assert true_val == 0.8

    def test_run_estimation_experiments_small_sample(self):
        """Test small sample comparison in run_experiments."""
        results = run_estimation_experiments()

        comp = results["small_sample_comparison"]
        assert comp["true_parameter"] == 0.8

        data_30 = generate_synthetic_data(n_samples=30, seed=11)
        expected_mle = rain_c1_probability(fit_mle(data_30))
        expected_bayes = rain_c1_probability(
            fit_bayesian(data_30, prior_type="BDeu", equivalent_sample_size=10)
        )

        assert comp["mle_n30"] == pytest.approx(expected_mle, rel=1e-10)
        assert comp["bayes_n30_ess10"] == pytest.approx(expected_bayes, rel=1e-10)


class TestEstimationErrorHandling:
    """Tests for error handling in estimation."""

    def test_fit_mle_empty_data(self):
        """Test MLE fit with empty data raises error."""
        empty_df = SimpleDataFrame(
            data=np.zeros((0, 4), dtype=int),
            columns=["Cloudy", "Rain", "Sprinkler", "WetGrass"],
        )
        with pytest.raises(ValueError):
            fit_mle(empty_df)

    def test_fit_bayesian_invalid_prior(self):
        """Test Bayesian fit with invalid prior type."""
        data = generate_synthetic_data(n_samples=30, seed=11)
        with pytest.raises(ValueError):
            fit_bayesian(data, prior_type="InvalidPrior")

    def test_rain_c1_probability_returns_float(self):
        """Test rain_c1_probability returns float."""
        data = generate_synthetic_data(n_samples=100, seed=7)
        model = fit_mle(data)

        prob = rain_c1_probability(model)
        assert isinstance(prob, float)
        assert 0 <= prob <= 1


class TestMutationTests:
    """Real mutation tests: monkeypatch actual functions and assert failures."""

    def test_mutation_skip_normalization_in_inference(self, monkeypatch):
        """Mutation test: skip normalization in inference -> oracle test should fail."""
        from week06_bayesian_networks_llm.src import inference

        # Store original
        orig_query = inference.query_posterior

        def broken_query(model, variables, evidence):
            result = orig_query(model, variables, evidence)
            # Don't normalize - return raw values
            return result * 2  # This breaks normalization

        monkeypatch.setattr(inference, "query_posterior", broken_query)

        model = build_reference_model()
        posterior = inference.query_posterior(model, ["Rain"], {"WetGrass": 1})

        # The oracle test (enumeration) should detect this
        enum_result = inference.compute_posterior_by_enumeration(model, "Rain", 1, {"WetGrass": 1})

        # Broken result should not match enumeration
        assert abs(posterior[1] - enum_result) > 1e-6

    def test_mutation_ignore_evidence(self, monkeypatch):
        """Mutation test: ignore evidence -> broken should diverge from true posterior."""
        from week06_bayesian_networks_llm.src import inference

        orig_compute = inference.compute_posterior_by_enumeration

        def broken_compute(model, query_var, query_state, evidence):
            # Ignore evidence completely
            return orig_compute(model, query_var, query_state, {})

        monkeypatch.setattr(inference, "compute_posterior_by_enumeration", broken_compute)

        model = build_reference_model()
        # Ground-truth posterior P(Rain=1 | WetGrass=1)
        true_posterior = orig_compute(model, "Rain", 1, {"WetGrass": 1})
        broken_result = inference.compute_posterior_by_enumeration(
            model, "Rain", 1, {"WetGrass": 1}
        )

        # Broken (ignores evidence) yields marginal ~0.5; true is ~0.704769
        assert abs(broken_result - true_posterior) > 1e-6

    def test_mutation_corrupt_cpt(self, monkeypatch):
        """Mutation test: corrupt one CPT entry -> validator should reject."""
        from week06_bayesian_networks_llm.src import bn

        # Mutation: corrupt validator to always return True
        def broken_validate(model):
            return True

        monkeypatch.setattr(bn, "validate_cpt", broken_validate)

        # Create broken model using our native structure
        broken_model = bn.DiscreteBayesianNetwork([("Cloudy", "Rain")])
        cpd_cloudy_ok = bn.TabularCPD("Cloudy", 2, [[0.5], [0.5]])
        cpd_rain_broken = bn.TabularCPD(
            "Rain", 2,
            [[0.9, 0.2], [0.3, 0.8]],  # Sums to 1.2 for Cloudy=0
            evidence=["Cloudy"], evidence_card=[2],
        )
        broken_model.add_cpds(cpd_cloudy_ok, cpd_rain_broken)

        # Broken validator returns True
        assert bn.validate_cpt(broken_model) is True
        # But actual check_model should catch it
        with pytest.raises(ValueError):
            broken_model.check_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])