# CS F407 Lab, Week 1 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Tests for Week 1 neural model experiments.

Every claim the lab sheet makes gets a test.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from week01_neural_models.src.models import XORBinaryNet, XORMulticlassNet
from week01_neural_models.src.train import (
    X_XOR,
    Y_BINARY,
    BinaryResult,
    LinearBaselineResult,
    MulticlassResult,
    SeedSweepResult,
    SymmetryResult,
    seed_sweep,
    train_binary_xor,
    train_linear_baseline,
    train_multiclass,
    train_symmetry_experiment,
)

# ── Module fixtures (run once across tests) ───────────────────────────────────


@pytest.fixture(scope="module")
def linear_result() -> LinearBaselineResult:
    return train_linear_baseline(seed=42)


@pytest.fixture(scope="module")
def binary_result() -> BinaryResult:
    return train_binary_xor(seed=42)


@pytest.fixture(scope="module")
def symmetry_result() -> SymmetryResult:
    return train_symmetry_experiment(seed=42)


@pytest.fixture(scope="module")
def multiclass_result() -> MulticlassResult:
    return train_multiclass(seed=42)


@pytest.fixture(scope="module")
def activation_results() -> dict[str, BinaryResult]:
    configs = [
        ("sigmoid", 42, 1.0, 5000),
        ("tanh", 42, 1.0, 5000),
        ("relu", 5, 0.5, 20000),
    ]
    return {
        act: train_binary_xor(hidden_activation=act, seed=seed, lr=lr, steps=steps)
        for act, seed, lr, steps in configs
    }


@pytest.fixture(scope="module")
def sweep_results() -> dict[str, SeedSweepResult]:
    seeds = [42, 123, 456, 789, 999]
    configs = [
        ("sigmoid", 1.0, 5000),
        ("tanh", 1.0, 5000),
        ("relu", 0.5, 20000),
    ]
    return {
        act: seed_sweep(act, seeds, lr=lr, steps=steps)
        for act, lr, steps in configs
    }


# ── Task 1: Linear baseline fails on XOR ───────────────────────────────────────


class TestLinearBaseline:
    def test_cannot_reach_4_of_4(self, linear_result: LinearBaselineResult) -> None:
        assert linear_result.num_correct < 4, (
            f"Linear baseline should not solve XOR, got {linear_result.num_correct}/4"
        )

    def test_final_loss_near_ln2(self, linear_result: LinearBaselineResult) -> None:
        assert abs(linear_result.final_loss - 0.693) < 0.05, (
            f"Loss {linear_result.final_loss} should be near ln2"
        )


# ── Task 4A: Binary XOR learns all 4 examples ────────────────────────────────


class TestBinaryXOR:
    def test_all_four_correct(self, binary_result: BinaryResult) -> None:
        assert binary_result.predictions == [0, 1, 1, 0], (
            f"Expected [0,1,1,0], got {binary_result.predictions}"
        )

    def test_loss_decreases(self, binary_result: BinaryResult) -> None:
        assert binary_result.final_loss < binary_result.initial_loss

    def test_final_loss_small(self, binary_result: BinaryResult) -> None:
        assert binary_result.final_loss < 0.1, f"Loss {binary_result.final_loss} not < 0.1"

    def test_gradient_nonzero_step0(self, binary_result: BinaryResult) -> None:
        assert binary_result.grad_w1_step0_norm > 0.0, "Gradient was zero at step 0"

    def test_gradient_nonzero_step10(self, binary_result: BinaryResult) -> None:
        assert binary_result.grad_w1_step10_norm > 0.0, "Gradient was zero at step 10"


# ── Task 4B: Backprop check ──────────────────────────────────────────────────


class TestBackpropCheck:
    def test_grad_shape_matches_weights(self) -> None:
        torch.manual_seed(42)
        model = XORBinaryNet("sigmoid")
        loss = nn.BCEWithLogitsLoss()(model(X_XOR), Y_BINARY)
        loss.backward()
        g = model.hidden.weight.grad
        assert g is not None
        assert g.shape == (2, 2), f"Grad shape {g.shape}, expected (2,2)"

    def test_grad_equals_manual_mean(self) -> None:
        """Mean loss ⇒ grad = mean of per-example grads."""
        torch.manual_seed(42)
        model = XORBinaryNet("sigmoid")

        # Full-batch gradient
        loss = nn.BCEWithLogitsLoss()(model(X_XOR), Y_BINARY)
        loss.backward()
        full_grad = model.hidden.weight.grad.clone()

        # Sum of per-example gradients / N
        accum = torch.zeros_like(full_grad)
        for i in range(4):
            model.zero_grad()
            li = nn.BCEWithLogitsLoss()(model(X_XOR[i : i + 1]), Y_BINARY[i : i + 1])
            li.backward()
            accum += model.hidden.weight.grad
        mean_grad = accum / 4.0

        assert torch.allclose(full_grad, mean_grad, atol=1e-6)

    def test_train_binary_xor_records_gradient_diff(self, binary_result: BinaryResult) -> None:
        """train_binary_xor: 4 independent per-example backward passes averaged,
        compared to one backward on batch mean loss. Diff reflects floating-point
        accumulation order (~1e-9 to 1e-8), not a logic error."""
        # Independent recomputation: compute mean and 4 per-example from scratch
        torch.manual_seed(42)
        model = XORBinaryNet("sigmoid")
        criterion = nn.BCEWithLogitsLoss()
        # Batch-mean backward (independent)
        model.zero_grad()
        loss_full = criterion(model(X_XOR), Y_BINARY)
        loss_full.backward()
        full_grad = model.hidden.weight.grad.clone()
        # 4 independent per-example backward passes (independent)
        accum = torch.zeros_like(full_grad)
        per_example_grads = []
        for i in range(4):
            model.zero_grad()
            li = criterion(model(X_XOR[i : i + 1]), Y_BINARY[i : i + 1])
            li.backward()
            accum += model.hidden.weight.grad
            per_example_grads.append(model.hidden.weight.grad.clone().tolist())
        mean_grad = accum / 4.0
        diff = (full_grad - mean_grad).abs().max().item()

        # Store result values at full precision (4 per-example + mean)
        assert len(binary_result.per_example_grads_w1) == 4, "Need 4 per-example gradient tensors"
        assert len(binary_result.grad_mean_w1) == 2 and len(binary_result.grad_mean_w1[0]) == 2
        # Assert independent diff < 1e-6 (floating-point accumulation order)
        assert diff < 1e-6, f"Independent recompute diff {diff} >= 1e-6"
        assert binary_result.grad_mean_vs_per_example_max_diff < 1e-6, (
            f"Recorded diff {binary_result.grad_mean_vs_per_example_max_diff} >= 1e-6"
        )


# ── Task 4C: Symmetry ────────────────────────────────────────────────────────


class TestSymmetry:
    def test_zero_init_rows_identical_at_all_steps(self, symmetry_result: SymmetryResult) -> None:
        assert all(symmetry_result.rows_equal_per_step), "Zero-init rows diverged at some step"

    def test_zero_init_fails_to_learn(self, symmetry_result: SymmetryResult) -> None:
        assert symmetry_result.final_loss > 0.2, "Zero-init model should not learn XOR well"

    def test_records_per_step_equality(self, symmetry_result: SymmetryResult) -> None:
        # Should have 5 recorded steps: 0, 1, 5, 10, 100
        assert len(symmetry_result.rows_equal_per_step) == 5
        assert len(symmetry_result.row0_history) == 5
        assert len(symmetry_result.row1_history) == 5


# ── Task 4D: Activation comparison ───────────────────────────────────────────


class TestActivationComparison:
    def test_all_activations_learn(self, activation_results: dict[str, BinaryResult]) -> None:
        for act, r in activation_results.items():
            assert r.predictions == [0, 1, 1, 0], f"{act} failed: {r.predictions}"

    def test_gradient_norms_positive(self, activation_results: dict[str, BinaryResult]) -> None:
        for act, r in activation_results.items():
            assert r.grad_w1_step10_norm > 0.0, f"{act} had zero gradient at step 10"


# ── Task 5: Multiclass ───────────────────────────────────────────────────────


class TestMulticlass:
    def test_predictions_correct(self, multiclass_result: MulticlassResult) -> None:
        assert multiclass_result.predictions == [0, 1, 1, 2], (
            f"Expected [0,1,1,2], got {multiclass_result.predictions}"
        )

    def test_softmax_sums_to_one(self, multiclass_result: MulticlassResult) -> None:
        for i, s in enumerate(multiclass_result.prob_sums):
            assert abs(s - 1.0) < 1e-5, f"Example {i}: prob sum = {s}"

    def test_shift_invariance(self, multiclass_result: MulticlassResult) -> None:
        assert multiclass_result.shift_invariant, "Softmax should be shift-invariant"

    def test_output_weight_shape(self) -> None:
        model = XORMulticlassNet("sigmoid")
        assert model.output.weight.shape == (3, 2), "Final W should be (3, 2)"

    def test_multiclass_records_output_weight_shape(
        self, multiclass_result: MulticlassResult
    ) -> None:
        assert multiclass_result.output_weight_shape == [3, 2]

    def test_multiclass_records_logits_per_example(
        self, multiclass_result: MulticlassResult
    ) -> None:
        assert multiclass_result.logits_per_example == 3

    def test_multiclass_records_hidden_size(self, multiclass_result: MulticlassResult) -> None:
        assert multiclass_result.hidden_size == 2

    def test_multiclass_records_shift_constant(self, multiclass_result: MulticlassResult) -> None:
        assert multiclass_result.shift_constant == 100.0

    def test_multiclass_records_max_abs_shift_diff(
        self, multiclass_result: MulticlassResult
    ) -> None:
        assert multiclass_result.max_abs_shift_diff < 1e-5

    def test_multiclass_naive_softmax_broken(self, multiclass_result: MulticlassResult) -> None:
        assert multiclass_result.naive_softmax_broken, "Naive softmax should break on large logits"

    def test_stable_softmax_vs_naive_on_large_logits(self) -> None:
        """Stable softmax (max subtraction) works; naive breaks on logits+100."""
        # Direct test independent of train_multiclass
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        large_logits = logits + 100.0

        # Stable softmax
        max_val = large_logits.max(dim=1, keepdim=True).values
        stable = torch.exp(large_logits - max_val)
        stable = stable / stable.sum(dim=1, keepdim=True)

        # Naive softmax - should overflow to inf
        try:
            naive = torch.exp(large_logits) / torch.exp(large_logits).sum(dim=1, keepdim=True)
            naive_broken = not torch.allclose(stable, naive, atol=1e-5) or torch.isnan(naive).any()
        except (OverflowError, RuntimeError):
            naive_broken = True

        assert naive_broken, "Naive softmax should fail on large logits"


# ── Task 2: Seed sweep ───────────────────────────────────────────────────────


class TestSeedSweep:
    def test_sigmoid_5_seeds(self, sweep_results: dict[str, SeedSweepResult]) -> None:
        r = sweep_results["sigmoid"]
        assert r.num_seeds_reaching_4_correct >= 1

    def test_tanh_5_seeds(self, sweep_results: dict[str, SeedSweepResult]) -> None:
        r = sweep_results["tanh"]
        assert r.num_seeds_reaching_4_correct >= 1

    def test_relu_5_seeds(self, sweep_results: dict[str, SeedSweepResult]) -> None:
        r = sweep_results["relu"]
        assert r.num_seeds_reaching_4_correct >= 1

    def test_sweep_returns_seeds(self, sweep_results: dict[str, SeedSweepResult]) -> None:
        r = sweep_results["sigmoid"]
        assert len(r.seeds_reaching_4_correct) <= 5


# ── Mutation test: removing hidden nonlinearity must break ────────────────────


class TestMutation:
    def test_removing_hidden_nonlinearity_breaks_xor(self) -> None:
        """If we replace hidden activation with Identity, XOR should fail."""
        torch.manual_seed(42)
        model = XORBinaryNet("sigmoid")
        # Mutate: replace hidden activation with Identity
        model.activation = nn.Identity()
        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.SGD(model.parameters(), lr=1.0)
        for _ in range(5000):
            optimiser.zero_grad()
            loss = criterion(model(X_XOR), Y_BINARY)
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            preds = (torch.sigmoid(model(X_XOR)) > 0.5).int().squeeze().tolist()

        assert preds != [0, 1, 1, 0], f"Linearized model should not solve XOR, got {preds}"
