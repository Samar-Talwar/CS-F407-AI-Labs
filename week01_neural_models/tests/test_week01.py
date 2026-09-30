# CS F407 Lab, Week 1 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Tests for Week 1 neural model experiments.

Every claim the lab sheet makes gets a test.
"""

from __future__ import annotations

import torch
from torch import nn

from week01_neural_models.src.models import XORBinaryNet, XORMulticlassNet
from week01_neural_models.src.train import (
    X_XOR,
    Y_BINARY,
    seed_sweep,
    train_binary_xor,
    train_linear_baseline,
    train_multiclass,
    train_symmetry_experiment,
)

# ── Task 1: Linear baseline fails on XOR ───────────────────────────────────────


class TestLinearBaseline:
    def test_cannot_reach_4_of_4(self) -> None:
        r = train_linear_baseline(seed=42)
        assert r.num_correct < 4, f"Linear baseline should not solve XOR, got {r.num_correct}/4"

    def test_final_loss_near_ln2(self) -> None:
        r = train_linear_baseline(seed=42)
        assert abs(r.final_loss - 0.693) < 0.05, f"Loss {r.final_loss} should be near ln2"


# ── Task 4A: Binary XOR learns all 4 examples ────────────────────────────────


class TestBinaryXOR:
    def test_all_four_correct(self) -> None:
        r = train_binary_xor(seed=42)
        assert r.predictions == [0, 1, 1, 0], f"Expected [0,1,1,0], got {r.predictions}"

    def test_loss_decreases(self) -> None:
        r = train_binary_xor(seed=42)
        assert r.final_loss < r.initial_loss

    def test_final_loss_small(self) -> None:
        r = train_binary_xor(seed=42)
        assert r.final_loss < 0.1, f"Loss {r.final_loss} not < 0.1"

    def test_gradient_nonzero_step0(self) -> None:
        r = train_binary_xor(seed=42)
        assert r.grad_w1_step0_norm > 0.0, "Gradient was zero at step 0"

    def test_gradient_nonzero_step10(self) -> None:
        r = train_binary_xor(seed=42)
        assert r.grad_w1_step10_norm > 0.0, "Gradient was zero at step 10"


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

    def test_train_binary_xor_records_gradient_diff(self) -> None:
        """train_binary_xor returns the max abs diff between mean and per-example gradients."""
        r = train_binary_xor(seed=42)
        assert r.grad_mean_vs_per_example_max_diff < 1e-6, (
            f"Mean vs per-example gradient diff {r.grad_mean_vs_per_example_max_diff} >= 1e-6"
        )


# ── Task 4C: Symmetry ────────────────────────────────────────────────────────


class TestSymmetry:
    def test_zero_init_rows_identical_at_all_steps(self) -> None:
        r = train_symmetry_experiment(seed=42)
        assert all(r.rows_equal_per_step), "Zero-init rows diverged at some step"

    def test_zero_init_fails_to_learn(self) -> None:
        r = train_symmetry_experiment(seed=42)
        assert r.final_loss > 0.2, "Zero-init model should not learn XOR well"

    def test_records_per_step_equality(self) -> None:
        r = train_symmetry_experiment(seed=42)
        # Should have 5 recorded steps: 0, 1, 5, 10, 100
        assert len(r.rows_equal_per_step) == 5
        assert len(r.row0_history) == 5
        assert len(r.row1_history) == 5


# ── Task 4D: Activation comparison ───────────────────────────────────────────


class TestActivationComparison:
    # ReLU needs tuned hyperparams to avoid dead-unit problem on XOR.
    ACT_CONFIGS = [
        ("sigmoid", 42, 1.0, 5000),
        ("tanh", 42, 1.0, 5000),
        ("relu", 5, 0.5, 20000),
    ]

    def test_all_activations_learn(self) -> None:
        for act, seed, lr, steps in self.ACT_CONFIGS:
            r = train_binary_xor(hidden_activation=act, seed=seed, lr=lr, steps=steps)
            assert r.predictions == [0, 1, 1, 0], f"{act} failed: {r.predictions}"

    def test_gradient_norms_positive(self) -> None:
        for act, seed, lr, steps in self.ACT_CONFIGS:
            r = train_binary_xor(hidden_activation=act, seed=seed, lr=lr, steps=steps)
            assert r.grad_w1_step10_norm > 0.0, f"{act} had zero gradient at step 10"


# ── Task 5: Multiclass ───────────────────────────────────────────────────────


class TestMulticlass:
    def test_predictions_correct(self) -> None:
        r = train_multiclass(seed=42)
        assert r.predictions == [0, 1, 1, 2], f"Expected [0,1,1,2], got {r.predictions}"

    def test_softmax_sums_to_one(self) -> None:
        r = train_multiclass(seed=42)
        for i, s in enumerate(r.prob_sums):
            assert abs(s - 1.0) < 1e-5, f"Example {i}: prob sum = {s}"

    def test_shift_invariance(self) -> None:
        r = train_multiclass(seed=42)
        assert r.shift_invariant, "Softmax should be shift-invariant"

    def test_output_weight_shape(self) -> None:
        model = XORMulticlassNet("sigmoid")
        assert model.output.weight.shape == (3, 2), "Final W should be (3, 2)"

    def test_multiclass_records_output_weight_shape(self) -> None:
        r = train_multiclass(seed=42)
        assert r.output_weight_shape == [3, 2]

    def test_multiclass_records_logits_per_example(self) -> None:
        r = train_multiclass(seed=42)
        assert r.logits_per_example == 3

    def test_multiclass_records_hidden_size(self) -> None:
        r = train_multiclass(seed=42)
        assert r.hidden_size == 2

    def test_multiclass_records_shift_constant(self) -> None:
        r = train_multiclass(seed=42)
        assert r.shift_constant == 100.0

    def test_multiclass_records_max_abs_shift_diff(self) -> None:
        r = train_multiclass(seed=42)
        assert r.max_abs_shift_diff < 1e-5

    def test_multiclass_naive_softmax_broken(self) -> None:
        r = train_multiclass(seed=42)
        assert r.naive_softmax_broken, "Naive softmax should break on large logits"

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
    def test_sigmoid_5_seeds(self) -> None:
        r = seed_sweep("sigmoid", [42, 123, 456, 789, 999], lr=1.0, steps=5000)
        assert r.num_seeds_reaching_4_correct >= 1

    def test_tanh_5_seeds(self) -> None:
        r = seed_sweep("tanh", [42, 123, 456, 789, 999], lr=1.0, steps=5000)
        assert r.num_seeds_reaching_4_correct >= 1

    def test_relu_5_seeds(self) -> None:
        r = seed_sweep("relu", [42, 123, 456, 789, 999], lr=0.5, steps=20000)
        assert r.num_seeds_reaching_4_correct >= 1

    def test_sweep_returns_seeds(self) -> None:
        r = seed_sweep("sigmoid", [42, 123], lr=1.0, steps=5000)
        assert len(r.seeds_reaching_4_correct) <= 2


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
            probs = torch.sigmoid(model(X_XOR)).squeeze().tolist()
            preds = [int(p > 0.5) for p in probs]

        # Should not reach 4/4 correct
        assert preds != [0, 1, 1, 0], "Linear model (no hidden nonlinearity) should not solve XOR"