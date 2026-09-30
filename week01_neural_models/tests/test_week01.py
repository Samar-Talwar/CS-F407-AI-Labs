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
    train_binary_xor,
    train_multiclass,
    train_symmetry_experiment,
)

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

    def test_gradient_nonzero(self) -> None:
        r = train_binary_xor(seed=42)
        assert r.grad_w1_norm > 0.0, "Gradient was zero at early step"


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


# ── Task 4C: Symmetry ────────────────────────────────────────────────────────


class TestSymmetry:
    def test_zero_init_rows_identical(self) -> None:
        r = train_symmetry_experiment(seed=42)
        assert r.rows_identical, "Zero-init rows diverged unexpectedly"

    def test_zero_init_fails_to_learn(self) -> None:
        r = train_symmetry_experiment(seed=42)
        # With identical hidden units, the network cannot represent XOR.
        # Loss stays high (≥ ln2 ≈ 0.693 for random guessing).
        assert r.final_loss > 0.2, "Zero-init model should not learn XOR well"


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
            assert r.grad_w1_norm > 0.0, f"{act} had zero gradient"


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

    def test_output_shape(self) -> None:
        model = XORMulticlassNet("sigmoid")
        logits = model(X_XOR)
        assert logits.shape == (4, 3), f"Expected (4,3), got {logits.shape}"

    def test_final_weight_shape(self) -> None:
        model = XORMulticlassNet("sigmoid")
        assert model.output.weight.shape == (3, 2), "Final W should be (3, 2)"
