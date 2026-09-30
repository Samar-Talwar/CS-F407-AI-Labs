# CS F407 Lab, Week 1 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Training routines for XOR experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .models import XORBinaryNet, XORLinearNet, XORMulticlassNet

# ── Data ──────────────────────────────────────────────────────────────────────

X_XOR = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_BINARY = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
Y_MULTICLASS = torch.tensor([0, 1, 1, 2])  # both-off / disagree / both-on


# ── Result containers ─────────────────────────────────────────────────────────


@dataclass
class BinaryResult:
    initial_loss: float
    final_loss: float
    probabilities: list[float]
    predictions: list[int]
    grad_w1_step0: list[list[float]]
    grad_w1_step0_norm: float
    grad_w1_step10: list[list[float]]
    grad_w1_step10_norm: float
    grad_mean_vs_per_example_max_diff: float


@dataclass
class LinearBaselineResult:
    initial_loss: float
    final_loss: float
    probabilities: list[float]
    predictions: list[int]
    num_correct: int


@dataclass
class SymmetryResult:
    row0_history: list[list[float]]
    row1_history: list[list[float]]
    rows_equal_per_step: list[bool]
    final_loss: float


@dataclass
class MulticlassResult:
    final_loss: float
    probabilities: list[list[float]]
    predictions: list[int]
    prob_sums: list[float]
    shift_invariant: bool
    shifted_probs: list[list[float]]
    output_weight_shape: list[int]
    logits_per_example: int
    hidden_size: int
    shift_constant: float
    max_abs_shift_diff: float
    naive_softmax_broken: bool


@dataclass
class SeedSweepResult:
    activation: str
    seeds_tested: list[int]
    num_seeds_reaching_4_correct: int
    seeds_reaching_4_correct: list[int]


# ── Training functions ────────────────────────────────────────────────────────


def train_binary_xor(
    hidden_activation: str = "sigmoid",
    lr: float = 1.0,
    steps: int = 5000,
    seed: int = 42,
) -> BinaryResult:
    """Train 2-2-1 binary XOR network, return diagnostics including step 0 and step 10 gradients."""
    torch.manual_seed(seed)
    model = XORBinaryNet(hidden_activation)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)

    initial_loss = criterion(model(X_XOR), Y_BINARY).item()

    # Capture gradient at step 0 (before any update)
    model.zero_grad()
    loss0 = criterion(model(X_XOR), Y_BINARY)
    loss0.backward()
    g0 = model.hidden.weight.grad
    assert g0 is not None
    grad_w1_step0 = g0.tolist()
    grad_w1_step0_norm = g0.norm().item()

    # Training loop
    for step in range(steps):
        optimiser.zero_grad()
        logits = model(X_XOR)
        loss = criterion(logits, Y_BINARY)
        loss.backward()

        if step == 10:
            g10 = model.hidden.weight.grad
            assert g10 is not None
            grad_w1_step10 = g10.tolist()
            grad_w1_step10_norm = g10.norm().item()

        optimiser.step()

    # Compute mean-loss gradient vs per-example gradient difference
    model.zero_grad()
    loss_full = criterion(model(X_XOR), Y_BINARY)
    loss_full.backward()
    grad_mean = model.hidden.weight.grad.clone()

    accum = torch.zeros_like(grad_mean)
    for i in range(4):
        model.zero_grad()
        li = criterion(model(X_XOR[i:i+1]), Y_BINARY[i:i+1])
        li.backward()
        accum += model.hidden.weight.grad
    grad_per_example_mean = accum / 4.0

    grad_diff = (grad_mean - grad_per_example_mean).abs().max().item()

    with torch.no_grad():
        logits = model(X_XOR)
        probs = torch.sigmoid(logits).squeeze().tolist()
        preds = [int(p > 0.5) for p in probs]

    return BinaryResult(
        initial_loss=initial_loss,
        final_loss=loss.item(),
        probabilities=probs,
        predictions=preds,
        grad_w1_step0=grad_w1_step0,
        grad_w1_step0_norm=grad_w1_step0_norm,
        grad_w1_step10=grad_w1_step10,
        grad_w1_step10_norm=grad_w1_step10_norm,
        grad_mean_vs_per_example_max_diff=grad_diff,
    )


def train_linear_baseline(
    lr: float = 1.0, steps: int = 5000, seed: int = 42
) -> LinearBaselineResult:
    """Train single affine layer + sigmoid on XOR (Task 1 linear baseline)."""
    torch.manual_seed(seed)
    model = XORLinearNet()
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)

    initial_loss = criterion(model(X_XOR), Y_BINARY).item()

    for _ in range(steps):
        optimiser.zero_grad()
        loss = criterion(model(X_XOR), Y_BINARY)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(X_XOR)).squeeze().tolist()
        preds = [int(p > 0.5) for p in probs]
        num_correct = sum(
            1 for p, t in zip(preds, Y_BINARY.squeeze().tolist(), strict=True) if p == t
        )

    return LinearBaselineResult(
        initial_loss=initial_loss,
        final_loss=loss.item(),
        probabilities=probs,
        predictions=preds,
        num_correct=num_correct,
    )


def train_symmetry_experiment(steps: int = 5000, lr: float = 1.0, seed: int = 42) -> SymmetryResult:
    """Zero-init all weights, train, track hidden-layer weight rows at specific steps."""
    torch.manual_seed(seed)
    model = XORBinaryNet("sigmoid")

    # Zero all parameters
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)

    record_steps = [0, 1, 5, 10, 100]
    row0_hist: list[list[float]] = []
    row1_hist: list[list[float]] = []
    rows_equal: list[bool] = []

    for step in range(steps):
        if step in record_steps:
            w = model.hidden.weight.data
            row0_hist.append(w[0].tolist())
            row1_hist.append(w[1].tolist())
            rows_equal.append(torch.allclose(w[0], w[1], atol=1e-7))

        optimiser.zero_grad()
        loss = criterion(model(X_XOR), Y_BINARY)
        loss.backward()
        optimiser.step()

    return SymmetryResult(
        row0_history=row0_hist,
        row1_history=row1_hist,
        rows_equal_per_step=rows_equal,
        final_loss=loss.item(),
    )


def train_multiclass(
    hidden_activation: str = "sigmoid",
    lr: float = 1.0,
    steps: int = 5000,
    seed: int = 42,
) -> MulticlassResult:
    """Train 2-2-3 network for the 3-class sensor task."""
    torch.manual_seed(seed)
    model = XORMulticlassNet(hidden_activation)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(steps):
        optimiser.zero_grad()
        loss = criterion(model(X_XOR), Y_MULTICLASS)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        logits = model(X_XOR)
        probs = torch.softmax(logits, dim=1)
        prob_sums = probs.sum(dim=1).tolist()
        preds = probs.argmax(dim=1).tolist()

        shift_constant = 100.0
        shifted_probs = torch.softmax(logits + shift_constant, dim=1)
        shift_invariant = torch.allclose(probs, shifted_probs, atol=1e-5)
        max_abs_shift_diff = (probs - shifted_probs).abs().max().item()

        # Test naive softmax (without max subtraction) on large logits
        # This would overflow, but we test the diff between stable and naive on
        # the same shifted logits
        # Stable: softmax(logits + 100) - subtract max
        # Naive: exp(logits + 100) / sum(exp(logits + 100)) - without max sub
        max_logit = (logits + shift_constant).max(dim=1, keepdim=True).values
        stable_probs = torch.exp(logits + shift_constant - max_logit)
        stable_probs = stable_probs / stable_probs.sum(dim=1, keepdim=True)

        naive_logits = logits + shift_constant
        # Naive softmax with very large values will overflow to inf
        try:
            naive_probs = torch.exp(naive_logits) / torch.exp(naive_logits).sum(
                dim=1, keepdim=True
            )
            naive_softmax_broken = (
                not torch.allclose(stable_probs, naive_probs, atol=1e-5)
                or torch.isnan(naive_probs).any()
            )
        except (OverflowError, RuntimeError):
            naive_softmax_broken = True

    return MulticlassResult(
        final_loss=loss.item(),
        probabilities=probs.tolist(),
        predictions=preds,
        prob_sums=prob_sums,
        shift_invariant=shift_invariant,
        shifted_probs=shifted_probs.tolist(),
        output_weight_shape=list(model.output.weight.shape),
        logits_per_example=logits.shape[1],
        hidden_size=model.hidden.out_features,
        shift_constant=shift_constant,
        max_abs_shift_diff=max_abs_shift_diff,
        naive_softmax_broken=naive_softmax_broken,
    )


def seed_sweep(
    hidden_activation: str,
    seeds: list[int],
    lr: float = 1.0,
    steps: int = 5000,
) -> SeedSweepResult:
    """Run multiple seeds for an activation, count how many reach 4/4 correct."""
    successful_seeds: list[int] = []
    for seed in seeds:
        r = train_binary_xor(hidden_activation=hidden_activation, lr=lr, steps=steps, seed=seed)
        if r.predictions == [0, 1, 1, 0]:
            successful_seeds.append(seed)

    return SeedSweepResult(
        activation=hidden_activation,
        seeds_tested=seeds,
        num_seeds_reaching_4_correct=len(successful_seeds),
        seeds_reaching_4_correct=successful_seeds,
    )