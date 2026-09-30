# CS F407 Lab, Week 1 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Training routines for XOR experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .models import XORBinaryNet, XORMulticlassNet

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
    grad_w1: list[list[float]]
    grad_w1_norm: float


@dataclass
class SymmetryResult:
    row0_history: list[list[float]]
    row1_history: list[list[float]]
    rows_identical: bool
    final_loss: float


@dataclass
class MulticlassResult:
    final_loss: float
    probabilities: list[list[float]]
    predictions: list[int]
    prob_sums: list[float]
    shift_invariant: bool
    shifted_probs: list[list[float]]


# ── Training functions ────────────────────────────────────────────────────────


def train_binary_xor(
    hidden_activation: str = "sigmoid",
    lr: float = 1.0,
    steps: int = 5000,
    seed: int = 42,
    capture_grad_at_step: int = 10,
) -> BinaryResult:
    """Train 2-2-1 binary XOR network, return diagnostics."""
    torch.manual_seed(seed)
    model = XORBinaryNet(hidden_activation)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)

    initial_loss = criterion(model(X_XOR), Y_BINARY).item()
    grad_w1: list[list[float]] = []
    grad_w1_norm = 0.0

    for step in range(steps):
        optimiser.zero_grad()
        logits = model(X_XOR)
        loss = criterion(logits, Y_BINARY)
        loss.backward()

        if step == capture_grad_at_step:
            g = model.hidden.weight.grad
            assert g is not None
            grad_w1 = g.tolist()
            grad_w1_norm = g.norm().item()

        optimiser.step()

    with torch.no_grad():
        logits = model(X_XOR)
        probs = torch.sigmoid(logits).squeeze().tolist()
        preds = [int(p > 0.5) for p in probs]

    return BinaryResult(
        initial_loss=initial_loss,
        final_loss=loss.item(),
        probabilities=probs,
        predictions=preds,
        grad_w1=grad_w1,
        grad_w1_norm=grad_w1_norm,
    )


def train_symmetry_experiment(steps: int = 5000, lr: float = 1.0, seed: int = 42) -> SymmetryResult:
    """Zero-init all weights, train, track hidden-layer weight rows."""
    torch.manual_seed(seed)
    model = XORBinaryNet("sigmoid")

    # Zero all parameters
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)

    record_steps = [0, 10, 100, 500, steps - 1]
    row0_hist: list[list[float]] = []
    row1_hist: list[list[float]] = []

    for step in range(steps):
        if step in record_steps:
            w = model.hidden.weight.data
            row0_hist.append(w[0].tolist())
            row1_hist.append(w[1].tolist())

        optimiser.zero_grad()
        loss = criterion(model(X_XOR), Y_BINARY)
        loss.backward()
        optimiser.step()

    # Check if rows stayed identical throughout
    rows_identical = all(
        torch.allclose(torch.tensor(r0), torch.tensor(r1), atol=1e-7)
        for r0, r1 in zip(row0_hist, row1_hist, strict=True)
    )

    return SymmetryResult(
        row0_history=row0_hist,
        row1_history=row1_hist,
        rows_identical=rows_identical,
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

        # Shift-invariance check: add 100 to all logits
        shifted_probs = torch.softmax(logits + 100.0, dim=1)
        shift_invariant = torch.allclose(probs, shifted_probs, atol=1e-5)

    return MulticlassResult(
        final_loss=loss.item(),
        probabilities=probs.tolist(),
        predictions=preds,
        prob_sums=prob_sums,
        shift_invariant=shift_invariant,
        shifted_probs=shifted_probs.tolist(),
    )
