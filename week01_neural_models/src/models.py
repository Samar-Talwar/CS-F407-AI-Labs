# CS F407 Lab, Week 1 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Neural model definitions for XOR experiments (binary and 3-class)."""

from __future__ import annotations

from torch import Tensor, nn


class XORBinaryNet(nn.Module):
    """2-2-1 network for binary XOR with configurable hidden activation."""

    def __init__(self, hidden_activation: str = "sigmoid") -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.activation = _get_activation(hidden_activation)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw logit (pre-sigmoid)."""
        return self.output(self.activation(self.hidden(x)))


class XORMulticlassNet(nn.Module):
    """2-2-3 network for 3-class sensor task."""

    def __init__(self, hidden_activation: str = "sigmoid") -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 3)
        self.activation = _get_activation(hidden_activation)

    def forward(self, x: Tensor) -> Tensor:
        """Return 3 raw logits."""
        return self.output(self.activation(self.hidden(x)))


def _get_activation(name: str) -> nn.Module:
    return {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "relu": nn.ReLU()}[name]
