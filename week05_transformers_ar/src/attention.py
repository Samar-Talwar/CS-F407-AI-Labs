# CS F407 Lab, Week 5 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Attention mechanisms: NumPy reference implementations, single-head, and multi-head attention."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def softmax_numpy(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax along the specified axis."""
    shifted_z = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def cross_entropy_numpy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute cross-entropy loss given predicted probabilities and 1D integer targets."""
    n = len(targets)
    log_probs = np.log(np.maximum(probs[np.arange(n), targets], 1e-12))
    return float(-np.mean(log_probs))


def scaled_dot_product_attention_numpy(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention in NumPy: softmax(Q @ K.T / sqrt(d_k) + M) @ V."""
    d_k = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    weights = softmax_numpy(scores, axis=-1)
    output = np.matmul(weights, v)
    return output, weights


def causal_self_attention_numpy(
    x: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy causal self-attention over sequence x of shape (T, C)."""
    t, _ = x.shape
    q = np.matmul(x, wq)
    k = np.matmul(x, wk)
    v = np.matmul(x, wv)
    causal_mask = np.tril(np.ones((t, t), dtype=np.float64))
    return scaled_dot_product_attention_numpy(q, k, v, mask=causal_mask)


class Head(nn.Module):
    """Single head of causal self-attention."""

    def __init__(self, head_size: int, n_embd: int, block_size: int) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal scaled dot-product attention. Shape: (B, T, C) -> (B, T, head_size)."""
        _, t, _ = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute scaled attention scores ("affinities")
        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v       # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of causal self-attention running in parallel."""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd
        self.heads = nn.ModuleList([
            Head(head_size=head_size, n_embd=n_embd, block_size=block_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all head outputs and project back to n_embd."""
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class CustomMultiHeadAttention(nn.Module):
    """Multi-Head Attention supporting arbitrary (num_heads * head_size != n_embd) dimensions."""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd
        self.heads = nn.ModuleList([
            Head(head_size=head_size, n_embd=n_embd, block_size=block_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate head outputs (B, T, num_heads * head_size) and project to (B, T, n_embd)."""
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
