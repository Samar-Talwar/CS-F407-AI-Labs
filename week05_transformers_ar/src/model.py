# CS F407 Lab, Week 5 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Transformer components and full TinyGPT model definition."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from week05_transformers_ar.src.attention import MultiHeadAttention


class TinyOneTokenLM(nn.Module):
    """Single-token neural language model baseline (token embedding + linear LM head)."""

    def __init__(self, vocab_size: int = 25, n_embd: int = 48) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits and optional cross-entropy loss."""
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        logits = self.lm_head(tok_emb)             # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), targets.view(b * t))

        return logits, loss


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with 4x hidden expansion and ReLU activation."""

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise MLP."""
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer decoder block with Pre-LayerNorm residual connections."""

    def __init__(self, n_heads: int, head_size: int, n_embd: int, block_size: int) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(
            num_heads=n_heads,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
        )
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Pre-LN self-attention and Pre-LN FFN residual updates."""
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Decoder-only autoregressive Transformer Language Model."""

    def __init__(
        self,
        vocab_size: int = 25,
        n_embd: int = 48,
        block_size: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_heads = n_heads
        self.n_layers = n_layers

        head_size = n_embd // n_heads
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_heads=n_heads,
                    head_size=head_size,
                    n_embd=n_embd,
                    block_size=block_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute next-token logits and cross-entropy loss."""
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(
                f"Input sequence length {t} exceeds maximum block_size {self.block_size}."
            )

        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(t, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                      # (B, T, n_embd)
        x = self.blocks(x)                         # (B, T, n_embd)
        x = self.ln_f(x)                           # (B, T, n_embd)
        logits = self.lm_head(x)                   # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(b * t, self.vocab_size), targets.view(b * t))

        return logits, loss

    def count_parameters(self) -> int:
        """Return total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict[str, Any]:
        """Return detailed parameter counts per layer and analytical component."""
        breakdown: dict[str, Any] = {}
        for name, param in self.named_parameters():
            breakdown[name] = {
                "shape": list(param.shape),
                "numel": param.numel(),
                "requires_grad": param.requires_grad,
            }
        return breakdown
