# CS F407 Lab, Week 5 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Autoregressive text generation and temperature scaling routines."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from week05_transformers_ar.src.dataset import CharTokenizer


def temperature_distribution(
    logits: torch.Tensor | np.ndarray,
    temperature: float = 1.0,
) -> torch.Tensor | np.ndarray:
    """Apply temperature scaling and softmax to logits."""
    if isinstance(logits, torch.Tensor):
        if temperature <= 1e-6:
            # Deterministic one-hot on argmax
            probs = torch.zeros_like(logits)
            max_idx = torch.argmax(logits, dim=-1, keepdim=True)
            probs.scatter_(-1, max_idx, 1.0)
            return probs
        scaled = logits / temperature
        return F.softmax(scaled, dim=-1)

    # NumPy implementation
    if temperature <= 1e-6:
        probs_np = np.zeros_like(logits)
        max_idx_np = np.argmax(logits, axis=-1, keepdims=True)
        np.put_along_axis(probs_np, max_idx_np, 1.0, axis=-1)
        return probs_np

    scaled_np = logits / temperature
    shifted = scaled_np - np.max(scaled_np, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def generate_tokens(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
    block_size: int = 32,
    temperature: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate tokens autoregressively conditioned on prompt tensor idx of shape (B, T)."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=idx.device).manual_seed(seed)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

            # Forward model to get next-token logits
            logits, _ = model(idx_cond)

            # Pluck logits at the last position
            logits = logits[:, -1, :]  # (B, V)

            if temperature <= 1e-4:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
            else:
                scaled_logits = logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)  # (B, V)
                idx_next = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

    return idx


def generate_text(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str = "",
    max_new_tokens: int = 100,
    block_size: int = 32,
    temperature: float = 1.0,
    seed: int | None = None,
    device: str = "cpu",
) -> str:
    """Generate string continuation from a character prompt."""
    if len(prompt) == 0:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        encoded = tokenizer.encode(prompt)
        idx = torch.tensor([encoded], dtype=torch.long, device=device)

    generated_ids = generate_tokens(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
        seed=seed,
    )
    return tokenizer.decode(generated_ids[0])
