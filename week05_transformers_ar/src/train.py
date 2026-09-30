# CS F407 Lab, Week 5 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Training pipeline and loss evaluation routines for TinyGPT."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from week05_transformers_ar.src.dataset import CANONICAL_CORPUS, CharTokenizer, get_batch
from week05_transformers_ar.src.model import TinyGPT


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data: torch.Tensor | str = CANONICAL_CORPUS,
    eval_iters: int = 50,
    block_size: int = 32,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 42,
) -> float:
    """Estimate average cross-entropy loss over multiple evaluation batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xb, yb = get_batch(
            data=data,
            block_size=block_size,
            batch_size=batch_size,
            device=device,
            seed=seed + k,
        )
        _, loss = model(xb, yb)
        assert loss is not None
        losses[k] = loss.item()
    model.train()
    return float(losses.mean())


def train_tiny_gpt(
    model: TinyGPT | None = None,
    text: str = CANONICAL_CORPUS,
    max_iters: int = 180,
    eval_interval: int = 20,
    eval_iters: int = 20,
    learning_rate: float = 3e-3,
    batch_size: int = 32,
    block_size: int = 32,
    device: str = "cpu",
    seed: int = 1337,
) -> dict[str, Any]:
    """Train TinyGPT using AdamW optimizer and return training history and metrics."""
    torch.manual_seed(seed)
    tokenizer = CharTokenizer(text)
    data_tensor = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    if model is None:
        model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            n_embd=48,
            block_size=block_size,
            n_heads=4,
            n_layers=2,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    step_losses: list[float] = []
    eval_losses: dict[str, float] = {}

    initial_loss = evaluate_loss(
        model,
        data=data_tensor,
        eval_iters=eval_iters,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    eval_losses["0"] = initial_loss

    for step in range(max_iters):
        xb, yb = get_batch(
            data=data_tensor,
            block_size=block_size,
            batch_size=batch_size,
            device=device,
            seed=seed + step,
        )

        _, loss = model(xb, yb)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        step_losses.append(loss_val)

        if (step + 1) % eval_interval == 0 or step == max_iters - 1:
            eval_l = evaluate_loss(
                model,
                data=data_tensor,
                eval_iters=eval_iters,
                block_size=block_size,
                batch_size=batch_size,
                device=device,
                seed=seed + 1000 + step,
            )
            eval_losses[str(step + 1)] = eval_l

    final_loss = eval_losses[str(max_iters)]

    first_10_avg = float(sum(step_losses[:10]) / min(10, len(step_losses)))
    last_10_avg = float(sum(step_losses[-10:]) / min(10, len(step_losses)))

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "step_losses": step_losses,
        "eval_losses": eval_losses,
        "first_10_avg_loss": first_10_avg,
        "last_10_avg_loss": last_10_avg,
        "total_steps": max_iters,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "block_size": block_size,
        "model": model,
    }
