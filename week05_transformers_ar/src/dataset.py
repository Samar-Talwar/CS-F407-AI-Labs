# CS F407 Lab, Week 5 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Character-level dataset, tokenizer, bigram baseline, and batch generation."""

from __future__ import annotations

import numpy as np
import torch

RAW_CORPUS = """
to learn is to change.
a model learns from examples.
attention lets a token look at earlier tokens.
a transformer predicts the next token.
we train by reducing prediction error.
generation repeats next token prediction.
"""


def build_corpus_string(repeat: int = 80) -> str:
    """Reproduce the canonical 6-sentence corpus repeated `repeat` times."""
    return RAW_CORPUS * repeat


CANONICAL_CORPUS = build_corpus_string(80)


class CharTokenizer:
    """Character-level tokenizer mapping characters to integer IDs and back."""

    def __init__(self, text: str = CANONICAL_CORPUS) -> None:
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str) -> list[int]:
        """Encode string into integer IDs."""
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Decode integer IDs back into string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[int(i)] for i in ids)


class BigramModel:
    """Character-level bigram counting baseline: P(x_t | x_{t-1})."""

    def __init__(self, text: str = CANONICAL_CORPUS, smoothing: float = 0.1) -> None:
        self.tokenizer = CharTokenizer(text)
        self.vocab = self.tokenizer.chars
        self.V = self.tokenizer.vocab_size
        self.stoi = self.tokenizer.stoi
        self.itos = self.tokenizer.itos
        self.smoothing = smoothing

        # Count bigram transitions
        self.counts = np.zeros((self.V, self.V), dtype=np.int64)
        for a, b in zip(text[:-1], text[1:], strict=False):
            self.counts[self.stoi[a], self.stoi[b]] += 1

        # Compute smoothed conditional probability table
        smoothed = self.counts.astype(np.float64) + self.smoothing
        self.probs = smoothed / smoothed.sum(axis=1, keepdims=True)

    def get_cpt_dict(self) -> dict[str, dict[str, float]]:
        """Return transition probabilities as a nested dict."""
        cpt: dict[str, dict[str, float]] = {}
        for ch, idx in self.stoi.items():
            cpt[ch] = {
                next_ch: float(self.probs[idx, next_idx])
                for next_ch, next_idx in self.stoi.items()
                if self.counts[idx, next_idx] > 0
            }
        return cpt

    def generate(
        self,
        start_char: str = "b",
        max_new_tokens: int = 40,
        seed: int | None = None,
        method: str = "sampling",
    ) -> str:
        """Generate text from the bigram model."""
        rng = np.random.default_rng(seed)
        current = start_char
        generated = [current]

        for _ in range(max_new_tokens):
            cur_idx = self.stoi[current]
            if method == "greedy":
                next_idx = int(np.argmax(self.probs[cur_idx]))
            else:
                next_idx = int(rng.choice(self.V, p=self.probs[cur_idx]))
            current = self.itos[next_idx]
            generated.append(current)

        return "".join(generated)

    def evaluate_loss(self, text: str = CANONICAL_CORPUS) -> float:
        """Compute cross-entropy loss of the bigram model on text."""
        total_loss = 0.0
        n = len(text) - 1
        for a, b in zip(text[:-1], text[1:], strict=False):
            prob = self.probs[self.stoi[a], self.stoi[b]]
            total_loss -= np.log(max(prob, 1e-12))
        return float(total_loss / n)


def get_batch(
    data: torch.Tensor | str = CANONICAL_CORPUS,
    block_size: int = 32,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate shifted input-target batch pairs (xb, yb) of shape (B, T)."""
    if isinstance(data, str):
        tokenizer = CharTokenizer(data)
        data_tensor = torch.tensor(tokenizer.encode(data), dtype=torch.long)
    else:
        data_tensor = data

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    starts = torch.randint(
        0,
        len(data_tensor) - block_size - 1,
        (batch_size,),
        generator=generator,
    )

    x = torch.stack([data_tensor[s : s + block_size] for s in starts]).to(device)
    y = torch.stack([data_tensor[s + 1 : s + block_size + 1] for s in starts]).to(device)
    return x, y
