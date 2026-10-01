# CS F407 Lab, Week 7 | Author: Samar Talwar | Not licensed for reuse or submission by others.

"""
Transformer building blocks in PyTorch.

Implements from first principles (CPU, deterministic):
- Scaled Dot-Product Attention (with optional mask and scaling factor)
- Causal (autoregressive) Masking
- Sinusoidal Positional Encoding
- Multi-Head Attention (supporting both self-attention and cross-attention)
- Position-wise Feed-Forward Network
- Transformer Decoder Block and Decoder-Only Autoregressive Model
- Training loop, greedy and temperature/top-k text generation
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M) V

    Parameters
    ----------
    query : torch.Tensor
        Query tensor of shape (..., seq_len_q, d_k).
    key : torch.Tensor
        Key tensor of shape (..., seq_len_k, d_k).
    value : torch.Tensor
        Value tensor of shape (..., seq_len_k, d_v).
    mask : torch.Tensor | None
        Optional mask tensor broadcastable to (..., seq_len_q, seq_len_k).
        Positions with 1 (or True) or large negative values are masked out.
    scale : float | None
        Scaling factor. Defaults to 1.0 / sqrt(d_k).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (output, attention_weights) where output has shape (..., seq_len_q, d_v)
        and attention_weights has shape (..., seq_len_q, seq_len_k).
    """
    d_k = query.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            scores = scores + mask

    weights = F.softmax(scores, dim=-1)
    # Handle NaNs if entire row is masked out (e.g. padding edge cases)
    weights = torch.nan_to_num(weights, nan=0.0)
    output = torch.matmul(weights, value)
    return output, weights


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Create an upper-triangular causal mask where position (i, j) is True if j > i.
    When applied with masked_fill(mask, -inf), token i cannot attend to token j > i.
    """
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    return mask


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as introduced in 'Attention Is All You Need':
    PE(pos, 2i)   = sin(pos / (10000^(2i/d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention supporting both self-attention and cross-attention.
    Projects inputs into `num_heads` subspaces, performs scaled dot-product attention
    in parallel, concatenates head outputs, and applies output linear projection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            msg = f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            raise ValueError(msg)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_causal = is_causal

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch, seq_len_q, d_model).
        key : torch.Tensor | None
            Key tensor of shape (batch, seq_len_k, d_model). If None, self-attention.
        value : torch.Tensor | None
            Value tensor of shape (batch, seq_len_k, d_model). If None, self-attention.
        mask : torch.Tensor | None
            Optional attention mask.
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.size(1)

        q = (
            self.w_q(query)
            .view(batch_size, seq_len_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = (
            self.w_v(value)
            .view(batch_size, seq_len_k, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_mask = mask
        if self.is_causal:
            causal_mask = create_causal_mask(seq_len_q, device=query.device)
            attn_mask = causal_mask if attn_mask is None else attn_mask | causal_mask

        out, attn_weights = scaled_dot_product_attention(q, k, v, mask=attn_mask)
        out = self.dropout(out)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(out)
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    FFN(x) = max(0, x W1 + b1) W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerDecoderBlock(nn.Module):
    """
    Single Decoder block with masked self-attention, residual connections,
    layer normalizations, and feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, is_causal=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, weights


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-Only Autoregressive Transformer (GPT-style next-token prediction).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token indices of shape (batch_size, seq_len).

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor]]
            (logits of shape (batch, seq_len, vocab_size), list of attention weights)
        """
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            msg = f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            raise ValueError(msg)

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        all_weights = []
        for block in self.blocks:
            x, weights = block(x, mask=mask)
            all_weights.append(weights)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, all_weights


class SimpleTokenizer:
    """
    Character or word level tokenizer for reproducible toy training.
    """

    def __init__(self, vocab: list[str], mode: str = "char") -> None:
        self.mode = mode
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for i, t in enumerate(vocab)}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"

    @classmethod
    def from_text(cls, text: str, mode: str = "char") -> SimpleTokenizer:
        special = ["<pad>", "<unk>", "<s>", "</s>"]
        if mode == "char":
            chars = sorted(set(text))
            vocab = special + [c for c in chars if c not in special]
        else:
            words = sorted(set(text.split()))
            vocab = special + [w for w in words if w not in special]
        return cls(vocab, mode=mode)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        tokens = list(text) if self.mode == "char" else text.split()
        unk_id = self.token_to_id.get(self.unk_token, 1)
        ids = [self.token_to_id.get(t, unk_id) for t in tokens]
        if add_special:
            bos_id = self.token_to_id.get(self.bos_token, 2)
            eos_id = self.token_to_id.get(self.eos_token, 3)
            ids = [bos_id] + ids + [eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens = []
        special_tokens = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        for idx in ids:
            tok = self.id_to_token.get(idx, self.unk_token)
            if skip_special and tok in special_tokens:
                continue
            tokens.append(tok)
        return "".join(tokens) if self.mode == "char" else " ".join(tokens)


def train_autoregressive_model(
    model: DecoderOnlyTransformer,
    text: str,
    tokenizer: SimpleTokenizer,
    epochs: int = 100,
    lr: float = 3e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train a decoder-only transformer on a toy text sequence using next-token prediction.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_ids = tokenizer.encode(text, add_special=False)
    if len(input_ids) < 2:
        msg = "Training text must contain at least 2 tokens"
        raise ValueError(msg)

    # Inputs: x[0 : T-1], Targets: x[1 : T]
    x_tensor = torch.tensor([input_ids[:-1]], dtype=torch.long)
    y_tensor = torch.tensor([input_ids[1:]], dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[dict[str, float]] = []

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, _ = model(x_tensor)
        # logits: (1, seq_len, vocab_size), targets: (1, seq_len)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y_tensor.view(-1))
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        perplexity = float(math.exp(min(loss_val, 20.0)))
        loss_history.append({"epoch": epoch, "loss": loss_val, "perplexity": perplexity})

    model.eval()
    return {
        "final_loss": loss_history[-1]["loss"],
        "final_perplexity": loss_history[-1]["perplexity"],
        "loss_history": loss_history,
        "epochs": epochs,
        "seed": seed,
    }


def generate_text(
    model: DecoderOnlyTransformer,
    prompt: str,
    tokenizer: SimpleTokenizer,
    max_new_tokens: int = 30,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = True,
    seed: int | None = 42,
) -> str:
    """
    Autoregressively generate text given a prompt.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    input_ids = tokenizer.encode(prompt, add_special=False)
    if not input_ids:
        input_ids = [tokenizer.token_to_id.get(tokenizer.bos_token, 2)]

    curr_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_tensor = torch.tensor([curr_ids[-model.max_seq_len :]], dtype=torch.long)
            logits, _ = model(seq_tensor)
            next_logits = logits[0, -1, :]

            if greedy or temperature <= 0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                scaled_logits = next_logits / temperature
                if top_k is not None and top_k > 0:
                    top_k_val = min(top_k, scaled_logits.size(-1))
                    indices_to_remove = (
                        scaled_logits < torch.topk(scaled_logits, top_k_val)[0][..., -1, None]
                    )
                    scaled_logits[indices_to_remove] = float("-inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            curr_ids.append(next_id)
            eos_id = tokenizer.token_to_id.get(tokenizer.eos_token, -1)
            if next_id == eos_id:
                break

    return tokenizer.decode(curr_ids, skip_special=True)
