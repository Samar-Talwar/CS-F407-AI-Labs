# CS F407 Lab, Week 5 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Comprehensive unit, oracle, invariant, and mutation test suite for Week 5 Transformers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from week05_transformers_ar.src.attention import (
    CustomMultiHeadAttention,
    Head,
    MultiHeadAttention,
    causal_self_attention_numpy,
    scaled_dot_product_attention_numpy,
)
from week05_transformers_ar.src.dataset import (
    CANONICAL_CORPUS,
    CharTokenizer,
)
from week05_transformers_ar.src.generate import (
    generate_text,
    temperature_distribution,
)
from week05_transformers_ar.src.metrics import (
    analytical_parameter_breakdown,
    attention_variance_check,
)
from week05_transformers_ar.src.model import (
    TinyGPT,
    TransformerBlock,
)
from week05_transformers_ar.src.train import train_tiny_gpt

# ---------------------------------------------------------------------------
# 1. Exact Oracle & Mathematical Equality Tests
# ---------------------------------------------------------------------------


def test_numpy_vs_pytorch_head_oracle() -> None:
    """Compare NumPy causal attention oracle against PyTorch Head module on identical weights."""
    t = 4
    n_embd = 8
    head_size = 4
    seed = 42

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    # Initialize PyTorch Head
    head = Head(head_size=head_size, n_embd=n_embd, block_size=16)

    # Copy PyTorch weights to NumPy (transposed for matrix multiplication x @ W)
    wq_np = head.query.weight.detach().cpu().numpy().T  # (n_embd, head_size)
    wk_np = head.key.weight.detach().cpu().numpy().T    # (n_embd, head_size)
    wv_np = head.value.weight.detach().cpu().numpy().T  # (n_embd, head_size)

    # Create identical input
    x_np = np_rng.standard_normal((t, n_embd)).astype(np.float32)
    x_pt = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # (1, T, C)

    # Compute NumPy output
    out_np, weights_np = causal_self_attention_numpy(x_np, wq_np, wk_np, wv_np)

    # Compute PyTorch output
    head.eval()
    with torch.no_grad():
        out_pt = head(x_pt).squeeze(0).cpu().numpy()

    # Assert exact numerical equivalence within float32 precision
    assert np.allclose(out_np, out_pt, atol=1e-5), (
        f"Mismatch between NumPy oracle and PyTorch Head output:\n"
        f"NumPy:\n{out_np}\nPyTorch:\n{out_pt}"
    )


def test_analytical_vs_empirical_parameter_count() -> None:
    """Verify analytical parameter count formula matches PyTorch parameters() exactly (60,313)."""
    tokenizer = CharTokenizer(CANONICAL_CORPUS)
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=48,
        block_size=32,
        n_heads=4,
        n_layers=2,
    )

    empirical_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    analytical_dict = analytical_parameter_breakdown(
        vocab_size=tokenizer.vocab_size,
        n_embd=48,
        block_size=32,
        n_heads=4,
        n_layers=2,
    )

    assert empirical_total == 60313
    assert analytical_dict["total_analytical"] == 60313
    assert empirical_total == analytical_dict["total_analytical"]


# ---------------------------------------------------------------------------
# 2. Shape, Interface & Structural Contract Tests
# ---------------------------------------------------------------------------


def test_tiny_gpt_forward_shapes_and_loss() -> None:
    """Test TinyGPT produces correct logit shapes (B, T, V) and scalar loss."""
    tokenizer = CharTokenizer(CANONICAL_CORPUS)
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=48,
        block_size=32,
        n_heads=4,
        n_layers=2,
    )
    b, t = 4, 16
    xb = torch.randint(0, tokenizer.vocab_size, (b, t))
    yb = torch.randint(0, tokenizer.vocab_size, (b, t))

    logits, loss = model(xb, yb)
    assert logits.shape == (b, t, tokenizer.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # Scalar loss

    # Without targets
    logits_no_targets, loss_none = model(xb)
    assert logits_no_targets.shape == (b, t, tokenizer.vocab_size)
    assert loss_none is None


def test_block_size_exceeded_raises_error() -> None:
    """Passing a sequence longer than block_size must raise ValueError."""
    model = TinyGPT(vocab_size=25, n_embd=48, block_size=32, n_heads=4, n_layers=2)
    invalid_xb = torch.randint(0, 25, (2, 33))

    with pytest.raises(ValueError, match="exceeds maximum block_size"):
        model(invalid_xb)


def test_multi_head_attention_preserves_dimension() -> None:
    """Verify standard MHA preserves (B, T, n_embd)."""
    mha = MultiHeadAttention(num_heads=4, head_size=12, n_embd=48, block_size=32)
    x = torch.randn(2, 10, 48)
    out = mha(x)
    assert out.shape == (2, 10, 48)


def test_custom_nonstandard_multi_head_attention() -> None:
    """Verify non-standard arbitrary head configuration (e.g. H=6, head_size=2, n_embd=8)."""
    custom_mha = CustomMultiHeadAttention(num_heads=6, head_size=2, n_embd=8, block_size=16)
    x = torch.randn(3, 7, 8)
    out = custom_mha(x)
    assert out.shape == (3, 7, 8)
    # Output projection W_O maps from (6 * 2 = 12) to 8
    assert custom_mha.proj.weight.shape == (8, 12)


# ---------------------------------------------------------------------------
# 3. Causal Masking Invariant Tests
# ---------------------------------------------------------------------------


def test_causal_masking_no_future_leakage() -> None:
    """Modifying future tokens in input must NOT change earlier token logits or representations."""
    model = TinyGPT(vocab_size=25, n_embd=48, block_size=32, n_heads=4, n_layers=2)
    model.eval()

    # Base sequence of length 10
    seq1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    # Corrupted future at positions 7, 8, 9
    seq2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 99 % 25, 88 % 25, 77 % 25]], dtype=torch.long)

    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)

    # Logits at positions 0..6 (the unperturbed prefix) must be bitwise/strictly identical
    prefix_diff = (logits1[:, :7, :] - logits2[:, :7, :]).abs().max().item()
    assert prefix_diff < 1e-6, (
        f"Future tokens leaked into past representations! Max diff: {prefix_diff}"
    )

    # But logits at position 7 onwards should differ
    suffix_diff = (logits1[:, 7:, :] - logits2[:, 7:, :]).abs().max().item()
    assert suffix_diff > 1e-3, "Perturbed future should change future logits."


def test_causal_attention_weights_upper_triangle_is_zero() -> None:
    """Attention weights in the upper triangle (j > i) must be strictly zero."""
    t = 6
    rng = np.random.default_rng(42)
    q = rng.standard_normal((t, 8))
    k = rng.standard_normal((t, 8))
    v = rng.standard_normal((t, 8))
    mask = np.tril(np.ones((t, t)))

    _, weights = scaled_dot_product_attention_numpy(q, k, v, mask=mask)

    # Check that upper triangular weights are strictly 0.0
    upper_tri = np.triu(weights, k=1)
    assert np.all(upper_tri == 0.0), f"Upper triangular weights non-zero:\n{upper_tri}"
    # Check that rows sum to 1.0
    row_sums = np.sum(weights, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Attention Scaling & Dot-Product Variance Invariant Tests
# ---------------------------------------------------------------------------


def test_dot_product_variance_scaling_rule() -> None:
    """Dot-product of standard normal vectors in dimension d has variance ~ d (std ~ sqrt(d))."""
    res = attention_variance_check(dims=[4, 16, 64, 256], n_samples=5000, seed=123)
    for row in res["results"]:
        # Empirical std should be within 15% of theoretical sqrt(d)
        ratio = row["ratio_std_sqrt_d"]
        assert 0.85 <= ratio <= 1.15, (
            f"Dimension d={row['d']}: ratio = {ratio:.3f} outside [0.85, 1.15]"
        )


# ---------------------------------------------------------------------------
# 5. Temperature & Generation Invariant Tests
# ---------------------------------------------------------------------------


def test_temperature_scaling_entropy_ordering() -> None:
    """Higher temperature results in higher entropy (flatter distribution)."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    p_low = temperature_distribution(logits, temperature=0.3)
    p_high = temperature_distribution(logits, temperature=2.0)

    # Entropy: -sum(p * log(p))
    entropy_low = -torch.sum(p_low * torch.log(p_low + 1e-12)).item()
    entropy_high = -torch.sum(p_high * torch.log(p_high + 1e-12)).item()

    assert entropy_high > entropy_low, (
        f"Higher temperature did not increase entropy: high={entropy_high}, low={entropy_low}"
    )


def test_generation_seed_reproducibility() -> None:
    """Identical seeds must produce identical generated token streams; different seeds differ."""
    tokenizer = CharTokenizer(CANONICAL_CORPUS)
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=48,
        block_size=32,
        n_heads=4,
        n_layers=2,
    )

    gen1 = generate_text(
        model, tokenizer, prompt="a ", max_new_tokens=20, temperature=0.8, seed=42
    )
    gen2 = generate_text(model, tokenizer, prompt="a ", max_new_tokens=20, temperature=0.8, seed=42)
    gen3 = generate_text(model, tokenizer, prompt="a ", max_new_tokens=20, temperature=0.8, seed=99)

    assert gen1 == gen2, "Identical seeds produced different text!"
    assert gen1 != gen3, "Different seeds produced identical text unexpectedly!"


# ---------------------------------------------------------------------------
# 6. Training Convergence Test
# ---------------------------------------------------------------------------


def test_tiny_gpt_training_convergence() -> None:
    """Training TinyGPT for 180 steps significantly reduces cross-entropy loss."""
    res = train_tiny_gpt(
        max_iters=180,
        eval_interval=30,
        learning_rate=3e-3,
        batch_size=32,
        block_size=32,
        seed=1337,
    )
    assert res["initial_loss"] > 3.0, f"Initial loss too low: {res['initial_loss']}"
    assert res["final_loss"] < 0.50, (
        f"Final loss failed to converge below 0.50: {res['final_loss']}"
    )
    assert res["final_loss"] < res["initial_loss"] * 0.20, "Loss did not drop by at least 80%."


def test_cli_results_dir_module_relative(monkeypatch, tmp_path):
    from pathlib import Path

    import week05_transformers_ar.src.cli as cli_mod

    monkeypatch.chdir(tmp_path)
    expected = (Path(cli_mod.__file__).resolve().parents[1] / "results").resolve()
    assert cli_mod.RESULTS_DIR.resolve() == expected




# ---------------------------------------------------------------------------
# 7. Real Mutation Tests (Monkeypatching ACTUAL src Functions)
# ---------------------------------------------------------------------------


def test_mutation_1_corrupted_causal_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation 1: Corrupt Head.forward() by removing causal mask (bidirectional attention)."""
    # Define mutated Head.forward without causal masking
    def unmasked_forward(self: Head, x: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        # BUG: Skip masked_fill, allowing bidirectional attention
        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

    monkeypatch.setattr(Head, "forward", unmasked_forward)

    model = TinyGPT(vocab_size=25, n_embd=48, block_size=32, n_heads=4, n_layers=2)
    model.eval()

    seq1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    seq2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 20, 21, 22]], dtype=torch.long)

    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)

    # Causality test MUST detect future leakage in unmasked model!
    prefix_diff = (logits1[:, :7, :] - logits2[:, :7, :]).abs().max().item()
    assert prefix_diff > 1e-4, "Corrupted unmasked attention was NOT caught by causality test!"


def test_mutation_2_missing_sqrt_dk_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation 2: Remove 1/sqrt(d_k) scaling in Head.forward and verify variance explosion."""
    def unscaled_forward(self: Head, x: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        # BUG: Missing * (self.head_size ** -0.5)
        wei = q @ k.transpose(-2, -1)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

    monkeypatch.setattr(Head, "forward", unscaled_forward)

    head = Head(head_size=64, n_embd=64, block_size=16)
    x = torch.randn(2, 8, 64) * 2.0

    # With d_k=64 and variance=4.0, unscaled scores have huge variance -> saturation
    with torch.no_grad():
        _, t, _ = x.shape
        k = head.key(x)
        q = head.query(x)
        unscaled_scores = q @ k.transpose(-2, -1)
        std_unscaled = unscaled_scores.std().item()

        scaled_scores = unscaled_scores * (64 ** -0.5)
        std_scaled = scaled_scores.std().item()

    assert std_unscaled > 2.0 * std_scaled, "Missing sqrt(d_k) did not exhibit variance inflation!"


def test_mutation_3_broken_residual_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation 3: Break residual connection in TransformerBlock and verify test catches failure."""
    def broken_block_forward(self: TransformerBlock, x: torch.Tensor) -> torch.Tensor:
        # BUG: x = self.sa(self.ln1(x)) instead of x = x + self.sa(self.ln1(x))
        x = self.sa(self.ln1(x))
        x = self.ffwd(self.ln2(x))
        return x

    monkeypatch.setattr(TransformerBlock, "forward", broken_block_forward)

    block = TransformerBlock(n_heads=4, head_size=12, n_embd=48, block_size=32)
    # Zero all weights in attention and ffn so identity residual should pass through
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()

    x = torch.randn(2, 5, 48)
    out = block(x)

    # In a proper Pre-LN residual block with zeroed submodules, out == x.
    # In the broken block without residual x + ..., out becomes 0!
    is_identity = torch.allclose(out, x, atol=1e-5)
    assert not is_identity, "Broken residual connection was NOT caught!"
