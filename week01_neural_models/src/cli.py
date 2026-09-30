# CS F407 Lab, Week 1 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""CLI: regenerate all Week 1 results into results/.

Usage:  python -m week01_neural_models.src.cli
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from .train import train_binary_xor, train_multiclass, train_symmetry_experiment

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _dump(name: str, obj: object) -> None:
    path = RESULTS_DIR / f"{name}.json"
    data = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  -> {path.relative_to(RESULTS_DIR.parent)}")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Week 1 - Neural Models: generating results ...\n")

    # ── Task 4A: Binary XOR (default sigmoid hidden) ──────────────────────
    print("[Task 4A] Binary XOR (sigmoid hidden):")
    res = train_binary_xor()
    _dump("binary_xor", res)
    print(f"  Final loss = {res.final_loss:.6f}")
    print(f"  Probs      = {[f'{p:.4f}' for p in res.probabilities]}")
    print(f"  Preds      = {res.predictions}")
    print(f"  ||grad_W1|| @step10 = {res.grad_w1_norm:.6f}\n")

    # ── Task 4C: Symmetry experiment ──────────────────────────────────────
    print("[Task 4C] Symmetry (zero-init):")
    sym = train_symmetry_experiment()
    _dump("symmetry", sym)
    print(f"  Rows stayed identical = {sym.rows_identical}")
    print(f"  Final loss = {sym.final_loss:.6f}\n")

    # ── Task 4D: Activation comparison ────────────────────────────────────
    # ReLU needs different seed/lr/steps to avoid dead-unit failure (justified per lab sheet).
    print("[Task 4D] Activation comparison:")
    act_configs = [
        ("sigmoid", 42, 1.0, 5000),
        ("tanh", 42, 1.0, 5000),
        ("relu", 5, 0.5, 20000),
    ]
    table: dict[str, dict[str, object]] = {}
    for act, seed, lr, steps in act_configs:
        r = train_binary_xor(hidden_activation=act, seed=seed, lr=lr, steps=steps)
        table[act] = {
            "final_loss": r.final_loss,
            "all_correct": r.predictions == [0, 1, 1, 0],
            "early_grad_norm": r.grad_w1_norm,
            "seed": seed,
            "lr": lr,
            "steps": steps,
        }
        print(f"  {act:8s}  loss={r.final_loss:.6f}  4/4={r.predictions == [0, 1, 1, 0]}  "
              f"||grad_W1||={r.grad_w1_norm:.6f}")
    _dump("activation_comparison", table)
    print()

    # ── Task 5: Multiclass ────────────────────────────────────────────────
    print("[Task 5] Multiclass (3-class):")
    mc = train_multiclass()
    _dump("multiclass", mc)
    print(f"  Final loss = {mc.final_loss:.6f}")
    print(f"  Predictions = {mc.predictions}")
    print(f"  Prob sums   = {[f'{s:.6f}' for s in mc.prob_sums]}")
    print(f"  Shift-invariant = {mc.shift_invariant}\n")

    print("Done. All results written to results/.")


if __name__ == "__main__":
    sys.exit(main() or 0)
