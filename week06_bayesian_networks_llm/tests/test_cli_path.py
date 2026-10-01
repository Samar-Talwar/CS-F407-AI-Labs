# CS F407 Lab, Week 6 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Test that CLI writes inside week folder regardless of cwd."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_cli_results_dir_module_relative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify that results directory is resolved relative to module, not cwd."""
    import week06_bayesian_networks_llm.src.cli as cli_mod

    monkeypatch.chdir(tmp_path)
    d = cli_mod.ensure_results_dir()
    assert d.resolve() == (Path(__file__).resolve().parents[1] / "results").resolve()


