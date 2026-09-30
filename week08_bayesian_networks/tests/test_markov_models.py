# CS F407 Lab, Week 8 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Comprehensive unit, oracle, sampling, and mutation test suite for Week 8 Markov models."""

from __future__ import annotations

import math
import random
from collections import Counter
from fractions import Fraction

import pytest

from week08_bayesian_networks.src.dataset import (
    CANONICAL_CORPUS,
    END_TOKEN,
    START_TOKEN,
)
from week08_bayesian_networks.src.markov_model import (
    FirstOrderMarkovModel,
    SecondOrderMarkovModel,
)

# ---------------------------------------------------------------------------
# Independent Exact Fraction Oracles (Zero Dependency on src training code)
# ---------------------------------------------------------------------------


def compute_oracle_first_order_fractions(
    corpus: list[str],
) -> dict[str, dict[str, Fraction]]:
    """Compute exact transition CPTs as Fraction objects directly from raw sentences."""
    counts: dict[str, Counter[str]] = {}
    for sentence in corpus:
        tokens = [START_TOKEN] + sentence.strip().lower().split() + [END_TOKEN]
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 not in counts:
                counts[w1] = Counter()
            counts[w1][w2] += 1

    cpt: dict[str, dict[str, Fraction]] = {}
    for w1, next_counts in counts.items():
        total = sum(next_counts.values())
        cpt[w1] = {w2: Fraction(cnt, total) for w2, cnt in next_counts.items()}
    return cpt


def compute_oracle_second_order_fractions(
    corpus: list[str],
) -> dict[tuple[str, str], dict[str, Fraction]]:
    """Compute exact second-order CPTs as Fraction objects directly from raw sentences."""
    counts: dict[tuple[str, str], Counter[str]] = {}
    for sentence in corpus:
        tokens = (
            [START_TOKEN, START_TOKEN]
            + sentence.strip().lower().split()
            + [END_TOKEN]
        )
        for i in range(len(tokens) - 2):
            ctx: tuple[str, str] = (tokens[i], tokens[i + 1])
            w3 = tokens[i + 2]
            if ctx not in counts:
                counts[ctx] = Counter()
            counts[ctx][w3] += 1

    cpt: dict[tuple[str, str], dict[str, Fraction]] = {}
    for ctx, next_counts in counts.items():
        total = sum(next_counts.values())
        cpt[ctx] = {w3: Fraction(cnt, total) for w3, cnt in next_counts.items()}
    return cpt


# ---------------------------------------------------------------------------
# 1. Exact Oracle Verification Tests
# ---------------------------------------------------------------------------


def test_first_order_exact_fraction_oracle() -> None:
    """Verify every first-order CPT entry against independently computed Fractions."""
    oracle_cpt = compute_oracle_first_order_fractions(CANONICAL_CORPUS)
    model = FirstOrderMarkovModel()
    model.train(CANONICAL_CORPUS)

    # Observed contexts must match exactly
    assert set(model.get_observed_contexts()) == set(oracle_cpt.keys())

    for ctx, oracle_dist in oracle_cpt.items():
        model_dist = model.get_cpt(ctx)
        assert model_dist is not None, f"Context {ctx} missing in model"
        assert set(model_dist.keys()) == set(oracle_dist.keys())

        for token, oracle_frac in oracle_dist.items():
            model_p = model_dist[token]
            oracle_float = float(oracle_frac)
            assert abs(model_p - oracle_float) < 1e-12, (
                f"Mismatch for P({token} | {ctx}): "
                f"model={model_p}, oracle={oracle_frac} ({oracle_float})"
            )


def test_second_order_exact_fraction_oracle() -> None:
    """Verify every second-order CPT entry against independently computed Fractions."""
    oracle_cpt = compute_oracle_second_order_fractions(CANONICAL_CORPUS)
    model = SecondOrderMarkovModel()
    model.train(CANONICAL_CORPUS)

    # Observed 2-token contexts must match exactly
    assert set(model.get_observed_contexts()) == set(oracle_cpt.keys())

    for ctx, oracle_dist in oracle_cpt.items():
        model_dist = model.get_cpt(ctx)
        assert model_dist is not None, f"Context {ctx} missing in second-order model"
        assert set(model_dist.keys()) == set(oracle_dist.keys())

        for token, oracle_frac in oracle_dist.items():
            model_p = model_dist[token]
            oracle_float = float(oracle_frac)
            assert abs(model_p - oracle_float) < 1e-12, (
                f"Mismatch for P({token} | {ctx}): "
                f"model={model_p}, oracle={oracle_frac} ({oracle_float})"
            )


# ---------------------------------------------------------------------------
# 2. Explicit Hand-Checked Probabilities (Manual Counting Assertions)
# ---------------------------------------------------------------------------


def test_five_hand_checked_probabilities() -> None:
    """Assert at least 5 key probabilities explicitly computed by hand."""
    m1 = FirstOrderMarkovModel()
    m1.train(CANONICAL_CORPUS)

    m2 = SecondOrderMarkovModel()
    m2.train(CANONICAL_CORPUS)

    # Hand Check 1: In first-order, 'the' appears 12 times.
    # Next tokens: cat (3), dog (3), mat (2), rug (2), park (2).
    # P(cat | the) = 3/12 = 1/4 = 0.25
    cpt_the = m1.get_cpt("the")
    assert cpt_the is not None
    assert abs(cpt_the["cat"] - 0.25) < 1e-12
    assert abs(cpt_the["dog"] - 0.25) < 1e-12
    assert abs(cpt_the["mat"] - (2 / 12)) < 1e-12

    # Hand Check 2: In first-order, 'cat' appears 3 times. Followed by 'sat' 2, 'ran' 1.
    # P(sat | cat) = 2/3
    cpt_cat = m1.get_cpt("cat")
    assert cpt_cat is not None
    assert abs(cpt_cat["sat"] - (2 / 3)) < 1e-12
    assert abs(cpt_cat["ran"] - (1 / 3)) < 1e-12

    # Hand Check 3: In first-order, 'sat' appears 4 times, always followed by 'on'.
    # P(on | sat) = 4/4 = 1.0
    cpt_sat = m1.get_cpt("sat")
    assert cpt_sat is not None
    assert abs(cpt_sat["on"] - 1.0) < 1e-12

    # Hand Check 4: In second-order, context ('<START>', 'the') appears 6 times.
    # Followed by 'cat' 3 times, 'dog' 3 times.
    # P(cat | <START>, the) = 3/6 = 0.5
    # P(dog | <START>, the) = 3/6 = 0.5
    cpt_start_the = m2.get_cpt((START_TOKEN, "the"))
    assert cpt_start_the is not None
    assert abs(cpt_start_the["cat"] - 0.5) < 1e-12
    assert abs(cpt_start_the["dog"] - 0.5) < 1e-12

    # Hand Check 5: In second-order, context ('on', 'the') appears 4 times.
    # Followed by 'mat' 2 times, 'rug' 2 times.
    # P(mat | on, the) = 2/4 = 0.5
    # P(rug | on, the) = 2/4 = 0.5
    cpt_on_the = m2.get_cpt(("on", "the"))
    assert cpt_on_the is not None
    assert abs(cpt_on_the["mat"] - 0.5) < 1e-12
    assert abs(cpt_on_the["rug"] - 0.5) < 1e-12

    # Hand Check 6: In second-order, ('to', 'the') appears 2 times, always followed by 'park'.
    # P(park | to, the) = 2/2 = 1.0
    cpt_to_the = m2.get_cpt(("to", "the"))
    assert cpt_to_the is not None
    assert abs(cpt_to_the["park"] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 3. Normalization and Unseen Context Tests
# ---------------------------------------------------------------------------


def test_normalization_first_and_second_order() -> None:
    """Verify that every observed conditional distribution sums to exactly 1.0 within 1e-12."""
    m1 = FirstOrderMarkovModel()
    m1.train(CANONICAL_CORPUS)
    norm1 = m1.check_normalization()
    assert norm1["all_contexts_normalized"] is True
    assert norm1["max_deviation"] < 1e-12

    m2 = SecondOrderMarkovModel()
    m2.train(CANONICAL_CORPUS)
    norm2 = m2.check_normalization()
    assert norm2["all_contexts_normalized"] is True
    assert norm2["max_deviation"] < 1e-12


def test_unseen_contexts_handling() -> None:
    """Verify that unobserved contexts report None/unseen and never invent probabilities."""
    m1 = FirstOrderMarkovModel()
    m1.train(CANONICAL_CORPUS)

    # Unseen token in first-order
    assert m1.get_cpt("elephant") is None
    assert m1.get_counts("elephant") is None
    assert m1.predict_next("elephant") is None

    # Unseen bigram context in second-order
    m2 = SecondOrderMarkovModel()
    m2.train(CANONICAL_CORPUS)
    assert m2.get_cpt(("cat", "dog")) is None
    assert m2.get_counts(("cat", "dog")) is None
    assert m2.predict_next(("cat", "dog")) is None
    assert m2.get_cpt(("elephant", "mat")) is None


# ---------------------------------------------------------------------------
# 4. Stochastic Sampling & Empirical Convergence ($N = 20,000$)
# ---------------------------------------------------------------------------


def test_sampling_empirical_frequency_convergence() -> None:
    """With N=20000 draws from context 'the', empirical frequencies are within 0.02 of CPT."""
    m1 = FirstOrderMarkovModel()
    m1.train(CANONICAL_CORPUS)

    true_cpt = m1.get_cpt("the")
    assert true_cpt is not None

    n_draws = 20000
    rng = random.Random(42)
    counts: Counter[str] = Counter()

    for _ in range(n_draws):
        token = m1.predict_next("the", method="sampling", rng=rng)
        assert token is not None
        counts[token] += 1

    for token, true_p in true_cpt.items():
        empirical_p = counts[token] / n_draws
        abs_diff = abs(empirical_p - true_p)
        assert abs_diff < 0.02, (
            f"Sampling frequency deviation for token '{token}' too high: "
            f"empirical={empirical_p:.4f}, true={true_p:.4f}, diff={abs_diff:.4f}"
        )


def test_sampling_seed_reproducibility() -> None:
    """Assert identical seeds generate identical sentences and different seeds differ."""
    m1 = FirstOrderMarkovModel()
    m1.train(CANONICAL_CORPUS)

    res_seed42_a = m1.generate(method="sampling", seed=42)
    res_seed42_b = m1.generate(method="sampling", seed=42)
    res_seed99 = m1.generate(method="sampling", seed=99)

    assert res_seed42_a.tokens == res_seed42_b.tokens
    assert res_seed42_a.text == res_seed42_b.text
    assert res_seed42_a.tokens != res_seed99.tokens


# ---------------------------------------------------------------------------
# 5. Greedy Decoding Determinism & Cycle Detection
# ---------------------------------------------------------------------------


def test_greedy_determinism_and_cycle_detection() -> None:
    """Verify greedy generation determinism and loop detection on cyclic Markov chains."""
    m1 = FirstOrderMarkovModel()
    m1.train(CANONICAL_CORPUS)

    # Run greedy generation multiple times
    gen1 = m1.generate(method="greedy")
    gen2 = m1.generate(method="greedy")

    assert gen1.tokens == gen2.tokens
    assert gen1.text == gen2.text
    # Greedy transitions: <START> -> the -> cat -> sat -> on -> the -> [LOOP]
    assert gen1.cycle_detected is True
    assert gen1.cycle_tokens is not None
    assert "the" in gen1.cycle_tokens
    assert "cat" in gen1.cycle_tokens


def test_custom_cycle_detection_model() -> None:
    """Test cycle detection on a synthetic deterministic cyclic Markov chain A -> B -> C -> A."""
    synthetic_corpus = ["x y z x y z x", "x y z x y z y"]
    m = FirstOrderMarkovModel()
    m.train(synthetic_corpus)

    res = m.generate(method="greedy", max_length=20)
    assert res.cycle_detected is True
    assert res.num_tokens < 20
    assert res.cycle_tokens is not None


def test_second_order_greedy_generation() -> None:
    """Second-order greedy generation terminates cleanly at <END> with deterministic tie-break."""
    m2 = SecondOrderMarkovModel()
    m2.train(CANONICAL_CORPUS)

    gen = m2.generate(method="greedy")
    assert gen.terminated_with_end is True
    assert gen.cycle_detected is False
    expected_words = ["the", "cat", "sat", "on", "the", "mat"]
    word_tokens = [t for t in gen.tokens if t not in (START_TOKEN, END_TOKEN)]
    assert word_tokens == expected_words


# ---------------------------------------------------------------------------
# 6. N-gram Validity of Generated Samples
# ---------------------------------------------------------------------------


def test_ngram_validity_of_generated_sentences() -> None:
    """Every generated token n-gram must be an observed transition in the training data."""
    m1 = FirstOrderMarkovModel()
    m1.train(CANONICAL_CORPUS)

    m2 = SecondOrderMarkovModel()
    m2.train(CANONICAL_CORPUS)

    # 1. Collect all training bigrams and trigrams
    training_bigrams: set[tuple[str, str]] = set()
    for s in CANONICAL_CORPUS:
        t1 = [START_TOKEN] + s.split() + [END_TOKEN]
        for i in range(len(t1) - 1):
            training_bigrams.add((t1[i], t1[i + 1]))

    training_trigrams: set[tuple[str, str, str]] = set()
    for s in CANONICAL_CORPUS:
        t2 = [START_TOKEN, START_TOKEN] + s.split() + [END_TOKEN]
        for i in range(len(t2) - 2):
            training_trigrams.add((t2[i], t2[i + 1], t2[i + 2]))

    # 2. Check 100 first-order generated samples
    for seed in range(100):
        gen1 = m1.generate(method="sampling", max_length=40, seed=seed)
        for i in range(len(gen1.tokens) - 1):
            bg = (gen1.tokens[i], gen1.tokens[i + 1])
            assert bg in training_bigrams, f"Invalid generated bigram {bg} for seed {seed}"

    # 3. Check 100 second-order generated samples
    for seed in range(100):
        gen2 = m2.generate(method="sampling", max_length=40, seed=seed)
        for i in range(len(gen2.tokens) - 2):
            tg = (gen2.tokens[i], gen2.tokens[i + 1], gen2.tokens[i + 2])
            assert tg in training_trigrams, f"Invalid generated trigram {tg} for seed {seed}"


# ---------------------------------------------------------------------------
# 7. Chain-Rule Sentence Probability & Factorisation Tests
# ---------------------------------------------------------------------------


def test_chain_rule_sentence_probabilities() -> None:
    """Test chain-rule joint and log-probabilities for corpus and zero-probability sentences."""
    m1 = FirstOrderMarkovModel()
    m1.train(CANONICAL_CORPUS)

    for sentence in CANONICAL_CORPUS:
        res = m1.sentence_probability(sentence)
        assert res.joint_probability > 0.0
        assert not math.isinf(res.log_probability)
        assert abs(math.exp(res.log_probability) - res.joint_probability) < 1e-9

    # Unseen transition sentence should evaluate to zero probability and -inf log-probability
    impossible_sentence = "the mat sat on the cat"
    res_imp = m1.sentence_probability(impossible_sentence)
    assert res_imp.is_zero_probability is True
    assert res_imp.joint_probability == 0.0
    assert math.isinf(res_imp.log_probability) and res_imp.log_probability < 0


# ---------------------------------------------------------------------------
# 8. Real Mutation Tests (Monkeypatching ACTUAL src Functions)
# ---------------------------------------------------------------------------


def test_mutation_1_unnormalized_cpt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation 1: Corrupt normalisation logic and verify normalisation tests fail."""
    original_train = FirstOrderMarkovModel.train

    def broken_train(self: FirstOrderMarkovModel, sentences: list[str] | None = None) -> None:
        original_train(self, sentences)
        # Mutate: overwrite CPT with unnormalised raw counts instead of dividing by total
        for ctx, next_counts in self.counts.items():
            self.cpt[ctx] = {k: float(v) for k, v in next_counts.items()}

    monkeypatch.setattr(FirstOrderMarkovModel, "train", broken_train)

    m = FirstOrderMarkovModel()
    m.train(CANONICAL_CORPUS)
    norm = m.check_normalization()

    # Normalization check MUST detect the corruption
    assert norm["all_contexts_normalized"] is False
    assert norm["max_deviation"] > 0.5


def test_mutation_2_second_order_ignores_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation 2: Make second-order model ignore w_{t-2} and verify oracle test fails."""
    original_train = SecondOrderMarkovModel.train

    def broken_second_order_train(
        self: SecondOrderMarkovModel, sentences: list[str] | None = None
    ) -> None:
        original_train(self, sentences)
        # Mutate: collapse all (w_{t-2}, w_{t-1}) contexts into (<START>, w_{t-1}), ignoring w_{t-2}
        corrupted_cpt: dict[tuple[str, str], dict[str, float]] = {}
        for (w_prev2, w_prev1), dist in self.cpt.items():
            corrupted_cpt[(w_prev2, w_prev1)] = dict(dist)
            # Override context ('on', 'the') to have distribution of ('<START>', 'the')
            if (w_prev2, w_prev1) == ("on", "the"):
                corrupted_cpt[(w_prev2, w_prev1)] = {"cat": 0.5, "dog": 0.5}
        self.cpt = corrupted_cpt

    monkeypatch.setattr(SecondOrderMarkovModel, "train", broken_second_order_train)

    m2 = SecondOrderMarkovModel()
    m2.train(CANONICAL_CORPUS)

    # Compare against independent fraction oracle
    oracle_cpt = compute_oracle_second_order_fractions(CANONICAL_CORPUS)
    model_dist = m2.get_cpt(("on", "the"))
    oracle_dist = oracle_cpt[("on", "the")]

    # Mutation check: model dist must diverge from true oracle
    divergence_found = False
    assert model_dist is not None
    for tok, oracle_frac in oracle_dist.items():
        if abs(model_dist.get(tok, 0.0) - float(oracle_frac)) > 0.1:
            divergence_found = True
            break
    assert divergence_found is True, "Mutation was not caught by oracle comparison!"


def test_mutation_3_broken_greedy_tie_break(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation 3: Break deterministic tie-break and verify output changes."""
    m1 = FirstOrderMarkovModel()
    m1.train(CANONICAL_CORPUS)

    # Standard greedy generation on CANONICAL_CORPUS produces cycle with 'cat'
    standard_gen = m1.generate(method="greedy")

    # Mutate predict_next greedy tie-breaking to pick reverse lexicographical order
    original_predict = FirstOrderMarkovModel.predict_next

    def reversed_tie_break_predict(
        self: FirstOrderMarkovModel,
        context: str,
        method: str = "greedy",
        rng: random.Random | None = None,
    ) -> str | None:
        dist = self.get_cpt(context)
        if dist is None or len(dist) == 0:
            return None
        if method == "greedy":
            # Reversed tie-break: highest prob, then lexicographically LARGEST token
            sorted_candidates = sorted(
                dist.items(), key=lambda kv: (-kv[1], -ord(kv[0][0]))
            )
            return sorted_candidates[0][0]
        return original_predict(self, context, method=method, rng=rng)

    monkeypatch.setattr(FirstOrderMarkovModel, "predict_next", reversed_tie_break_predict)

    mutated_gen = m1.generate(method="greedy")
    # 'the' transitions to 'dog' instead of 'cat'
    assert standard_gen.tokens != mutated_gen.tokens
    assert mutated_gen.tokens[2] == "dog"
