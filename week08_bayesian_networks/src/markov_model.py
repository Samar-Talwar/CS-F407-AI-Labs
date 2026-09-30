# CS F407 Lab, Week 8 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""First- and second-order Markov language models and autoregressive CPT estimators."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

from week08_bayesian_networks.src.dataset import (
    CANONICAL_CORPUS,
    END_TOKEN,
    START_TOKEN,
    tokenize_corpus,
    tokenize_sentence,
)


@dataclass
class GenerationResult:
    """Result of a single text generation run."""

    tokens: list[str]
    text: str
    method: Literal["greedy", "sampling"]
    seed: int | None
    cycle_detected: bool
    cycle_tokens: list[str] | None
    terminated_with_end: bool
    num_tokens: int
    is_in_training_corpus: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SentenceProbabilityResult:
    """Result of chain-rule probability evaluation for a sentence."""

    sentence: str
    tokens: list[str]
    joint_probability: float
    log_probability: float
    is_zero_probability: bool
    factors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FirstOrderMarkovModel:
    """First-order autoregressive Markov language model: P(w_t | w_{t-1}).

    Independence assumption:
        P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1})
    Chain-rule factorisation:
        P(X_1, ..., X_T) = P(X_1) * prod_{t=2}^T P(X_t | X_{t-1})
    """

    def __init__(self) -> None:
        self.counts: dict[str, Counter[str]] = {}
        self.cpt: dict[str, dict[str, float]] = {}
        self.vocab: set[str] = set()
        self.observed_contexts: set[str] = set()
        self.training_sentences: list[str] = []

    def train(self, sentences: list[str] | None = None) -> None:
        """Train first-order Markov model by counting bigram transitions."""
        self.counts.clear()
        self.cpt.clear()
        self.vocab.clear()
        self.observed_contexts.clear()

        corpus_text = CANONICAL_CORPUS if sentences is None else sentences
        self.training_sentences = list(corpus_text)

        tokenized = tokenize_corpus(self.training_sentences, order=1)
        for seq in tokenized:
            for i in range(len(seq) - 1):
                prev_tok = seq[i]
                next_tok = seq[i + 1]
                self.vocab.add(prev_tok)
                self.vocab.add(next_tok)
                if prev_tok not in self.counts:
                    self.counts[prev_tok] = Counter()
                self.counts[prev_tok][next_tok] += 1
                self.observed_contexts.add(prev_tok)

        # Normalize counts to build CPT
        for prev_tok, next_counts in self.counts.items():
            total = sum(next_counts.values())
            self.cpt[prev_tok] = {}
            for next_tok, cnt in sorted(next_counts.items()):
                self.cpt[prev_tok][next_tok] = cnt / total

    def get_counts(self, context: str) -> dict[str, int] | None:
        """Return raw transition counts for an observed context, or None if unseen."""
        if context not in self.observed_contexts:
            return None
        return dict(self.counts.get(context, {}))

    def get_cpt(self, context: str) -> dict[str, float] | None:
        """Return conditional probability distribution for an observed context (or None)."""
        if context not in self.observed_contexts:
            return None
        return dict(self.cpt.get(context, {}))

    def get_vocabulary(self) -> list[str]:
        """Return sorted list of all unique vocabulary tokens."""
        return sorted(self.vocab)

    def get_observed_contexts(self) -> list[str]:
        """Return sorted list of all observed context tokens."""
        return sorted(self.observed_contexts)

    def predict_next(
        self,
        context: str,
        method: Literal["greedy", "sampling"] = "greedy",
        rng: random.Random | None = None,
    ) -> str | None:
        """Predict the next token given a context.

        Args:
            context: The single previous token w_{t-1}.
            method: 'greedy' (argmax with alphabetical tie-break) or 'sampling' (inverse-CDF).
            rng: random.Random instance for reproducible sampling.

        Returns:
            Next token string, or None if context is unseen.
        """
        dist = self.get_cpt(context)
        if dist is None or len(dist) == 0:
            return None

        if method == "greedy":
            # Deterministic tie-break: highest probability, then lexicographically smallest token
            sorted_candidates = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
            return sorted_candidates[0][0]

        # Sampling mode: inverse-CDF over deterministically sorted items
        r = (rng.random() if rng is not None else random.random())
        cumulative = 0.0
        sorted_items = sorted(dist.items(), key=lambda kv: kv[0])
        for word, prob in sorted_items:
            cumulative += prob
            if r <= cumulative + 1e-12:
                return word
        return sorted_items[-1][0]

    def generate(
        self,
        method: Literal["greedy", "sampling"] = "greedy",
        max_length: int = 50,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate a token sequence starting with <START>.

        Args:
            method: 'greedy' or 'sampling'.
            max_length: Maximum number of generated tokens before forced stop.
            seed: Seed for random.Random (sampling mode only).

        Returns:
            GenerationResult dataclass.
        """
        rng = random.Random(seed) if seed is not None else random.Random()
        tokens = [START_TOKEN]
        visited_contexts: list[str] = [START_TOKEN]
        cycle_detected = False
        cycle_tokens: list[str] | None = None

        for _ in range(max_length):
            context = tokens[-1]
            next_token = self.predict_next(context, method=method, rng=rng)

            if next_token is None:
                # Unseen context encountered
                break

            tokens.append(next_token)

            if next_token == END_TOKEN:
                break

            if method == "greedy":
                # In first-order greedy decoding, if context has been visited before,
                # the deterministic transition causes an infinite periodic loop.
                if next_token in visited_contexts:
                    cycle_detected = True
                    cycle_start_idx = visited_contexts.index(next_token)
                    cycle_tokens = visited_contexts[cycle_start_idx:]
                    break
                visited_contexts.append(next_token)

        # Check if the generated word sequence matches any training sentence verbatim
        word_tokens = [t for t in tokens if t not in (START_TOKEN, END_TOKEN)]
        sentence_str = " ".join(word_tokens)
        is_verbatim = sentence_str in self.training_sentences

        return GenerationResult(
            tokens=tokens,
            text=sentence_str,
            method=method,
            seed=seed,
            cycle_detected=cycle_detected,
            cycle_tokens=cycle_tokens,
            terminated_with_end=(len(tokens) > 0 and tokens[-1] == END_TOKEN),
            num_tokens=len(tokens),
            is_in_training_corpus=is_verbatim,
        )

    def sentence_probability(
        self, sentence: str | list[str]
    ) -> SentenceProbabilityResult:
        """Compute joint probability and log-probability via chain-rule factorisation.

        Args:
            sentence: Raw string sentence or token list with <START> and <END>.

        Returns:
            SentenceProbabilityResult dataclass.
        """
        if isinstance(sentence, str):
            tokens = tokenize_sentence(sentence, order=1)
            raw_text = sentence.strip().lower()
        else:
            tokens = list(sentence)
            word_tokens = [t for t in tokens if t not in (START_TOKEN, END_TOKEN)]
            raw_text = " ".join(word_tokens)

        joint_prob = 1.0
        log_prob = 0.0
        is_zero = False
        factors: list[dict[str, Any]] = []

        for i in range(len(tokens) - 1):
            prev_tok = tokens[i]
            next_tok = tokens[i + 1]
            dist = self.get_cpt(prev_tok)
            p = 0.0 if dist is None else dist.get(next_tok, 0.0)

            step_log_p = math.log(p) if p > 0.0 else float("-inf")
            factors.append({
                "step": i + 1,
                "context": prev_tok,
                "token": next_tok,
                "probability": p,
                "log_probability": step_log_p,
            })

            joint_prob *= p
            if p > 0.0 and not is_zero:
                log_prob += math.log(p)
            else:
                is_zero = True
                log_prob = float("-inf")

        return SentenceProbabilityResult(
            sentence=raw_text,
            tokens=tokens,
            joint_probability=joint_prob,
            log_probability=log_prob,
            is_zero_probability=is_zero,
            factors=factors,
        )

    def check_normalization(self) -> dict[str, Any]:
        """Check sum of probabilities for all observed contexts.

        Returns:
            Dict mapping context to its probability sum and maximum absolute deviation from 1.0.
        """
        sums: dict[str, float] = {}
        max_deviation = 0.0
        for ctx, dist in self.cpt.items():
            s = sum(dist.values())
            sums[ctx] = s
            dev = abs(s - 1.0)
            if dev > max_deviation:
                max_deviation = dev

        return {
            "context_sums": sums,
            "max_deviation": max_deviation,
            "all_contexts_normalized": max_deviation < 1e-12,
        }


class SecondOrderMarkovModel:
    """Second-order autoregressive Markov language model: P(w_t | w_{t-2}, w_{t-1}).

    Independence assumption:
        P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-2}, X_{t-1})
    Chain-rule factorisation:
        P(X_1, ..., X_T) = P(X_1, X_2) * prod_{t=3}^T P(X_t | X_{t-2}, X_{t-1})
    With padding [<START>, <START>, w1, ..., wT, <END>],
    every real token w_t is conditioned on the previous 2 tokens.
    """

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], Counter[str]] = {}
        self.cpt: dict[tuple[str, str], dict[str, float]] = {}
        self.vocab: set[str] = set()
        self.observed_contexts: set[tuple[str, str]] = set()
        self.training_sentences: list[str] = []

    def train(self, sentences: list[str] | None = None) -> None:
        """Train second-order Markov model by counting trigram transitions."""
        self.counts.clear()
        self.cpt.clear()
        self.vocab.clear()
        self.observed_contexts.clear()

        corpus_text = CANONICAL_CORPUS if sentences is None else sentences
        self.training_sentences = list(corpus_text)

        tokenized = tokenize_corpus(self.training_sentences, order=2)
        for seq in tokenized:
            for i in range(len(seq) - 2):
                ctx: tuple[str, str] = (seq[i], seq[i + 1])
                next_tok = seq[i + 2]
                self.vocab.add(ctx[0])
                self.vocab.add(ctx[1])
                self.vocab.add(next_tok)
                if ctx not in self.counts:
                    self.counts[ctx] = Counter()
                self.counts[ctx][next_tok] += 1
                self.observed_contexts.add(ctx)

        # Normalize counts to build CPT
        for ctx, next_counts in self.counts.items():
            total = sum(next_counts.values())
            self.cpt[ctx] = {}
            for next_tok, cnt in sorted(next_counts.items()):
                self.cpt[ctx][next_tok] = cnt / total

    def get_counts(self, context: tuple[str, str]) -> dict[str, int] | None:
        """Return raw transition counts for an observed context, or None if unseen."""
        if context not in self.observed_contexts:
            return None
        return dict(self.counts.get(context, {}))

    def get_cpt(self, context: tuple[str, str]) -> dict[str, float] | None:
        """Return conditional probability distribution for an observed context (or None)."""
        if context not in self.observed_contexts:
            return None
        return dict(self.cpt.get(context, {}))

    def get_vocabulary(self) -> list[str]:
        """Return sorted list of all unique vocabulary tokens."""
        return sorted(self.vocab)

    def get_observed_contexts(self) -> list[tuple[str, str]]:
        """Return sorted list of all observed 2-token context tuples."""
        return sorted(self.observed_contexts)

    def predict_next(
        self,
        context: tuple[str, str],
        method: Literal["greedy", "sampling"] = "greedy",
        rng: random.Random | None = None,
    ) -> str | None:
        """Predict the next token given a 2-token context.

        Args:
            context: Tuple of (w_{t-2}, w_{t-1}).
            method: 'greedy' or 'sampling'.
            rng: random.Random instance for reproducible sampling.

        Returns:
            Next token string, or None if context is unseen.
        """
        dist = self.get_cpt(context)
        if dist is None or len(dist) == 0:
            return None

        if method == "greedy":
            sorted_candidates = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
            return sorted_candidates[0][0]

        r = (rng.random() if rng is not None else random.random())
        cumulative = 0.0
        sorted_items = sorted(dist.items(), key=lambda kv: kv[0])
        for word, prob in sorted_items:
            cumulative += prob
            if r <= cumulative + 1e-12:
                return word
        return sorted_items[-1][0]

    def generate(
        self,
        method: Literal["greedy", "sampling"] = "greedy",
        max_length: int = 50,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate a token sequence starting with [<START>, <START>].

        Args:
            method: 'greedy' or 'sampling'.
            max_length: Maximum number of generated tokens before forced stop.
            seed: Seed for random.Random (sampling mode only).

        Returns:
            GenerationResult dataclass.
        """
        rng = random.Random(seed) if seed is not None else random.Random()
        tokens = [START_TOKEN, START_TOKEN]
        visited_contexts: list[tuple[str, str]] = [(START_TOKEN, START_TOKEN)]
        cycle_detected = False
        cycle_tokens: list[str] | None = None

        for _ in range(max_length):
            context = (tokens[-2], tokens[-1])
            next_token = self.predict_next(context, method=method, rng=rng)

            if next_token is None:
                # Unseen context
                break

            tokens.append(next_token)

            if next_token == END_TOKEN:
                break

            if method == "greedy":
                next_context = (context[1], next_token)
                if next_context in visited_contexts:
                    cycle_detected = True
                    cycle_start_idx = visited_contexts.index(next_context)
                    cycle_tokens = [f"{c[0]} {c[1]}" for c in visited_contexts[cycle_start_idx:]]
                    break
                visited_contexts.append(next_context)

        word_tokens = [t for t in tokens if t not in (START_TOKEN, END_TOKEN)]
        sentence_str = " ".join(word_tokens)
        is_verbatim = sentence_str in self.training_sentences

        return GenerationResult(
            tokens=tokens,
            text=sentence_str,
            method=method,
            seed=seed,
            cycle_detected=cycle_detected,
            cycle_tokens=cycle_tokens,
            terminated_with_end=(len(tokens) > 0 and tokens[-1] == END_TOKEN),
            num_tokens=len(tokens),
            is_in_training_corpus=is_verbatim,
        )

    def sentence_probability(
        self, sentence: str | list[str]
    ) -> SentenceProbabilityResult:
        """Compute joint probability and log-probability via second-order chain rule."""
        if isinstance(sentence, str):
            tokens = tokenize_sentence(sentence, order=2)
            raw_text = sentence.strip().lower()
        else:
            tokens = list(sentence)
            word_tokens = [t for t in tokens if t not in (START_TOKEN, END_TOKEN)]
            raw_text = " ".join(word_tokens)

        joint_prob = 1.0
        log_prob = 0.0
        is_zero = False
        factors: list[dict[str, Any]] = []

        for i in range(len(tokens) - 2):
            ctx: tuple[str, str] = (tokens[i], tokens[i + 1])
            next_tok = tokens[i + 2]
            dist = self.get_cpt(ctx)
            p = 0.0 if dist is None else dist.get(next_tok, 0.0)

            step_log_p = math.log(p) if p > 0.0 else float("-inf")
            factors.append({
                "step": i + 1,
                "context": list(ctx),
                "token": next_tok,
                "probability": p,
                "log_probability": step_log_p,
            })

            joint_prob *= p
            if p > 0.0 and not is_zero:
                log_prob += math.log(p)
            else:
                is_zero = True
                log_prob = float("-inf")

        return SentenceProbabilityResult(
            sentence=raw_text,
            tokens=tokens,
            joint_probability=joint_prob,
            log_probability=log_prob,
            is_zero_probability=is_zero,
            factors=factors,
        )

    def check_normalization(self) -> dict[str, Any]:
        """Check sum of probabilities for all observed contexts."""
        sums: dict[str, float] = {}
        max_deviation = 0.0
        for ctx, dist in self.cpt.items():
            ctx_key = f"{ctx[0]} {ctx[1]}"
            s = sum(dist.values())
            sums[ctx_key] = s
            dev = abs(s - 1.0)
            if dev > max_deviation:
                max_deviation = dev

        return {
            "context_sums": sums,
            "max_deviation": max_deviation,
            "all_contexts_normalized": max_deviation < 1e-12,
        }
