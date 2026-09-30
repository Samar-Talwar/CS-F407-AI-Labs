# CS F407 Lab, Week 8 | Author: Samar Talwar | Not licensed for reuse or submission by others.
"""Dataset definition and tokenization helpers for the Week 8 language model laboratory."""

START_TOKEN = "<START>"
END_TOKEN = "<END>"

CANONICAL_CORPUS: list[str] = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenize_sentence(sentence: str, order: int = 1) -> list[str]:
    """Tokenize a raw sentence into lowercase words with <START> and <END> padding.

    Args:
        sentence: Raw input sentence string.
        order: Markov model order (1 for first-order, 2 for second-order).
            Padding adds `order` <START> tokens at the start and one <END> token at the end.

    Returns:
        List of padded token strings.
    """
    words = sentence.strip().lower().split()
    start_padding = [START_TOKEN] * max(1, order)
    return start_padding + words + [END_TOKEN]


def tokenize_corpus(
    sentences: list[str] | None = None, order: int = 1
) -> list[list[str]]:
    """Tokenize an entire corpus of sentences.

    Args:
        sentences: List of sentence strings. Defaults to `CANONICAL_CORPUS`.
        order: Markov model order (1 or 2).

    Returns:
        List of tokenized sentences.
    """
    corpus = CANONICAL_CORPUS if sentences is None else sentences
    return [tokenize_sentence(s, order=order) for s in corpus]
