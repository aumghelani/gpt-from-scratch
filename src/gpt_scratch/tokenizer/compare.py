"""Compression measurements for comparing tokenizers fairly (same held-out text, bytes per token)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Compression:
    name: str
    vocab_size: int
    tokens: int
    bytes_per_token: float


def measure(name: str, vocab_size: int, encode: Callable[[str], list[int]], text: str) -> Compression:
    """Higher bytes/token = better compression = fewer tokens for the model to process."""
    if not text:
        raise ValueError("text must not be empty")
    tokens = len(encode(text))
    if tokens == 0:
        raise ValueError(f"{name} produced no tokens")
    return Compression(name, vocab_size, tokens, len(text.encode("utf-8")) / tokens)


def split_for_comparison(text: str, test_fraction: float = 0.1) -> tuple[str, str]:
    """(train, test): learn merges on the head, measure on the unseen tail."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    cut = len(text) - max(1, round(len(text) * test_fraction))
    return text[:cut], text[cut:]
