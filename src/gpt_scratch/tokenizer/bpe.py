"""Byte-level Byte-Pair-Encoding tokenizer (trainer + decoder).

Ids 0..255 are raw bytes; id 256+i is the i-th learned merge.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass

# GPT-2-style pre-split: contractions, words, numbers, punctuation, whitespace.
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""", re.UNICODE
)
BASE_VOCAB = 256

Pair = tuple[int, int]


def _merge_ids(ids: tuple[int, ...], pair: Pair, new_id: int) -> tuple[int, ...]:
    """Return a new tuple with every non-overlapping occurrence of `pair` replaced."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def _count_pairs(words: dict[tuple[int, ...], int]) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for word, freq in words.items():
        for pair in itertools.pairwise(word):
            counts[pair] += freq
    return counts


@dataclass(frozen=True)
class BPETokenizer:
    merges: tuple[Pair, ...]

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> BPETokenizer:
        if vocab_size < BASE_VOCAB:
            raise ValueError(f"vocab_size must be >= {BASE_VOCAB}")
        words: Counter[tuple[int, ...]] = Counter(
            tuple(chunk.encode("utf-8")) for chunk in SPLIT_PATTERN.findall(text)
        )
        merges: list[Pair] = []
        current = dict(words)
        while BASE_VOCAB + len(merges) < vocab_size:
            counts = _count_pairs(current)
            if not counts:
                break
            best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            if counts[best] < 2:
                break
            new_id = BASE_VOCAB + len(merges)
            merges.append(best)
            merged: dict[tuple[int, ...], int] = {}
            for word, freq in current.items():
                new_word = _merge_ids(word, best, new_id)
                merged[new_word] = merged.get(new_word, 0) + freq
            current = merged
        return cls(merges=tuple(merges))

    def _token_bytes(self) -> list[bytes]:
        table = [bytes([i]) for i in range(BASE_VOCAB)]
        for left, right in self.merges:
            table.append(table[left] + table[right])
        return table

    def decode(self, ids: list[int]) -> str:
        table = self._token_bytes()
        return b"".join(table[i] for i in ids).decode("utf-8", errors="replace")

    def _ranks(self) -> dict[Pair, int]:
        return {pair: rank for rank, pair in enumerate(self.merges)}

    def _encode_chunk(self, chunk: bytes, ranks: dict[Pair, int]) -> list[int]:
        ids = list(chunk)
        while len(ids) >= 2:
            candidates = {pair: ranks[pair] for pair in itertools.pairwise(ids) if pair in ranks}
            if not candidates:
                break
            best = min(candidates, key=candidates.__getitem__)
            ids = list(_merge_ids(tuple(ids), best, BASE_VOCAB + ranks[best]))
        return ids

    def encode(self, text: str) -> list[int]:
        ranks = self._ranks()
        cache: dict[bytes, list[int]] = {}
        out: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            raw = chunk.encode("utf-8")
            if raw not in cache:
                cache[raw] = self._encode_chunk(raw, ranks)
            out.extend(cache[raw])
        return out
