"""Byte-level Byte-Pair-Encoding tokenizer (trainer + decoder).

Ids 0..255 are raw bytes; id 256+i is the i-th learned merge.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

# GPT-2-style pre-split: contractions, words, numbers, punctuation, whitespace.
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""", re.UNICODE
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
    special_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB + len(self.merges) + len(self.special_tokens)

    def special_id(self, token: str) -> int:
        if token not in self.special_tokens:
            raise KeyError(token)
        return BASE_VOCAB + len(self.merges) + self.special_tokens.index(token)

    @classmethod
    def train(
        cls, text: str, vocab_size: int, special_tokens: tuple[str, ...] = ()
    ) -> BPETokenizer:
        if vocab_size < BASE_VOCAB:
            raise ValueError(f"vocab_size must be >= {BASE_VOCAB}")
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special_tokens must be unique")
        words: Counter[tuple[int, ...]] = Counter(
            tuple(chunk.encode("utf-8")) for chunk in SPLIT_PATTERN.findall(text)
        )
        merges: list[Pair] = []
        current = dict(words)
        merge_budget = vocab_size - len(special_tokens)
        while BASE_VOCAB + len(merges) < merge_budget:
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
        return cls(merges=tuple(merges), special_tokens=tuple(special_tokens))

    @cached_property
    def _byte_table(self) -> list[bytes]:
        """id -> bytes for all regular tokens; built once per tokenizer instance."""
        table = [bytes([i]) for i in range(BASE_VOCAB)]
        for left, right in self.merges:
            table.append(table[left] + table[right])
        return table

    def token_bytes(self, token_id: int) -> bytes:
        """Raw bytes of one token (special tokens yield their literal text)."""
        regular = BASE_VOCAB + len(self.merges)
        if token_id >= regular:
            return self.special_tokens[token_id - regular].encode("utf-8")
        return self._byte_table[token_id]

    def decode(self, ids: list[int]) -> str:
        table = self._byte_table
        n_regular = len(table)
        specials = {n_regular + j: tok.encode("utf-8") for j, tok in enumerate(self.special_tokens)}
        return b"".join(table[i] if i < n_regular else specials[i] for i in ids).decode(
            "utf-8", errors="replace"
        )

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

    def _encode_plain(self, text: str, ranks: dict[Pair, int]) -> list[int]:
        cache: dict[bytes, list[int]] = {}
        out: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            raw = chunk.encode("utf-8")
            if raw not in cache:
                cache[raw] = self._encode_chunk(raw, ranks)
            out.extend(cache[raw])
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        ranks = self._ranks()
        if not (allow_special and self.special_tokens):
            return self._encode_plain(text, ranks)
        pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        out: list[int] = []
        for piece in re.split(pattern, text):
            if piece in self.special_tokens:
                out.append(self.special_id(piece))
            elif piece:
                out.extend(self._encode_plain(piece, ranks))
        return out

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "bpe",
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": list(self.special_tokens),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "bpe":
            raise ValueError(f"not a BPE tokenizer file: type={data.get('type')!r}")
        return cls(
            merges=tuple((a, b) for a, b in data["merges"]),
            special_tokens=tuple(data["special_tokens"]),
        )
