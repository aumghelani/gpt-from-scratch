"""Character-level tokenizer, the baseline used in the video."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharTokenizer:
    """Maps every distinct character seen at training time to an integer id."""

    chars: tuple[str, ...]

    @classmethod
    def train(cls, text: str) -> CharTokenizer:
        if not text:
            raise ValueError("cannot train a tokenizer on empty text")
        return cls(chars=tuple(sorted(set(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def _index(self) -> dict[str, int]:
        return {ch: i for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        index = self._index()
        return [index[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "char", "chars": list(self.chars)}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chars=tuple(data["chars"]))
