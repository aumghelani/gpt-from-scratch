"""Tokenizers: character-level baseline and byte-level BPE, with a type-dispatching loader."""

from __future__ import annotations

import json
from pathlib import Path

from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.char import CharTokenizer

Tokenizer = CharTokenizer | BPETokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a saved tokenizer, choosing the class from the file's `type` field."""
    kind = json.loads(Path(path).read_text(encoding="utf-8")).get("type")
    if kind == "char":
        return CharTokenizer.load(path)
    if kind == "bpe":
        return BPETokenizer.load(path)
    raise ValueError(f"unknown tokenizer type: {kind!r}")
