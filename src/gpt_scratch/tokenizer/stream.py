"""Incremental decoding so multi-byte characters split across tokens are never emitted half-formed."""

from __future__ import annotations

import codecs
from collections.abc import Callable

from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.char import CharTokenizer

Feed = Callable[[int], str]
Flush = Callable[[], str]


def make_stream_decoder(tok: BPETokenizer | CharTokenizer) -> tuple[Feed, Flush]:
    """Return `(feed, flush)`: `feed(id)` yields newly completed text; `flush()` drains the buffer."""
    if isinstance(tok, CharTokenizer):
        return (lambda i: tok.decode([i])), (lambda: "")
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(token_id: int) -> str:
        return decoder.decode(tok.token_bytes(token_id))

    def flush() -> str:
        return decoder.decode(b"", final=True)

    return feed, flush
