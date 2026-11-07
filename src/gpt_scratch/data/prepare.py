"""Turn a raw text file into tokenizer + train/val shards."""

from __future__ import annotations

from pathlib import Path

from gpt_scratch.chat.template import CHAT_SPECIALS
from gpt_scratch.data.clean import clean_text
from gpt_scratch.data.shards import write_shards
from gpt_scratch.tokenizer import Tokenizer
from gpt_scratch.tokenizer.bpe import BPETokenizer
from gpt_scratch.tokenizer.char import CharTokenizer

TOKENIZERS = ("char", "bpe")


def prepare_dataset(
    text_path: str | Path,
    out_dir: str | Path,
    tokenizer: str = "char",
    vocab_size: int = 512,
    val_fraction: float = 0.1,
) -> dict[str, int]:
    """Clean `text_path`, train a tokenizer on it, and write shards + tokenizer.json to `out_dir`."""
    if tokenizer not in TOKENIZERS:
        raise ValueError(f"tokenizer must be one of {TOKENIZERS}")
    text = clean_text(Path(text_path).read_text(encoding="utf-8"))
    tok: Tokenizer
    if tokenizer == "char":
        tok = CharTokenizer.train(text)
    else:
        tok = BPETokenizer.train(text, vocab_size=vocab_size, special_tokens=CHAT_SPECIALS)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")
    # corpus text is data, never control: literal '<|end|>' etc. must not become special ids
    ids = tok.encode(text) if isinstance(tok, CharTokenizer) else tok.encode(text, allow_special=False)
    return write_shards(ids, out, vocab_size=tok.vocab_size, val_fraction=val_fraction)
