"""Chat template. Role/turn markers are special tokens; message *content* is never parsed for them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gpt_scratch.tokenizer.bpe import BPETokenizer

CHAT_SPECIALS = ("<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
ROLES = ("system", "user", "assistant")
END = "<|end|>"


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")


def _marker(role: str) -> str:
    return f"<|{role}|>"


def render_chat(messages: Sequence[Message], add_generation_prompt: bool = False) -> str:
    """Human-readable form of the template (for display and tests)."""
    if not messages:
        raise ValueError("messages must not be empty")
    text = "".join(f"{_marker(m.role)}{m.content}{END}" for m in messages)
    return text + (_marker("assistant") if add_generation_prompt else "")


def encode_chat(
    tok: BPETokenizer, messages: Sequence[Message], add_generation_prompt: bool = False
) -> tuple[list[int], list[bool]]:
    """Token ids plus a mask that is True exactly where the model should learn to predict.

    Content is encoded with `allow_special=False`, so user text containing "<|end|>" is plain bytes.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    ids: list[int] = []
    mask: list[bool] = []

    def emit(token_ids: list[int], trainable: bool) -> None:
        ids.extend(token_ids)
        mask.extend([trainable] * len(token_ids))

    for m in messages:
        learn = m.role == "assistant"
        emit([tok.special_id(_marker(m.role))], False)
        emit(tok.encode(m.content, allow_special=False), learn)
        emit([tok.special_id(END)], learn)
    if add_generation_prompt:
        emit([tok.special_id(_marker("assistant"))], False)
    return ids, mask
