"""Immutable model configuration with validation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

POS_ENCODINGS = ("learned", "rope")
NORMS = ("layernorm", "rmsnorm")
FFNS = ("gelu", "swiglu")


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    pos_encoding: str = "learned"
    norm: str = "layernorm"
    ffn: str = "gelu"
    bias: bool = True
    tie_weights: bool = True

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.pos_encoding not in POS_ENCODINGS:
            raise ValueError(f"pos_encoding must be one of {POS_ENCODINGS}")
        if self.norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if self.ffn not in FFNS:
            raise ValueError(f"ffn must be one of {FFNS}")
        if self.pos_encoding == "rope" and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("rope requires an even head_size")

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    def with_changes(self, **changes: Any) -> GPTConfig:
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPTConfig:
        return cls(**data)
