"""Pre-norm Transformer block: x + attn(norm(x)), then x + ffn(norm(x))."""

from __future__ import annotations

import torch
from torch import nn

from gpt_scratch.model.attention import KV, CausalSelfAttention
from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.ffn import FeedForward
from gpt_scratch.model.norm import make_norm


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.norm2 = make_norm(config)
        self.ffn = FeedForward(config)

    def forward(
        self, x: torch.Tensor, past: KV | None = None, return_kv: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, KV]:
        attended = self.attn(self.norm1(x), past=past, return_kv=return_kv)
        present = None
        if return_kv:
            attended, present = attended
        x = x + attended
        x = x + self.ffn(self.norm2(x))
        return (x, present) if return_kv else x
