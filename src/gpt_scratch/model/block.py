"""Pre-norm Transformer block: x + attn(norm(x)), then x + ffn(norm(x))."""

from __future__ import annotations

import torch
from torch import nn

from gpt_scratch.model.attention import CausalSelfAttention
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))
