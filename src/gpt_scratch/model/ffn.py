"""Position-wise feed-forward blocks: GELU MLP and SwiGLU."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from gpt_scratch.model.config import GPTConfig


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.kind = config.ffn
        n = config.n_embd
        if self.kind == "gelu":
            self.hidden = 4 * n
            self.up = nn.Linear(n, self.hidden, bias=config.bias)
            self.down = nn.Linear(self.hidden, n, bias=config.bias)
        else:  # swiglu: 3 matrices, so use ~8/3 n hidden to match the 2-matrix 4n parameter count
            self.hidden = _round_up(int(8 * n / 3), 8)
            self.gate = nn.Linear(n, self.hidden, bias=config.bias)
            self.up = nn.Linear(n, self.hidden, bias=config.bias)
            self.down = nn.Linear(self.hidden, n, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "gelu":
            return self.drop(self.down(F.gelu(self.up(x))))
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))
