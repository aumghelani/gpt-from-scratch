"""Normalisation layers: RMSNorm and a config-driven factory."""

from __future__ import annotations

import torch
from torch import nn

from gpt_scratch.model.config import GPTConfig


class RMSNorm(nn.Module):
    """Root-mean-square norm (no mean-centering, no bias): x / rms(x) * weight."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(config: GPTConfig) -> nn.Module:
    if config.norm == "rmsnorm":
        return RMSNorm(config.n_embd)
    return nn.LayerNorm(config.n_embd, bias=config.bias)
