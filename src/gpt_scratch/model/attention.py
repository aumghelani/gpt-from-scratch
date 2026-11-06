"""Causal multi-head self-attention, written out explicitly (as derived in the video)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from gpt_scratch.model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.head_size
        self.block_size = config.block_size
        self.flash = config.flash
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, channels = x.shape
        if seq > self.block_size:
            raise ValueError(f"sequence length {seq} exceeds block_size {self.block_size}")
        q, k, v = self.qkv(x).split(channels, dim=2)
        q, k, v = (
            t.view(batch, seq, self.n_head, self.head_size).transpose(1, 2) for t in (q, k, v)
        )
        if self.flash:
            dropout_p = self.attn_drop.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            scores = scores.masked_fill(~self.mask[:, :, :seq, :seq], float("-inf"))
            out = self.attn_drop(torch.softmax(scores, dim=-1)) @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq, channels)
        return self.resid_drop(self.proj(out))
