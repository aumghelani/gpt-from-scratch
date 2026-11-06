"""Causal multi-head self-attention with optional RoPE, fused SDPA and a functional KV cache."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from gpt_scratch.model.config import GPTConfig
from gpt_scratch.model.rope import apply_rope, build_rope_cache

KV = tuple[torch.Tensor, torch.Tensor]  # (keys, values), each (batch, heads, seq, head_size)


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
        if config.pos_encoding == "rope":
            cos, sin = build_rope_cache(config.head_size, config.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

    def forward(
        self, x: torch.Tensor, past: KV | None = None, return_kv: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, KV]:
        batch, seq, channels = x.shape
        offset = 0 if past is None else past[0].size(2)
        if offset + seq > self.block_size:
            raise ValueError(f"sequence length {offset + seq} exceeds block_size {self.block_size}")
        q, k, v = self.qkv(x).split(channels, dim=2)
        q, k, v = (
            t.view(batch, seq, self.n_head, self.head_size).transpose(1, 2) for t in (q, k, v)
        )
        if self.rope_cos is not None:
            q = apply_rope(q, self.rope_cos, self.rope_sin, offset)
            k = apply_rope(k, self.rope_cos, self.rope_sin, offset)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        total = k.size(2)
        visible = self.mask[:, :, offset : offset + seq, :total]
        if self.flash:
            dropout_p = self.attn_drop.p if self.training else 0.0
            if offset == 0:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=visible, dropout_p=dropout_p)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            scores = scores.masked_fill(~visible, float("-inf"))
            out = self.attn_drop(torch.softmax(scores, dim=-1)) @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq, channels)
        y = self.resid_drop(self.proj(out))
        return (y, (k, v)) if return_kv else y
