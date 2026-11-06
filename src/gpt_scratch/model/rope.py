"""Rotary position embeddings (RoPE): rotate (q, k) pairs by a position-dependent angle."""

from __future__ import annotations

import torch


def build_rope_cache(
    head_size: int, max_len: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (max_len, head_size // 2)."""
    if head_size % 2 != 0:
        raise ValueError("RoPE requires an even head_size")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Rotate `x` of shape (B, heads, T, head_size); position i uses table row offset + i."""
    seq = x.size(-2)
    if offset + seq > cos.size(0):
        raise ValueError(f"positions up to {offset + seq} exceed RoPE cache length {cos.size(0)}")
    c = cos[offset : offset + seq].to(x.dtype)
    s = sin[offset : offset + seq].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
