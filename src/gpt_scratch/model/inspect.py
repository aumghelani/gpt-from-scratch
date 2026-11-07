"""Introspection: recover per-layer attention probability maps from a trained model."""

from __future__ import annotations

import math

import torch

from gpt_scratch.model.gpt import GPT
from gpt_scratch.model.rope import apply_rope


@torch.no_grad()
def attention_weights(model: GPT, idx: torch.Tensor) -> list[torch.Tensor]:
    """Return one (heads, T, T) tensor per layer for a single sequence `idx` of shape (1, T).

    Works whichever attention path (manual or fused) the model uses, by capturing each attention
    module's input and recomputing the softmax(QK^T/sqrt(d)) probabilities from its weights.
    """
    if idx.dim() != 2 or idx.size(0) != 1:
        raise ValueError("attention_weights expects a single sequence of shape (1, T)")
    inputs: list[torch.Tensor] = []
    handles = [b.attn.register_forward_pre_hook(lambda _m, args: inputs.append(args[0])) for b in model.blocks]
    was_training = model.training
    model.eval()
    try:
        model(idx)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    seq = idx.size(1)
    maps: list[torch.Tensor] = []
    for block, x in zip(model.blocks, inputs, strict=True):
        attn = block.attn
        q, k, _ = attn.qkv(x).split(x.size(-1), dim=2)
        q, k = (t.view(1, seq, attn.n_head, attn.head_size).transpose(1, 2) for t in (q, k))
        if attn.rope_cos is not None:
            q = apply_rope(q, attn.rope_cos, attn.rope_sin)
            k = apply_rope(k, attn.rope_cos, attn.rope_sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_size)
        scores = scores.masked_fill(~attn.mask[:, :, :seq, :seq], float("-inf"))
        maps.append(torch.softmax(scores, dim=-1)[0])
    return maps
