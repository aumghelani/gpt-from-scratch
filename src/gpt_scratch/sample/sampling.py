"""Pure logit-processing functions and the next-token sampler. No function mutates its input."""

from __future__ import annotations

from collections.abc import Sequence

import torch

_NEG_INF = float("-inf")


def top_k_filter(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    """Keep the k largest logits per row (ties at the threshold kept); others -> -inf."""
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, _NEG_INF)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose probability mass >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # drop a token if the mass *before* it already reached p (top token is always kept)
    drop_sorted = (cumulative - torch.softmax(sorted_logits, dim=-1)) >= p
    drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
    return logits.masked_fill(drop, _NEG_INF)


def apply_repetition_penalty(
    logits: torch.Tensor, history: Sequence[torch.Tensor], penalty: float
) -> torch.Tensor:
    """Discourage tokens already generated (CTRL-style): shrink positive, grow negative logits."""
    if penalty == 1.0:
        return logits
    out = logits.clone()
    for row, seen in enumerate(history):
        if seen.numel() == 0:
            continue
        ids = torch.unique(seen)
        values = out[row, ids]
        out[row, ids] = torch.where(values > 0, values / penalty, values * penalty)
    return out


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    history: Sequence[torch.Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id per row from (batch, vocab) logits -> (batch, 1)."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if history is not None:
        logits = apply_repetition_penalty(logits, history, repetition_penalty)
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = top_p_filter(top_k_filter(logits / temperature, top_k), top_p)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
