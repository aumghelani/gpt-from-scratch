"""Pure functions that compute the real numbers shown by `gpt-scratch tour` (no printing here)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gpt_scratch.model.gpt import GPT
from gpt_scratch.model.inspect import attention_weights
from gpt_scratch.tokenizer import Tokenizer


def token_pieces(tok: Tokenizer, text: str) -> list[tuple[int, str]]:
    """(id, text) for each token of `text`."""
    return [(i, tok.decode([i])) for i in tok.encode(text)]


def param_breakdown(model: GPT) -> dict[str, int]:
    """Parameters grouped by role; the groups always sum to `model.num_params()`."""
    groups = {"token embedding": 0, "position embedding": 0, "attention": 0, "feed-forward": 0, "norms": 0, "output head": 0}
    for name, param in model.named_parameters():
        if name.startswith("tok_emb"):
            key = "token embedding"
        elif name.startswith("pos_emb"):
            key = "position embedding"
        elif ".attn." in name:
            key = "attention"
        elif ".ffn." in name:
            key = "feed-forward"
        elif name.startswith("lm_head"):
            key = "output head"
        else:
            key = "norms"
        groups[key] += param.numel()
    return groups


def random_guess_loss(vocab_size: int) -> float:
    """Cross-entropy of a model that knows nothing: it spreads probability evenly over the vocabulary."""
    return math.log(vocab_size)


@torch.no_grad()
def last_logits(model: GPT, ids: list[int]) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        logits, _ = model(torch.tensor([ids], device=device))
    finally:
        model.train(was_training)
    return logits[0, -1].float().cpu()


def next_token_candidates(model: GPT, tok: Tokenizer, ids: list[int], k: int = 8) -> list[tuple[str, float]]:
    """The model's `k` most likely next tokens with probabilities (temperature 1)."""
    probs = torch.softmax(last_logits(model, ids), dim=-1)
    top = torch.topk(probs, min(k, probs.numel()))
    return [(tok.decode([int(i)]), float(p)) for p, i in zip(top.values, top.indices, strict=True)]


def entropy_bits(probs: torch.Tensor) -> float:
    """Shannon entropy in bits: 0 = certain, log2(n) = a uniform guess among n tokens."""
    nonzero = probs[probs > 0]
    return float(-(nonzero * nonzero.log2()).sum())


def temperature_entropies(logits: torch.Tensor, temperatures: tuple[float, ...]) -> dict[float, float]:
    return {t: entropy_bits(torch.softmax(logits / t, dim=-1)) for t in temperatures}


@dataclass(frozen=True)
class AttentionLink:
    query_position: int
    key_position: int
    weight: float


def attention_links(model: GPT, tok: Tokenizer, ids: list[int], layer: int) -> list[AttentionLink]:
    """For each token, the earlier (or same) token it attends to most, averaged over the layer's heads."""
    if not 0 <= layer < len(model.blocks):
        raise ValueError(f"layer must be in [0, {len(model.blocks)}), got {layer}")
    device = next(model.parameters()).device
    maps = attention_weights(model, torch.tensor([ids], device=device))[layer].mean(dim=0).float().cpu()
    links = []
    for query in range(len(ids)):
        weight, key = maps[query, : query + 1].max(dim=0)
        links.append(AttentionLink(query, int(key), float(weight)))
    return links


@dataclass(frozen=True)
class CacheStep:
    tokens_fed: int
    cached_positions: int
    key_shape: tuple[int, ...]  # (batch, heads, positions, head_size) for layer 0


@torch.no_grad()
def cache_growth(model: GPT, ids: list[int], steps: int) -> list[CacheStep]:
    """Feed the prompt once, then one token at a time (greedy), recording how the K/V cache grows."""
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        feed = ids
        past = None
        rows: list[CacheStep] = []
        for _ in range(steps + 1):
            logits, past = model.forward_cached(torch.tensor([feed], device=device), past)
            keys = past[0][0]
            rows.append(CacheStep(len(feed), keys.size(2), tuple(keys.shape)))
            feed = [int(logits[0, -1].argmax())]
    finally:
        model.train(was_training)
    return rows
