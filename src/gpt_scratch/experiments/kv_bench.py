"""Measure what the KV cache buys: same greedy tokens, how much faster."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from gpt_scratch.sample.generate import generate


@dataclass(frozen=True)
class CacheBench:
    new_tokens: int
    cached_seconds: float
    uncached_seconds: float
    identical: bool

    @property
    def speedup(self) -> float:
        return self.uncached_seconds / self.cached_seconds


def _timed_greedy(
    model: nn.Module, prompt: Sequence[int], new_tokens: int, use_cache: bool, device: str | torch.device
) -> tuple[list[int], float]:
    start = time.perf_counter()
    ids = generate(model, prompt, new_tokens, temperature=0.0, use_cache=use_cache, device=device)
    return ids, time.perf_counter() - start


def benchmark_cache(
    model: nn.Module, prompt_ids: Sequence[int], new_tokens: int, device: str | torch.device
) -> CacheBench:
    """Greedy-decode `new_tokens` with and without the cache; report timings and token equality.

    The total length must fit in block_size so the cached and uncached paths see the same context
    (the uncached path otherwise slides its window and the two are no longer comparable).
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if new_tokens <= 0:
        raise ValueError("new_tokens must be positive")
    block = model.config.block_size  # type: ignore[attr-defined]
    if len(prompt_ids) + new_tokens > block:
        raise ValueError(f"prompt + new tokens ({len(prompt_ids) + new_tokens}) exceeds block_size {block}")
    model.to(device)
    cached, cached_seconds = _timed_greedy(model, prompt_ids, new_tokens, True, device)
    uncached, uncached_seconds = _timed_greedy(model, prompt_ids, new_tokens, False, device)
    return CacheBench(new_tokens, cached_seconds, uncached_seconds, cached == uncached)
