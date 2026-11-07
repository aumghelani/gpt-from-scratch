"""Autoregressive generation: streaming generator with optional KV cache."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn

from gpt_scratch.sample.sampling import sample_next


def _config_block_size(model: nn.Module) -> int:
    return model.config.block_size  # type: ignore[attr-defined,no-any-return]


@torch.no_grad()
def stream_generate(
    model: nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_ids: Sequence[int] = (),
    seed: int | None = None,
    use_cache: bool = True,
    device: str | torch.device | None = None,
) -> Iterator[int]:
    """Yield new token ids one at a time. A stop id ends generation and is *not* yielded.

    With `use_cache` the total context is capped at block_size (generation stops when full);
    without it the context window slides, so output length is unbounded.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    block = _config_block_size(model)
    device = device or next(model.parameters()).device
    generator = torch.Generator().manual_seed(torch.seed() if seed is None else seed)
    stops = frozenset(stop_ids)
    was_training = model.training
    model.eval()
    try:
        tokens = list(prompt_ids)[-(block - 1) :] if use_cache else list(prompt_ids)
        past = None
        logits = None
        for _ in range(max_new_tokens):
            if use_cache:
                if len(tokens) >= block:
                    return
                feed = tokens if past is None else tokens[-1:]
                out, past = model.forward_cached(torch.tensor([feed], device=device), past)
                logits = out[:, -1, :]
            else:
                window = torch.tensor([tokens[-block:]], device=device)
                logits = model(window)[0][:, -1, :]
            history = [torch.tensor(tokens)] if repetition_penalty != 1.0 else None
            nxt = int(
                sample_next(
                    logits.cpu(), temperature, top_k, top_p, repetition_penalty, history, generator
                ).item()
            )
            if nxt in stops:
                return
            tokens.append(nxt)
            yield nxt
    finally:
        model.train(was_training)


def generate(model: nn.Module, prompt_ids: Sequence[int], max_new_tokens: int, **kwargs: object) -> list[int]:
    """Eagerly collect `stream_generate` into a list."""
    return list(stream_generate(model, prompt_ids, max_new_tokens, **kwargs))  # type: ignore[arg-type]
