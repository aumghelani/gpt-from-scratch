"""One optimizer step with gradient accumulation and global-norm clipping."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StepStats:
    loss: float  # mean loss over the micro-batches
    grad_norm: float  # global gradient norm *before* clipping


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    micro_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    grad_clip: float,
) -> StepStats:
    """Accumulate gradients over `micro_batches`, clip (if grad_clip > 0), then step."""
    if not micro_batches:
        raise ValueError("micro_batches must not be empty")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    n = len(micro_batches)
    total_loss = 0.0
    for x, y in micro_batches:
        _, loss = model(x, y)
        (loss / n).backward()
        total_loss += loss.item()
    max_norm = grad_clip if grad_clip > 0 else math.inf
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    return StepStats(loss=total_loss / n, grad_norm=float(grad_norm))
