"""Deterministic validation-loss estimate."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from gpt_scratch.data.loader import get_batch


def estimate_loss(
    model: nn.Module,
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    n_batches: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> float:
    """Mean loss over `n_batches` random windows; eval mode, no grad, mode restored afterwards."""
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            losses = []
            for _ in range(n_batches):
                x, y = get_batch(data, batch_size, block_size, generator, device)
                _, loss = model(x, y)
                losses.append(loss.item())
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)
