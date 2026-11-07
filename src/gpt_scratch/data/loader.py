"""Random-batch loader: samples `block_size` windows and their next-token targets."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

BatchSource = Callable[[int, int, torch.Generator, "str | torch.device"], tuple[torch.Tensor, torch.Tensor]]


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(x, y)` with `y[i] = x[i]` shifted one token to the right."""
    if len(data) <= block_size:
        raise ValueError(f"need more than block_size={block_size} tokens, have {len(data)}")
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[s : s + block_size].astype(np.int64)) for s in starts.tolist()])
    y = torch.stack(
        [torch.from_numpy(data[s + 1 : s + 1 + block_size].astype(np.int64)) for s in starts.tolist()]
    )
    return x.to(device), y.to(device)


def sample_batch(
    source: np.ndarray | BatchSource,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a batch from either a flat token array or a custom batch-source callable."""
    if callable(source):
        return source(batch_size, block_size, generator, device)
    return get_batch(source, batch_size, block_size, generator, device)
