"""Learning-rate schedule: linear warmup, then cosine decay to a floor."""

from __future__ import annotations

import math


def lr_at(step: int, *, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if max_lr <= 0 or min_lr < 0 or min_lr > max_lr:
        raise ValueError("require 0 <= min_lr <= max_lr and max_lr > 0")
    if warmup_steps < 1 or warmup_steps >= max_steps:
        raise ValueError("require 1 <= warmup_steps < max_steps")
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)
