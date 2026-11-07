"""Atomic checkpoint save/load including optimizer and every RNG stream."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gpt_scratch.model.config import GPTConfig


@dataclass(frozen=True)
class CheckpointState:
    step: int
    config: GPTConfig
    extra: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: GPTConfig,
    generator: torch.Generator | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config.to_dict(),
        "extra": dict(extra or {}),
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "generator": generator.get_state() if generator is not None else None,
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    generator: torch.Generator | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointState:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    rng = payload["rng"]
    torch.set_rng_state(rng["torch"].cpu())
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    if generator is not None and rng["generator"] is not None:
        generator.set_state(rng["generator"].cpu())
    return CheckpointState(
        step=payload["step"], config=GPTConfig.from_dict(payload["config"]), extra=payload["extra"]
    )
