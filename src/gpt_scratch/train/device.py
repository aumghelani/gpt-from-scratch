"""Reproducibility and device selection helpers."""

from __future__ import annotations

import random

import numpy as np
import torch

_VALID = ("auto", "cpu", "mps", "cuda")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(preferred: str = "auto") -> str:
    if preferred not in _VALID:
        raise ValueError(f"device must be one of {_VALID}, got {preferred!r}")
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    if preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
