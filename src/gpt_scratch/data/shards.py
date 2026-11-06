"""Write/read tokenized corpora as flat uint16 binary shards (train/val)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MAX_UINT16_VOCAB = 1 << 16


def write_shards(
    ids: Sequence[int] | np.ndarray,
    out_dir: str | Path,
    vocab_size: int,
    val_fraction: float = 0.1,
) -> dict[str, int]:
    """Split token ids contiguously into train/val and persist them."""
    if vocab_size > MAX_UINT16_VOCAB:
        raise ValueError(f"vocab_size {vocab_size} does not fit in uint16")
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= vocab_size):
        raise ValueError("token ids fall outside [0, vocab_size)")
    n_val = int(arr.size * val_fraction)
    n_train = arr.size - n_val
    if n_train < 2 or n_val < 1:
        raise ValueError("not enough tokens to create train and val splits")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr[:n_train].astype(np.uint16).tofile(out / "train.bin")
    arr[n_train:].astype(np.uint16).tofile(out / "val.bin")
    meta = {"vocab_size": vocab_size, "n_train": n_train, "n_val": n_val}
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def load_shard(path: str | Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")
